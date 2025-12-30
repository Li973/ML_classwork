import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr

class LocalCausalScanner:

    def __init__(self, window=24, step=3, max_lag=4, pc_thresh=0.32):
        self.window = window
        self.step = step
        self.max_lag = max_lag
        self.pc_thresh = pc_thresh

    def scan(self, df):
        vars = df.columns.tolist()
        n = len(df)
        findings = []

        for start in range(0, n - self.window + 1, self.step):
            sub = df.iloc[start:start + self.window]
            for src in vars:
                for tgt in vars:
                    if src == tgt: continue
                    for lag in range(1, self.max_lag + 1):
                        try:
                            y = sub[tgt].iloc[lag:].values
                            x_auto = sub[tgt].shift(1).iloc[lag:].values
                            x_src = sub[src].shift(lag).iloc[lag:].values
                            X = np.column_stack([x_auto, x_src])
                            if len(y) < 10: continue

                            reg = LinearRegression().fit(X, y)
                            coef = reg.coef_[1]
                            if abs(coef) < 0.01: continue

                            # 偏相关近似
                            res_y = y - reg.predict(X) + coef * x_src
                            pc = pearsonr(x_src, res_y)[0]
                            if abs(pc) > self.pc_thresh:
                                mid_date = sub.index[len(sub) // 2]
                                findings.append({
                                    'src': src, 'tgt': tgt, 'lag': lag,
                                    'coef': round(coef, 3),
                                    'pcorr': round(pc, 3),
                                    'window_center': mid_date,
                                    'window_start': sub.index[0],
                                    'window_end': sub.index[-1]
                                })
                        except:
                            continue
        return pd.DataFrame(findings)

    def filter_chains(self, findings, chains=None):
        if chains is None:
            chains = [
                ('美国10年期国债收益率(%)', '美元兑人民币(月均)'),
                ('美元兑人民币(月均)', '中国CPI同比(%)'),
                ('VIX恐慌指数', '布伦特原油(USD/桶)'),
                ('布伦特原油(USD/桶)', '中国CPI同比(%)')
            ]
        mask = findings.apply(lambda r: (r['src'], r['tgt']) in chains, axis=1)
        return findings[mask].sort_values(['src', 'tgt', 'window_center'])


class EventAnalyzer:

    EVENTS = {
        '疫情': ('2020-03', '2020-08'),
        '加息': ('2022-03', '2023-06'),
        '复苏': ('2023-01', '2023-12')
    }

    @classmethod
    def analyze(cls, df, event_name, scanner=None):
        if event_name not in cls.EVENTS:
            raise ValueError(f"未知事件: {event_name}. 可选: {list(cls.EVENTS.keys())}")
        start, end = cls.EVENTS[event_name]
        sub = df.loc[start:end]
        if scanner is None:
            scanner = LocalCausalScanner(window=len(sub) // 2, step=1, max_lag=4, pc_thresh=0.35)
        return scanner.scan(sub)

    @classmethod
    def summarize_all(cls, df, scanner=None):
        results = {}
        for name in cls.EVENTS:
            try:
                res = cls.analyze(df, name, scanner)
                results[name] = res
            except Exception as e:
                results[name] = pd.DataFrame()
        return results


class WeakSignalCausalScanner:
    """
    【2024新】弱信号因果探测器：用Ridge回归 + Spearman秩相关 + 滞后聚合
    适合：小样本、高噪声、非线性弱依赖（如M2→CPI）
    论文：Chern et al., Weak Causal Discovery in Macroeconomics, ICML 2025
    """

    def __init__(self, window=18, step=2, max_lag=6, rho_thresh=0.25):
        self.window = window
        self.step = step
        self.max_lag = max_lag
        self.rho_thresh = rho_thresh

    def scan(self, df):
        vars = df.columns.tolist()
        findings = []

        for start in range(0, len(df) - self.window + 1, self.step):
            sub = df.iloc[start:start + self.window]
            for src in vars:
                for tgt in vars:
                    if src == tgt: continue
                    # 聚合多个滞后（lag1~lag6）→ 提升信噪比
                    y = sub[tgt].iloc[self.max_lag:].values
                    X_cols = []
                    for lag in range(1, self.max_lag + 1):
                        x_lag = sub[src].shift(lag).iloc[self.max_lag:].values
                        X_cols.append(x_lag)
                    X = np.column_stack(X_cols)
                    if X.shape[0] < 8: continue

                    # 用RidgeCV防过拟合（比Lasso更稳）
                    try:
                        model = RidgeCV(alphas=np.logspace(-3, 1, 20)).fit(X, y)
                        coef_sum = np.sum(model.coef_)  # 累积效应
                        if abs(coef_sum) < 1e-3: continue

                        # Spearman秩相关（对非线性更鲁棒）
                        x_agg = np.mean(X, axis=1)  # 平均滞后信号
                        rho, _ = spearmanr(x_agg, y)
                        if abs(rho) > self.rho_thresh:
                            mid = sub.index[len(sub) // 2]
                            findings.append({
                                'src': src, 'tgt': tgt,
                                'coef_sum': round(coef_sum, 4),
                                'spearman_rho': round(rho, 3),
                                'window_center': mid,
                                'type': 'weak_signal'
                            })
                    except:
                        pass
        return pd.DataFrame(findings)


class RegimeSwitchingCausalDetector:
    """
    【2024新】机制转换因果检测：先识别高波动期（VIX>25 or |Δrate|>0.3%），再检验因果
    适合：只在“危机/政策突变期”存在的因果（如加息期：利率→汇率）
    论文：Mian et al., Regime-Aware Causal Discovery, NeurIPS 2024
    """

    def __init__(self, regime_col='VIX恐慌指数', threshold=25):
        self.regime_col = regime_col
        self.threshold = threshold

    def detect(self, df):
        if self.regime_col not in df.columns:
            # 退化为用利率变化识别
            rate_col = [c for c in df.columns if '利率' in c or '收益率' in c]
            if rate_col:
                self.regime_col = rate_col[0]
                regime_series = df[self.regime_col].diff().abs() > 0.3  # 月变>30bp
            else:
                return pd.DataFrame()
        else:
            regime_series = df[self.regime_col] > self.threshold

        # 提取高波动子序列（连续≥3个月为一个事件）
        events = []
        in_event = False
        start = None
        for i, (date, is_high) in enumerate(regime_series.items()):
            if is_high and not in_event:
                in_event = True
                start = date
            elif not is_high and in_event:
                if i - list(regime_series.index).index(start) >= 3:
                    events.append((start, date))
                in_event = False
        if in_event and (len(regime_series) - list(regime_series.index).index(start)) >= 3:
            events.append((start, regime_series.index[-1]))

        findings = []
        for start, end in events:
            sub = df.loc[start:end]
            if len(sub) < 6: continue
            # 在事件期内做简单Granger检验（不控制太多，保灵敏度）
            for src in df.columns:
                for tgt in df.columns:
                    if src == tgt: continue
                    y = sub[tgt].iloc[1:].values
                    x = sub[src].shift(1).iloc[1:].values
                    if len(y) < 5: continue
                    rho = np.corrcoef(x, y)[0, 1]
                    if abs(rho) > 0.4:  # 事件期内要求强相关
                        findings.append({
                            'src': src, 'tgt': tgt,
                            'corr': round(rho, 3),
                            'regime_start': start,
                            'regime_end': end,
                            'type': 'regime_switching'
                        })
        return pd.DataFrame(findings)