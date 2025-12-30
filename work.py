import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import warnings
from tools import load_macro_data, make_stationary, LocalCausalScanner, EventAnalyzer
warnings.filterwarnings('ignore')


df = pd.read_csv("macro_data_2015_2025_extended.csv", parse_dates=['month'], index_col='month')
df = df.select_dtypes(include=[np.number]).asfreq('MS').dropna()
df_raw = load_macro_data()
df_stat = make_stationary(df_raw)
df_d = df.diff().dropna()  # 一阶差分
print("差分平稳化完成 | 形状:", df_d.shape)


core = ['美国10年期国债收益率(%)', '美元兑人民币(月均)',
        '布伦特原油(USD/桶)', '中国CPI同比(%)', 'VIX恐慌指数']
df_c = df_d[[c for c in core if c in df_d.columns]]


#  局部因果探测函数（窗口小 + 阈值适中）
def local_causal_scan(df, window=39, step=1, max_lag=7, pc_thresh=0.32):
    vars = df.columns.tolist()
    n = len(df)
    findings = []

    for start in range(0, n - window + 1, step):
        sub = df.iloc[start:start + window]
        for src in vars:
            for tgt in vars:
                if src == tgt: continue
                for lag in range(1, max_lag + 1):
                    # 构建：tgt(t) ~ tgt(t-1) + src(t-lag)
                    y = sub[tgt].iloc[lag:].values
                    x_auto = sub[tgt].shift(1).iloc[lag:].values
                    x_src = sub[src].shift(lag).iloc[lag:].values
                    X = np.column_stack([x_auto, x_src])
                    if len(y) < 10: continue
                    try:
                        reg = LinearRegression().fit(X, y)
                        coef = reg.coef_[1]
                        # 偏相关近似：控制 tgt(t-1) 后的 src 系数
                        if abs(coef) > 0.02:  # 系数显著非零
                            # 检验偏相关强度
                            res_y = y - reg.predict(X) + coef * x_src  # 近似残差
                            pc = pearsonr(x_src, res_y)[0]
                            if abs(pc) > pc_thresh:
                                mid_date = sub.index[len(sub) // 2]
                                findings.append({
                                    'src': src, 'tgt': tgt, 'lag': lag,
                                    'coef': round(coef, 3),
                                    'pcorr': round(pc, 3),
                                    'window_center': mid_date.strftime('%Y-%m'),
                                    'window': f"{sub.index[0].year}-{sub.index[-1].year}"
                                })
                    except:
                        pass
    return pd.DataFrame(findings)


#  执行局部扫描
local_edges = local_causal_scan(df_c, window=24, step=6, pc_thresh=0.35)
print(f" 局部因果探测完成 | 发现 {len(local_edges)} 条显著边")

#  按经济逻辑筛选 & 汇总
if not local_edges.empty:
    # 只保留关键传导链
    interest_pairs = [
        ('美国10年期国债收益率(%)', '美元兑人民币(月均)'),
        ('美元兑人民币(月均)', '中国CPI同比(%)'),
        ('VIX恐慌指数', '布伦特原油(USD/桶)'),
        ('布伦特原油(USD/桶)', '中国CPI同比(%)')
    ]
    mask = local_edges.apply(lambda r:
                             (r['src'], r['tgt']) in interest_pairs, axis=1)
    selected = local_edges[mask].sort_values(['src', 'tgt', 'window_center'])

    print("\n 局部显著因果关系（按传导链分组）：")
    for (s, t), group in selected.groupby(['src', 'tgt']):
        print(f"\n   {s} → {t}")
        for _, row in group.iterrows():
            sign = "↑→↑" if row['coef'] > 0 else "↑→↓"
            print(f"    {row['window_center']} | lag{row['lag']} | {sign} β={row['coef']:+.3f}, ρ={row['pcorr']:+.2f}")
else:
    print("\n 无局部显著边（可尝试降低 pc_thresh=0.3）")


print("\n 结果使用样例：")
events = {
    '疫情': ('2020-01', '2020-12'),
    '加息': ('2022-03', '2023-06'),
    '复苏': ('2023-01', '2023-12')
}
for name, (start, end) in events.items():
    sub = df_d.loc[start:end]
    print(f"\n  {name} ({start} ~ {end}) | {len(sub)} 个月")
    for src, tgt in [('VIX恐慌指数', '布伦特原油(USD/桶)'),
                     ('美国10年期国债收益率(%)', '美元兑人民币(月均)')]:
        if src in sub.columns and tgt in sub.columns:
            # 简单 Granger-type：Δoil ~ Δoil_l1 + Δvix_l1
            y = sub[tgt].iloc[1:].values
            X = np.column_stack([
                sub[tgt].shift(1).iloc[1:].values,
                sub[src].shift(1).iloc[1:].values
            ])
            if len(y) > 5:
                coef = LinearRegression().fit(X, y).coef_[1]
                corr = pearsonr(sub[src].iloc[1:], y)[0]
                print(f"    {src} → {tgt}: β={coef:+.3f}, corr={corr:+.2f}")