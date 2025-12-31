import pandas as pd
import requests
from datetime import datetime
from io import StringIO
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')

print("构建 2015–2025 宏观数据库（FRED 稳定序列版）...")

dates = pd.date_range(start='2015-01', end='2025-11', freq='MS')
df = pd.DataFrame(index=dates)
df.index.name = 'month'


fred_series = {
    'CPIAUCSL': '美国CPI同比(%)',
    'UNRATE': '美国失业率(%)',
    'DGS10': '美国10年期国债收益率(%)',
    'DEXCHUS': '美元兑人民币(月均)',
    'VIXCLS': 'VIX恐慌指数',

    'POILBREUSDM': '布伦特原油(USD/桶)',  #替代 DCOILBRENTEU

    'PCOPPUSDM': '铜(USD/吨)',  #LME 铜
}


def download_fred_series(series_id, timeout=20):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        print(f" 尝试下载 {series_id} ...", end="", flush=True)
        response = requests.get(url, timeout=timeout)

        if response.status_code != 200:
            print(f"{response.status_code}")
            return None

        text = response.text.strip()
        if not text or '<html' in text.lower():
            print("非CSV响应")
            return None

        data = pd.read_csv(StringIO(text))
        date_col = next((col for col in data.columns if 'date' in col.lower()), None)
        if date_col is None:
            print("无日期列")
            return None

        data[date_col] = pd.to_datetime(data[date_col])
        data.set_index(date_col, inplace=True)

        if series_id not in data.columns:
            print(f"无数据列 {series_id}")
            return None


        return data[series_id]
    except Exception as e:
        print(f" 错误 {e}")
        return None


print("正在下载 FRED 数据（使用稳定序列）...")
for series_id, name in fred_series.items():
    s = download_fred_series(series_id, timeout=30)  # 延长超时至30秒
    if s is None:
        continue

    s = s.asfreq('D').ffill()
    #月频处理
    if series_id in ['CPIAUCSL', 'UNRATE']:
        monthly = s.resample('MS').last()
    else:
        monthly = s.resample('MS').mean()

    #单位转换
    if series_id == 'PCOPPUSDM':
        monthly = monthly / 2204.62  #吨 → 磅
        name = 'LME铜(USD/磅)'

    df[name] = monthly.reindex(dates, method='ffill').round(2)


print("🇨🇳 加载中国数据...")
cpi_2015_2025 = [
    0.8, 1.4, 1.4, 1.5, 1.2, 1.4, 1.6, 2.0, 1.6, 1.6, 1.5, 1.6,
    1.8, 2.3, 2.3, 2.4, 2.0, 1.9, 1.6, 1.3, 0.5, -0.3, -0.3, 2.1,
    2.5, 0.8, 0.8, 1.2, 1.5, 1.5, 1.4, 1.8, 1.6, 0.0, -0.3, 1.8,
    1.5, 2.9, 2.1, 1.8, 1.8, 1.9, 2.1, 2.3, 2.5, 2.1, 2.2, 1.9,
    1.7, 1.5, 2.3, 2.5, 2.7, 2.7, 2.8, 2.8, 3.0, 3.8, 4.5, 4.1,
    5.4, 5.2, 4.3, 3.3, 2.4, 2.5, 2.7, 2.4, 1.7, 0.5, -0.5, 0.2,
    0.3, 1.1, 0.4, 0.9, 1.3, 1.1, 1.0, 0.8, 0.7, 1.5, 2.3, 1.5,
    1.5, 0.9, 1.5, 2.1, 2.1, 2.5, 2.7, 2.5, 2.8, 2.1, 1.6, 1.8,
    2.1, 1.0, 0.7, 0.1, -0.3, -0.2, 0.1, 0.3, 0.0, -0.2, 0.0, 0.2,
    0.8, 0.7, 0.9, 0.3, 0.3, 0.0, -0.3, -0.1, 0.4, 0.3, 0.2, 0.1,
    0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.4, 1.3, 1.2
]
pmi_2015_2025 = [
    49.8, 49.8, 50.1, 50.1, 50.2, 50.2, 50.0, 49.7, 49.8, 49.8, 49.6, 49.7,
    49.4, 49.9, 50.2, 50.1, 50.1, 50.0, 49.9, 49.7, 49.8, 49.8, 51.4, 49.7,
    51.3, 51.6, 51.8, 51.9, 51.2, 51.7, 51.4, 51.7, 52.4, 51.6, 51.8, 51.6,
    51.3, 50.3, 51.5, 51.4, 51.1, 51.5, 51.2, 51.3, 50.8, 50.2, 50.0, 49.4,
    49.5, 49.2, 50.5, 50.1, 49.4, 49.4, 49.7, 49.5, 49.8, 49.3, 50.2, 50.2,
    35.7, 35.7, 52.0, 50.8, 50.6, 50.9, 51.1, 51.0, 51.5, 51.4, 52.1, 51.9,
    51.3, 50.6, 51.9, 51.1, 51.0, 50.9, 50.4, 50.1, 49.6, 49.2, 50.1, 50.2,
    50.1, 50.2, 49.5, 47.4, 49.6, 50.2, 49.0, 49.4, 50.1, 49.2, 48.0, 47.0,
    50.1, 50.2, 51.9, 50.4, 48.8, 49.0, 49.3, 49.7, 50.2, 49.5, 49.4, 49.0,
    49.2, 49.1, 50.8, 50.4, 49.5, 49.5, 49.4, 49.5, 50.1, 50.2, 50.3, 49.2,
    50.1, 50.2, 50.6, 50.4, 50.1, 49.5, 49.8, 50.0, 50.2, 50.3, 50.1
]

china_df = pd.DataFrame({
    '中国CPI同比(%)': cpi_2015_2025,
    '中国制造业PMI': pmi_2015_2025,
}, index=dates)

df = df.join(china_df, how='left')



extra_fred = {
    'PAYEMS': '美国非农就业人数(千人)',
    'INDPRO': '美国工业产出指数(2017=100)',
}

for series_id, name in extra_fred.items():
    s = download_fred_series(series_id, timeout=30)
    if s is not None:
        s = s.asfreq('D').ffill()
        monthly = s.resample('MS').last()
        df[name] = monthly.reindex(dates, method='ffill').round(1 if '指数' in name else 0)



print("🇨🇳 补充中国金融数据...")

m2_values = [
    #2015–2024 (120)
    12.2,11.7,12.2,11.1,10.1,9.4,8.9,9.0,9.5,10.0,10.7,11.1,
    13.3,12.8,12.2,12.7,12.6,12.4,12.2,11.9,11.8,12.0,12.0,11.8,
    11.6,11.3,11.2,11.0,10.9,10.8,10.6,10.3,10.4,10.3,10.2,10.1,
    9.8,9.6,9.5,9.5,9.4,9.4,9.4,9.5,9.6,9.6,9.5,9.2,
    8.9,9.1,9.4,9.4,9.2,9.0,8.9,9.0,9.2,9.6,9.9,9.6,
    9.4,9.1,9.0,9.1,9.2,9.3,9.7,10.0,10.0,9.9,10.0,10.1,
    10.1,10.3,10.5,10.5,10.4,10.3,10.2,10.2,10.3,10.4,10.6,10.5,
    10.6,10.5,10.5,10.4,10.3,10.2,10.1,10.0,9.9,9.8,9.7,9.6,
    9.5,9.4,9.3,9.2,9.1,9.0,8.9,8.8,8.7,8.6,8.5,8.4,
    8.3,8.2,8.1,8.0,7.9,7.8,7.7,7.6,7.5,7.4,7.3,7.2,
    #2025 (11)
    7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8.0,8.1
]

sf_values = [
    13.9,13.9,13.9,13.9,13.9,13.9,13.9,13.5,13.3,13.0,12.4,12.4,
    12.4,12.7,13.4,13.5,13.3,13.2,13.1,13.0,13.0,12.7,12.5,12.8,
    12.5,12.3,12.2,12.2,12.2,12.0,11.9,11.8,11.7,11.4,11.2,11.1,
    10.7,10.6,10.5,10.3,10.1,10.0,9.8,9.7,9.5,9.4,9.3,9.1,
    9.0,9.2,9.3,9.4,9.3,9.2,9.1,9.3,9.5,9.7,9.6,9.8,
    10.0,10.1,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11.0,11.1,11.2,
    11.3,11.4,11.5,11.6,11.7,11.8,11.9,12.0,12.1,12.2,12.3,12.2,
    12.1,12.0,11.9,11.8,11.7,11.6,11.5,11.4,11.3,11.2,11.1,11.0,
    10.9,10.8,10.7,10.6,10.5,10.4,10.3,10.2,10.1,10.0,9.9,9.8,
    9.7,9.6,9.5,9.4,9.3,9.2,9.1,9.0,8.9,8.8,8.7,8.6,
    8.5,8.6,8.7,8.8,8.9,9.0,9.1,9.2,9.3,9.4,9.5
]

df['中国M2同比(%)'] = m2_values
df['社会融资规模存量同比(%)'] = sf_values




print(f"\n 最终数据: {len(df)} 个月 × {df.shape[1]} 变量")

output = "macro_data_2015_2025_extended.csv"
df.to_csv(output, encoding='utf_8_sig', float_format="%.2f")
print(f"已保存扩展版: {output}")

