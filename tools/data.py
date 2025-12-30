import pandas as pd
import numpy as np

def load_macro_data(path="macro_data_2015_2025_extended.csv"):

    df = pd.read_csv(path, parse_dates=['month'], index_col='month')
    df = df.select_dtypes(include=[np.number]).asfreq('MS')
    return df.dropna()

def make_stationary(df, method='diff'):

    df_stat = pd.DataFrame(index=df.index)
    for col in df.columns:
        if '同比' in col or '指数' in col or '人数' in col or 'M2' in col or '社融' in col:
            df_stat[col] = df[col].diff()
        else:
            df_stat[col] = df[col].pct_change()  
    return df_stat.dropna()