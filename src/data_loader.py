import pandas as pd
import numpy as np
import config

def load_data(path:str):
        df = pd.read_csv(path)
        df = df[df["processed_func"].str.len() < 500] # remove long functions
        # predict CWEs
        df = df[((df["CWE ID"].isin(config.CWEs)) & (df["target"] == 1)) | (df["target"] == 0)]
        df["CWE target"] = np.where(df["target"] == 1, df["CWE ID"], np.nan)
        # downsampling nan
        df = pd.concat([df[df["CWE target"].isna() == False], df[df["CWE target"].isna() == True].sample(10000)])

        X = df["processed_func"].tolist()
        y = df["CWE target"].tolist()
        y = [config.CWEs.index(i) if not pd.isna(i) else 0 for i in y]
        return X, y
