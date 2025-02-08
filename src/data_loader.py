import pandas as pd
import numpy as np
import config
import tinygrad

def load_data(path:str, batch_size:int, percentage:float=1.0):
        print("Loading data from", path)
        df = pd.read_csv(path)
        df = df[df["processed_func"].str.len() < 500] # remove long functions
        # predict CWEs
        df = df[((df["CWE ID"].isin(config.CWEs)) & (df["target"] == 1)) | (df["target"] == 0)]
        df["CWE target"] = np.where(df["target"] == 1, df["CWE ID"], np.nan)
        # downsampling nan
        df = pd.concat([df[df["CWE target"].isna() == False], df[df["CWE target"].isna() == True].sample(len(df[df["CWE target"].isna() == False])*3)])

        X = df["processed_func"].tolist()
        y = df["CWE target"].tolist()
        y = tinygrad.Tensor([config.CWEs.index(i) if not pd.isna(i) else 0 for i in y], requires_grad=False)
        X = X[:int(len(X)*percentage)//batch_size*batch_size]
        y = y[:int(len(y)*percentage)//batch_size*batch_size]
        print("Data loaded: X", len(X), "y", len(y))
        return X, y
