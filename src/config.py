import numpy as np

# df["CWE target"] = np.nan
# df.loc[df["target"] == 1, "CWE target"] = df["CWE ID"]
# df = df[df["CWE target"].isin(df["CWE target"].value_counts(dropna=False)[df["CWE target"].value_counts(dropna=False) > 10].index)]
# df["CWE target"].value_counts().index
global CWEs
CWEs = [np.nan, 'CWE-119', 'CWE-20', 'CWE-399', 'CWE-264', 'CWE-200', 'CWE-190', \
                'CWE-416', 'CWE-125', 'CWE-189', 'CWE-362', 'CWE-284', 'CWE-254', \
                'CWE-476', 'CWE-787', 'CWE-732', 'CWE-310', 'CWE-404', 'CWE-79']
assert len(CWEs) == 19
