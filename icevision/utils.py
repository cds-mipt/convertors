import pandas as pd


def load_annot_as_df(path):
    return pd.read_csv(path, sep="\t", na_values=[], keep_default_na=False, dtype={
        "class": str, "xtl": int, "ytl": int, "xbr": int, "ybr": int,
        "temporary": bool, "occluded": bool, "data": str
    })
