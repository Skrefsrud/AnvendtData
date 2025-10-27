# src/io_utils.py
import pandas as pd
from pathlib import Path

def load_df(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    return pd.read_csv(path)

def save_df(df, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in [".parquet", ".pq"]:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)