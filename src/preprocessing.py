# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .config import COLS, SELECTED_FEATURES

def encode_target(df: pd.DataFrame):
    df = df.copy()
    le = LabelEncoder()
    df[COLS["TARGET"]] = le.fit_transform(df[COLS["TARGET"]])
    return df, le

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SELECTED_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected features: {missing}")
    return df[SELECTED_FEATURES + [COLS["TARGET"]]].copy()

def basic_sanity_checks(df: pd.DataFrame):
    # Ensure numeric types where expected can be added here if needed
    return True