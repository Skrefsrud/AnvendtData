# src/features.py
import numpy as np
import pandas as pd
from .config import COLS, ENG

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Average grade across semesters
    df[ENG["AVG_GRADE"]] = (
        df[COLS["U1_GRADE"]].astype(float).fillna(0.0) + 
        df[COLS["U2_GRADE"]].astype(float).fillna(0.0)
    ) / 2.0

    # Totals
    df[ENG["TOTAL_APPROVED"]] = (
        df[COLS["U1_APPROVED"]].astype(float).fillna(0.0) + 
        df[COLS["U2_APPROVED"]].astype(float).fillna(0.0)
    )

    df[ENG["TOTAL_ENROLLED"]] = (
        df[COLS["U1_ENROLLED"]].astype(float).fillna(0.0) + 
        df[COLS["U2_ENROLLED"]].astype(float).fillna(0.0)
    )

    # Approval rate with safe division
    denom = df[ENG["TOTAL_ENROLLED"]].replace(0, np.nan)
    df[ENG["APPROVAL_RATE"]] = (df[ENG["TOTAL_APPROVED"]] / denom).fillna(0.0)

    # Clip sanity
    df[ENG["APPROVAL_RATE"]] = df[ENG["APPROVAL_RATE"]].clip(0, 1)

    return df