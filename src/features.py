# src/features.py
import numpy as np
import pandas as pd
from .config import COLS, ENG

def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Safe division that returns 0 when denominator is 0 or NaN and removes infs."""
    den = den.replace(0, np.nan)
    out = (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

def _to_binary(series: pd.Series) -> pd.Series:
    """Coerce a column that should be 0/1 into a clean binary 0/1 numeric series."""
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    # Clip into [0,1] in case of weird encodings
    return s.clip(0, 1)

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Existing engineered features ---

    # Average grade across semesters
    df[ENG["AVG_GRADE"]] = (
        pd.to_numeric(df[COLS["U1_GRADE"]], errors="coerce").fillna(0.0) +
        pd.to_numeric(df[COLS["U2_GRADE"]], errors="coerce").fillna(0.0)
    ) / 2.0

    # Totals
    df[ENG["TOTAL_APPROVED"]] = (
        pd.to_numeric(df[COLS["U1_APPROVED"]], errors="coerce").fillna(0.0) +
        pd.to_numeric(df[COLS["U2_APPROVED"]], errors="coerce").fillna(0.0)
    )

    df[ENG["TOTAL_ENROLLED"]] = (
        pd.to_numeric(df[COLS["U1_ENROLLED"]], errors="coerce").fillna(0.0) +
        pd.to_numeric(df[COLS["U2_ENROLLED"]], errors="coerce").fillna(0.0)
    )

    # Approval rate with safe division
    df[ENG["APPROVAL_RATE"]] = _safe_div(df[ENG["TOTAL_APPROVED"]], df[ENG["TOTAL_ENROLLED"]]).clip(0, 1)

    # --- New engineered features ---

    # Grade change (2nd - 1st)
    df[ENG["GRADE_CHANGE"]] = (
        pd.to_numeric(df[COLS["U2_GRADE"]], errors="coerce").fillna(0.0) -
        pd.to_numeric(df[COLS["U1_GRADE"]], errors="coerce").fillna(0.0)
    )

    # Failure ratios: "without evaluations" / "enrolled"
    if COLS["U1_WO_EVAL"] in df.columns and COLS["U1_ENROLLED"] in df.columns:
        num1 = pd.to_numeric(df[COLS["U1_WO_EVAL"]], errors="coerce").fillna(0.0)
        den1 = pd.to_numeric(df[COLS["U1_ENROLLED"]], errors="coerce").fillna(0.0)
        df[ENG["FAIL_RATIO_1ST"]] = _safe_div(num1, den1).clip(0, 1)
    else:
        df[ENG["FAIL_RATIO_1ST"]] = 0.0

    if COLS["U2_WO_EVAL"] in df.columns and COLS["U2_ENROLLED"] in df.columns:
        num2 = pd.to_numeric(df[COLS["U2_WO_EVAL"]], errors="coerce").fillna(0.0)
        den2 = pd.to_numeric(df[COLS["U2_ENROLLED"]], errors="coerce").fillna(0.0)
        df[ENG["FAIL_RATIO_2ND"]] = _safe_div(num2, den2).clip(0, 1)
    else:
        df[ENG["FAIL_RATIO_2ND"]] = 0.0

    # Engagement index: 0.6 * approval_rate + 0.4 * (total_approved / total_enrolled)
    # (The second term equals approval_rate by definition here, but we keep the form explicit for clarity/extension.)
    appr = pd.to_numeric(df[ENG["APPROVAL_RATE"]], errors="coerce").fillna(0.0)
    ratio = _safe_div(
        pd.to_numeric(df[ENG["TOTAL_APPROVED"]], errors="coerce").fillna(0.0),
        pd.to_numeric(df[ENG["TOTAL_ENROLLED"]], errors="coerce").fillna(0.0),
    ).clip(0, 1)
    df[ENG["ENGAGEMENT_INDEX"]] = (0.6 * appr + 0.4 * ratio).fillna(0.0)

    # Financial pressure: (1 - tuition_ok) * (1 - scholarship_holder)
    # Assumes TUITION_OK and SCHOLARSHIP are 0/1-like encodings
    tuition_ok = _to_binary(df.get(COLS["TUITION_OK"], 1))
    scholarship = _to_binary(df.get(COLS["SCHOLARSHIP"], 0))
    no_scholar = (1 - scholarship)
    df[ENG["FINANCIAL_PRESSURE"]] = ((1 - tuition_ok) * no_scholar).astype(float)

    # Final cleanup to ensure no NaNs/Infs slipped through
    for c in [
        ENG["AVG_GRADE"], ENG["TOTAL_APPROVED"], ENG["TOTAL_ENROLLED"],
        ENG["APPROVAL_RATE"], ENG["GRADE_CHANGE"], ENG["FAIL_RATIO_1ST"],
        ENG["FAIL_RATIO_2ND"], ENG["ENGAGEMENT_INDEX"], ENG["FINANCIAL_PRESSURE"]
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df
