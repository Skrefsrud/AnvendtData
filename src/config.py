# src/config.py
# Central place for seeds, paths, and column names.

SEED = 42

# Raw -> engineered column names (use the exact dataset headers)
COLS = {
    "TARGET": "Target",
    "AGE": "Age at enrollment",
    "GENDER": "Gender",
    "MARITAL": "Marital status",
    "SCHOLARSHIP": "Scholarship holder",
    "TUITION_OK": "Tuition fees up to date",
    "DEBTOR": "Debtor",
    "INTL": "International",
    "U1_GRADE": "Curricular units 1st sem (grade)",
    "U2_GRADE": "Curricular units 2nd sem (grade)",
    "U1_APPROVED": "Curricular units 1st sem (approved)",
    "U2_APPROVED": "Curricular units 2nd sem (approved)",
    "U1_ENROLLED": "Curricular units 1st sem (enrolled)",
    "U2_ENROLLED": "Curricular units 2nd sem (enrolled)",
    "UNEMP": "Unemployment rate",
    "INFL": "Inflation rate",
    "GDP": "GDP",
    # optional keepers
    "MARITAL_STATUS": "Marital status",
}

# Engineered column names
ENG = {
    "AVG_GRADE": "average_grade",
    "TOTAL_APPROVED": "total_approved_units",
    "TOTAL_ENROLLED": "total_enrolled_units",
    "APPROVAL_RATE": "overall_approval_rate",
}

# Feature sets
SELECTED_FEATURES = [
    ENG["AVG_GRADE"],
    ENG["TOTAL_APPROVED"],
    ENG["APPROVAL_RATE"],
    COLS["TUITION_OK"],
    COLS["DEBTOR"],
    COLS["SCHOLARSHIP"],
    COLS["AGE"],
    COLS["GENDER"],
    COLS["MARITAL_STATUS"],
    COLS["UNEMP"],
    COLS["INFL"],
    COLS["GDP"],
    COLS["INTL"],
]