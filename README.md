# 🎓 Student Dropout Prediction — Data Preparation & Modeling

## 📁 Project Overview

This project predicts **student academic outcomes** — _Dropout_, _Enrolled_, or _Graduate_ — using demographic, financial, and academic features from the `graduation_dataset`.

We follow **CRISP-DM**: data understanding → preparation → modeling → evaluation → interpretation. The goal is accurate prediction **and** actionable insights for **preventive dropout interventions**.

---

## 🧭 Current Progress (up to Step 04)

### 1️⃣ `01_eda.ipynb` — Exploratory Data Analysis

**Purpose:** Understand data, spot issues, surface predictive signals.

**Done**

- Integrity checks: **no missing values / duplicates**.
- Class imbalance confirmed:

  - _Graduate_ (majority), _Dropout_ and _Enrolled_ (minorities).

- Distributions & correlations:

  - **Academic performance** (grades/approvals) strongly related to outcomes.
  - **Demographics/parental** mostly weak.

- Outputs: 3 figures (class balance, distributions, correlation heatmap) + short EDA report.

**Key insight:**

> Academic + financial performance dominate; demographics add little.

---

### 2️⃣ `02_prepare_features.ipynb` — Data Preparation & Feature Engineering

**Purpose:** Build a modeling-ready table.

**Done**

- Project structure with reusable `src/` helpers.
- **Target encoding** (`LabelEncoder`): Dropout→0, Graduate→1, Enrolled→2.
- **Engineered features**

  - `average_grade` (mean of 1st/2nd semester)
  - `total_approved_units`
  - `total_enrolled_units`
  - `overall_approval_rate` (safe divide, clipped [0,1])

- **Feature reduction**: kept academic, financial, key demographic, and economic indicators; dropped weak/duplicate columns.
- **Stratified 80/20 split**.
- **Saved** processed datasets:

  - `data/processed/modeling.csv`
  - `data/processed/X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`

---

### 3️⃣ `03_train_rf_baseline.ipynb` — Random Forest Baseline

**Purpose:** Solid, interpretable benchmark.

**Model**

```python
RandomForestClassifier(
  n_estimators=300, max_depth=20,
  min_samples_split=4, min_samples_leaf=2,
  max_features='sqrt', class_weight='balanced',
  random_state=42, n_jobs=-1
)
```

**Test performance**

| Class                    | Precision | Recall |   F1 |
| ------------------------ | --------: | -----: | ---: |
| Dropout                  |      0.79 |   0.72 | 0.75 |
| Graduate                 |      0.44 |   0.52 | 0.48 |
| Enrolled                 |      0.85 |   0.84 | 0.85 |
| **Accuracy:** **0.7458** |           |        |      |

**Confusion (normalized)**

- Dropout: **72%** correct; 19%→Graduate; 8%→Enrolled
- Graduate: **52%** correct; 21%→Dropout; 27%→Enrolled
- Enrolled: **84%** correct; 11%→Graduate; 5%→Dropout

**Top features (importance)**

1. average_grade 2) overall_approval_rate 3) total_approved_units
2. tuition_fees_up_to_date 5) age_at_enrollment

**Artifacts**

- `models/rf_baseline.pkl`
- `reports/rf_metrics.json`
- `figures/rf_confusion_raw.png`, `figures/rf_confusion_norm.png`
- `figures/rf_feature_importance.png`

---

### 4️⃣ `04_train_xgb.ipynb` — XGBoost Classifier

**Purpose:** Try boosted trees to improve class separation, especially **Graduate**.

**Model**

```python
XGBClassifier(
  n_estimators=600, learning_rate=0.05, max_depth=6,
  subsample=0.8, colsample_bytree=0.8,
  reg_lambda=1.0, reg_alpha=0.0,
  objective="multi:softprob", eval_metric="mlogloss",
  tree_method="hist", random_state=42, n_jobs=-1
)
```

**Test performance**

| Class                    | Precision | Recall |    F1 |
| ------------------------ | --------: | -----: | ----: |
| Dropout                  |     0.748 |  0.722 | 0.735 |
| Graduate                 |     0.452 |  0.384 | 0.415 |
| Enrolled                 |     0.817 |  0.880 | 0.847 |
| **Accuracy:** **0.7401** |           |        |       |

**Confusion (normalized)**

- Dropout: **72%** correct; 16%→Graduate; 12%→Enrolled
- Graduate: **38%** correct; 28%→Dropout; 34%→Enrolled
- Enrolled: **88%** correct; 6%→Graduate; 6%→Dropout

**Artifacts**

- `models/xgb_classifier.pkl`
- `reports/xgb_metrics.json`
- `figures/xgb_confusion_raw.png`, `figures/xgb_confusion_norm.png`
- `figures/xgb_feature_importance_gain.png`

---

## 🥊 RF vs XGB — Head-to-Head (test set)

| Metric            | Random Forest |            XGBoost |
| ----------------- | ------------: | -----------------: |
| **Accuracy**      |    **0.7458** |             0.7401 |
| **F1 — Dropout**  |      **0.75** |              0.735 |
| **F1 — Graduate** |      **0.48** |              0.415 |
| **F1 — Enrolled** |          0.85 | **0.847** (≈ same) |

**Blunt take:** RF edges out XGB overall here, especially on **Graduate**. Both models are strong on **Enrolled** and decent on **Dropout**. The **Graduate** class remains the pain point (behavior overlaps with Enrolled/Dropout near completion).

---

## 🖼️ Figures

This project generates several figures to visualize the data and model performance.

### EDA Figures

-   `eda_corr_heatmap.png`: A heatmap showing the correlation between different features in the dataset.
-   `eda_histograms.png`: Histograms for each feature to visualize their distribution.
-   `eda_target.png`: A bar chart showing the distribution of the target variable (Dropout, Enrolled, Graduate).

### Model Evaluation Figures

-   `rf_confusion_norm.png` / `xgb_confusion_norm.png`: Normalized confusion matrices for the Random Forest and XGBoost models. These show the percentage of correct and incorrect predictions for each class.
-   `rf_confusion_raw.png` / `xgb_confusion_raw.png`: Raw confusion matrices with the count of predictions.
-   `rf_feature_importance.png` / `xgb_feature_importance_gain.png`: Feature importance plots for both models, showing which features have the most impact on the predictions.

### SHAP (SHapley Additive exPlanations) Figures

These plots explain the output of the machine learning models.

-   `random forest_shap_summary.png` / `xgboost_shap_summary.png`: SHAP summary plots. They combine feature importance with feature effects. Each point on the summary plot is a Shapley value for a feature and an instance. The position on the y-axis is determined by the feature and on the x-axis by the Shapley value.
-   `random forest_shap_dependence_{feature}.png` / `xgboost_shap_dependence_{feature}.png`: These plots show the effect of a single feature on the SHAP value of that feature. They are useful for understanding the relationship between a feature and the model's output. The plots are generated for the following features:
    -   `Age at enrollment`
    -   `average_grade`
    -   `overall_approval_rate`
    -   `total_approved_units`
    -   `Tuition fees up to date`

---

## ⚙️ Next Steps

### 5️⃣ `05_interpretation_shap.ipynb`

- Compute **SHAP** values for best model (start with RF; you can compare with XGB).
- Plots: **summary beeswarm** + **dependence** for top features.
- Deliver **actionable thresholds** (e.g., approval_rate < X, average_grade < Y increases dropout risk).

### 6️⃣ `06_report_figures.ipynb`

- Publication-ready visuals:

  - Side-by-side metrics (RF vs XGB)
  - Confusion matrices
  - SHAP plots

- Short narrative tying findings to **preventive actions** (finance flags, academic support triggers).

### (Optional) Targeted improvements

- Tune RF (`min_samples_leaf`, `max_depth`) and XGB (small randomized search).
- Try **class-specific weighting** or modest **SMOTE** only for the Graduate class if recall remains poor.
- **Ablations:** with/without reduced features; keep semester-specifics vs aggregates.

---

## 🧩 Tools & Libraries

- **Python 3.10+**
- **pandas**, **numpy** (data)
- **scikit-learn**, **xgboost** (models)
- **matplotlib**, **seaborn** (viz)
- **joblib** (persistence)
- **shap** (interpretability)
- Optional: **pyarrow** (Parquet I/O)

A matching `requirements.txt` is included.

---

## 🧪 Reproducibility & Run Order

**Setup**

```bash
pip install -r requirements.txt
```

**Run order**

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_prepare_features.ipynb`
3. `notebooks/03_train_rf_baseline.ipynb`
4. `notebooks/04_train_xgb.ipynb`
5. `notebooks/05_interpretation_shap.ipynb`
6. `notebooks/06_report_figures.ipynb`

Artifacts are saved automatically under `data/processed/`, `models/`, `reports/`, and `figures/`.

---

## ✅ Summary

- Clean, engineered dataset ✅
- RF baseline: **good accuracy**, strong **Dropout**/**Enrolled** detection ✅
- XGB: comparable accuracy; **Graduate** remains hardest ✅
- Ready to **explain** models with **SHAP** and extract thresholds for **early interventions** ✅