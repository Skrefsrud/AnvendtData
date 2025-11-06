# 🎓 Student Dropout Prediction — Data Preparation & Modeling

## 📁 Project Overview

This project predicts **student academic outcomes** — _Dropout_, _Enrolled_, or _Graduate_ — using demographic, financial, and academic features from the `graduation_dataset`.

We follow **CRISP-DM**: data understanding → preparation → modeling → evaluation → interpretation.
The goal is accurate prediction **and** actionable insights for **preventive dropout interventions**.

---

## 🚀 Results Overview

| Metric            | Random Forest |            XGBoost |
| ----------------- | ------------: | -----------------: |
| **Accuracy**      |    **0.7458** |             0.7401 |
| **F1 — Dropout**  |      **0.75** |              0.735 |
| **F1 — Graduate** |      **0.48** |              0.415 |
| **F1 — Enrolled** |          0.85 | **0.847** (≈ same) |

**Blunt take:**
Random Forest slightly outperforms XGBoost overall — especially for **Graduate**, the hardest class.
Both models are strong on **Dropout** and **Enrolled**, making them reliable for early detection of at-risk students.

---

## 🧱 Notebooks Overview

### 1️⃣ `01_eda.ipynb` — Exploratory Data Analysis

**Purpose:** Understand data, spot issues, surface predictive signals.

- Integrity checks: **no missing values or duplicates**
- Class imbalance confirmed:

  - _Graduate_ (majority), _Dropout_ and _Enrolled_ (minorities)

- **Key findings:**

  - Academic performance and financial status dominate prediction power.
  - Demographics have minimal correlation.

**Outputs:**

- `eda_corr_heatmap.png`
- `eda_histograms.png`
- `eda_target.png`

---

### 2️⃣ `02_prepare_features.ipynb` — Data Preparation & Feature Engineering

**Purpose:** Build a clean, modeling-ready dataset.

- **Target encoding**: Dropout→0, Graduate→1, Enrolled→2
- **Engineered features:**

  - `average_grade`
  - `total_approved_units`
  - `total_enrolled_units`
  - `overall_approval_rate`

- Dropped redundant and weak demographic features.
- Stratified 80/20 split to preserve class ratios.

**Artifacts:**

- `data/processed/modeling.csv`
- `X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`

---

### 3️⃣ `03_train_rf_baseline.ipynb` — Random Forest Baseline

**Model**

```python
RandomForestClassifier(
  n_estimators=300, max_depth=20,
  min_samples_split=4, min_samples_leaf=2,
  max_features='sqrt', class_weight='balanced',
  random_state=42, n_jobs=-1
)
```

| Class         |  Precision | Recall |   F1 |
| ------------- | ---------: | -----: | ---: |
| Dropout       |       0.79 |   0.72 | 0.75 |
| Graduate      |       0.44 |   0.52 | 0.48 |
| Enrolled      |       0.85 |   0.84 | 0.85 |
| **Accuracy:** | **0.7458** |        |      |

**Top features:**

1. average_grade
2. overall_approval_rate
3. total_approved_units
4. tuition_fees_up_to_date
5. age_at_enrollment

**Artifacts**

- `models/rf_baseline.pkl`
- `reports/rf_metrics.json`
- `figures/rf_confusion_raw.png`, `rf_confusion_norm.png`
- `figures/rf_feature_importance.png`

---

### 4️⃣ `04_train_xgb.ipynb` — XGBoost Classifier

**Model**

```python
XGBClassifier(
  n_estimators=600, learning_rate=0.05, max_depth=6,
  subsample=0.8, colsample_bytree=0.8,
  reg_lambda=1.0, objective="multi:softprob",
  eval_metric="mlogloss", tree_method="hist",
  random_state=42, n_jobs=-1
)
```

| Class         |  Precision | Recall |   F1 |
| ------------- | ---------: | -----: | ---: |
| Dropout       |       0.75 |   0.72 | 0.73 |
| Graduate      |       0.45 |   0.38 | 0.42 |
| Enrolled      |       0.82 |   0.88 | 0.85 |
| **Accuracy:** | **0.7401** |        |      |

**Artifacts**

- `models/xgb_classifier.pkl`
- `reports/xgb_metrics.json`
- `figures/xgb_confusion_raw.png`, `xgb_confusion_norm.png`
- `figures/xgb_feature_importance_gain.png`

---

### 5️⃣ `05_interpretation_shap.ipynb` — SHAP Explainability

**Purpose:** Explain model predictions and quantify feature impact.

- Used **TreeExplainer** with interventional background for stability.
- Computed **global** and **per-class** SHAP values.
- Exported global mean |SHAP| scores → `reports/random_forest_shap_importance.csv`.
- Generated **summary** and **dependence** plots for interpretability.

**Top predictive drivers (RF):**

1. average_grade
2. overall_approval_rate
3. total_approved_units
4. tuition_fees_up_to_date
5. age_at_enrollment

**Artifacts**

- `figures/random_forest_shap_summary.png`
- `figures/random_forest_shap_dependence_[feature].png`

---

### 6️⃣ `06_report_figures.ipynb` — Final Report & Visualization

**Purpose:** Produce publication-ready figures and comparisons.

- Side-by-side metric plots (RF vs XGB)
- Top-10 SHAP feature bar charts
- Optional scatter: model vs SHAP importance alignment
- Final Markdown summary (`reports/final_summary.md`)

**Artifacts**

- `figures/model_comparison.png`
- `figures/rf_top_shap_features.png`
- `figures/xgb_top_shap_features.png`
- `figures/rf_shap_vs_importance.png`
- `reports/final_summary.md`

---

## 🖼️ Figures Overview

### EDA

- `eda_corr_heatmap.png` — feature correlation heatmap
- `eda_histograms.png` — per-feature distributions
- `eda_target.png` — target class distribution

### Model Evaluation

- `rf_confusion_norm.png` / `xgb_confusion_norm.png` — normalized confusion matrices
- `rf_confusion_raw.png` / `xgb_confusion_raw.png` — raw prediction counts
- `rf_feature_importance.png` / `xgb_feature_importance_gain.png` — top features by model

### SHAP Explainability

- `random_forest_shap_summary.png` / `xgboost_shap_summary.png` — overall feature effects
- `random_forest_shap_dependence_[feature].png` — detailed per-feature influence

---

## 🧩 Tools & Libraries

- **Python 3.10+**
- **pandas**, **numpy**
- **scikit-learn**, **xgboost**
- **matplotlib**, **seaborn**
- **shap**
- **joblib**
- Optional: **pyarrow** for Parquet I/O

A complete `requirements.txt` is included.

---

## 🧪 Reproducibility & Run Order

```bash
pip install -r requirements.txt
```

**Run in order:**

1. `01_eda.ipynb`
2. `02_prepare_features.ipynb`
3. `03_train_rf_baseline.ipynb`
4. `04_train_xgb.ipynb`
5. `05_interpretation_shap.ipynb`
6. `06_report_figures.ipynb`

Artifacts are automatically saved under:

- `data/processed/`
- `models/`
- `reports/`
- `figures/`

---

## ✅ Summary

- Clean, engineered dataset ✅
- Two ensemble models trained and evaluated ✅
- Random Forest slightly stronger overall ✅
- SHAP provides clear interpretability ✅
- End-to-end reproducible ML pipeline ✅

> **Next:** optional hyperparameter tuning or dashboard deployment for stakeholder presentation.

# Ensemble_metrics.json explained

Perfect — this is exactly the kind of evaluation we want to interpret now that your **ensemble (RF + XGB)** is running.

Let’s unpack it clearly:

---

## 🧩 1. Overall picture

| Metric          | Ensemble Result | What It Means                                                 |
| :-------------- | :-------------- | :------------------------------------------------------------ |
| **Accuracy**    | **0.749**       | About **75%** of all predictions match the true class.        |
| **Macro F1**    | **0.676**       | Average F1 across the three classes (treats all equally).     |
| **Weighted F1** | **0.742**       | Weighted by class size — dominated by large “Enrolled” class. |
| **Micro F1**    | **0.749**       | Equivalent to overall accuracy for multi-class tasks.         |

So overall, your ensemble slightly outperforms your **XGBoost** (0.740 acc, 0.67 F1) and is roughly on par or a hair below your **Random Forest** baseline (0.746 acc, 0.69 F1) — but we need to look **per class** to see _why_.

---

## 🎓 2. Per-class results

| Class        | Precision | Recall |       F1 | Interpretation                                                          |
| :----------- | --------: | -----: | -------: | :---------------------------------------------------------------------- |
| **Dropout**  |      0.77 |   0.73 | **0.75** | Strong — your model detects dropouts quite well.                        |
| **Graduate** |      0.47 |   0.39 | **0.43** | Still the hardest — some improvement over XGB (≈0.41), but not much.    |
| **Enrolled** |      0.81 |   0.89 | **0.85** | Excellent — your model is very good at recognizing continuing students. |

### 💡 Summary

- The **Dropout** class remains solid.
- **Graduate** (the minority class) still lags — both **recall** and **precision** are low, meaning the model confuses graduates with the other two outcomes.
- **Enrolled** dominates and performs strongly, slightly inflating overall metrics due to its size.

---

## 📈 3. Macro vs. Weighted F1

- **Macro F1 (0.676):** Treats each class equally. Penalized by poor Graduate performance.
- **Weighted F1 (0.742):** Higher because Enrolled (large, easy class) performs well and dominates.

→ The gap between them (**0.742 vs. 0.676**) indicates class imbalance still matters — the model’s success is uneven across classes.

---

## ⚖️ 4. How the Ensemble Affects Each Class

Compared to your individual models:

| Class    | RF F1 | XGB F1 | Ensemble F1 | Trend                           |
| :------- | ----: | -----: | ----------: | :------------------------------ |
| Dropout  |  0.75 |   0.73 |    **0.75** | Same or slightly better         |
| Graduate |  0.48 |   0.41 |    **0.43** | Slightly below RF but above XGB |
| Enrolled |  0.85 |   0.85 |    **0.85** | Essentially unchanged           |

**Interpretation:**
The ensemble combined RF’s stronger handling of Dropout/Graduate with XGB’s stronger Enrolled predictions. It stabilized generalization (accuracy slightly higher than XGB), but didn’t solve the Graduate-class weakness.

---

## 🧠 5. What It Tells You Practically

1. **Ensembling helps stability** — similar accuracy to RF but with slightly smoother precision/recall trade-offs.
2. **The "Graduate" class remains underrepresented** — recall below 0.40 suggests the model still misses many graduates.
3. **Next improvements** should target that class specifically:

   - Use **class-weight tuning** for Graduate.
   - Try **oversampling** or **SMOTE** for Graduate only.
   - Engineer features capturing _completion indicators_ (e.g. “credits completed” or “final-year performance”).

4. **Interpretation goal reached** — you’ve achieved a consistent, balanced model that’s explainable, with next steps clearly defined.

---

## 🧾 TL;DR

| Metric                  | Meaning                                                      |
| :---------------------- | :----------------------------------------------------------- |
| **Accuracy (0.749)**    | Predicts ~3/4 of outcomes correctly.                         |
| **Macro F1 (0.676)**    | Average performance across classes — Graduate is still weak. |
| **Weighted F1 (0.742)** | Reflects good performance on the majority class.             |
| **Graduate F1 (0.43)**  | Main area for improvement.                                   |

**In words:**

> The ensemble achieves solid overall performance and strong dropout/enrolled detection, but struggles to accurately classify graduates — the most difficult and underrepresented group. Future work should rebalance or add graduate-specific features rather than rely on model blending alone.

---

Would you like me to summarize this in a short paragraph suitable for including in your _report section_ (“Method and Analysis”)?
