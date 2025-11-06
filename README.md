# Project Overview (Results & Figures)

This repository summarizes our work on predicting student outcomes (**Dropout / Enrolled / Graduate**) from academic, financial, and demographic data. The goal is twofold: (1) predict at-risk students and (2) understand which factors drive predictions.

---

## Snapshot

- **Data:** 4,424 students · 35 columns · no missing values
- **Class mix:** **Graduate 49.9%**, **Dropout 32.1%**, **Enrolled 17.9%**
- **Best model:** **Soft-voting Ensemble (RF + XGBoost)**

  - **Accuracy 0.749** · F1(D)=**0.752** · F1(G)=**0.428** · F1(E)=**0.848**

- **Key drivers (consistent across models/SHAP):**
  **engagement_index**, **overall_approval_rate**, **total_approved_units**, **average_grade**, **tuition_fees_up_to_date**, **financial_pressure**, **Age at enrollment**

---

## “If you only look at three things”

1. **Model comparison & confusion**

   - `figures/ensemble_confusion_norm.png` _(if present)_
   - `figures/rf_confusion_norm.png`, `figures/xgb_confusion_norm.png`
     What it shows: Enrolled and Dropout are predicted well; **Graduate** is the hardest (often confused with Enrolled).

2. **What drives predictions**

   - `figures/rf_feature_importance.png`, `figures/xgb_feature_importance_gain.png`
   - `figures/shap_summary_rf.png`, `figures/shap_summary_xgb.png`
     What it shows: academic progress and engagement dominate; tuition compliance and financial pressure matter; age has a supporting role.

3. **The numbers**

   - `reports/ensemble_metrics.json`, `reports/xgb_metrics.json`, `reports/rf_metrics.json`
     What it shows: exact accuracy/precision/recall/F1 per class.

---

## Performance: final numbers

| Metric            | Random Forest |    XGBoost | Ensemble (RF+XGB) |
| ----------------- | ------------: | ---------: | ----------------: |
| **Accuracy**      |    **0.7175** | **0.7322** |        **0.7492** |
| **F1 – Dropout**  |        0.7430 |     0.7436 |        **0.7523** |
| **F1 – Graduate** |        0.4183 | **0.4438** |            0.4276 |
| **F1 – Enrolled** |        0.8198 |     0.8392 |        **0.8479** |

**Interpretation (one line):** XGBoost edges RF overall; the **ensemble is best**, mainly by lifting Dropout/Enrolled while Graduate remains the toughest class.

---

## What the figures say (and where they are)

### 1) EDA (data reality check)

- **Class distribution:** `figures/eda_target.png`

  - Confirms the imbalance (Graduate ~50%).

- **Feature shapes:** `figures/eda_histograms.png`

  - **Age at enrollment is right-skewed**; grades and economics show expected ranges.

- **Correlations:** `figures/eda_corr_heatmap.png`

  - Strong cluster among curricular/approval/grade variables; weak demographic correlations.

### 2) Model quality

- **Confusion matrices (normalized):**

  - RF: `figures/rf_confusion_norm.png`
  - XGB: `figures/xgb_confusion_norm.png`
  - Ensemble (if saved): `figures/ensemble_confusion_norm.png`
  - Read: rows = true class; columns = predicted. Biggest confusion: **Graduate → Enrolled**.

- **(Optional)** Overall comparison

  - `figures/model_comparison.png` _(if generated in the report notebook)_

### 3) Why the models decide (feature importance & SHAP)

- **Importance bars:**

  - RF: `figures/rf_feature_importance.png`
  - XGB: `figures/xgb_feature_importance_gain.png`

- **Global SHAP (beeswarm):**

  - RF: `figures/shap_summary_rf.png`
  - XGB: `figures/shap_summary_xgb.png`

- **Top-10 SHAP tables (for clean ranking):**

  - RF: `reports/random_forest_shap_importance.csv`
  - XGB: `reports/xgboost_shap_importance.csv`

- **Takeaway:** **engagement + approval/grades + tuition** drive outcomes. Financial pressure and age add signal; demographics are comparatively weaker.

---

## Files you may want to open first

- **High-level text summaries**

  - `reports/final_summary.md` – short narrative of the results
  - `reports/results.md` – EDA bullets (shape, missingness, class mix)

- **Metrics (exact numbers)**

  - `reports/ensemble_metrics.json`
  - `reports/xgb_metrics.json`
  - `reports/rf_metrics.json`

- **Explanations**

  - `reports/random_forest_shap_importance.csv`
  - `reports/xgboost_shap_importance.csv`

- **Figures**

  - EDA: `eda_target.png`, `eda_histograms.png`, `eda_corr_heatmap.png`
  - Performance: `rf_confusion_norm.png`, `xgb_confusion_norm.png`, _(ensemble)_
  - Drivers: `rf_feature_importance.png`, `xgb_feature_importance_gain.png`, `shap_summary_rf.png`, `shap_summary_xgb.png`
  - _(Optional)_ `model_comparison.png`, `rf_top_shap_features.png`, `xgb_top_shap_features.png`

---

## What to remember (for discussions)

- **Model choice:** XGB > RF; **ensemble best**.
- **Operational levers:** address **tuition payment issues**, support students with **low approval rate/grades**, and monitor **engagement_index**.
- **Limitations:** the “Dropout” label means **not enrolled at this time**, which can include transfers/temporary breaks; **Graduate vs Enrolled** separation is inherently hard mid-path.
