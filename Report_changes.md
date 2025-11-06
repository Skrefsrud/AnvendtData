# Global fixes (applies in multiple places)

- **Features count:** Change “**36 features**” → “**35 columns**” (4,424 rows; no missing values).
- **Class shares:** Use exact values everywhere you describe imbalance: **Graduate 49.9%, Dropout 32.1%, Enrolled 17.9%**.
- **Age distribution:** Replace claims of “roughly normal” with **“right-skewed with a long tail (most in early 20s)”** (your histogram shows this).

---

# Section-by-section edits

## 3 Method and analysis

### 3.1 Data understanding

**3.1.1 Describing the dataset – replace these lines:**

- “This dataset contains … **4,424 instances and 35 columns**, with **no missing values**.”
- “Class distribution: **Graduate 49.9%, Dropout 32.1%, Enrolled 17.9%**.”

**3.1.2 EDA – tweak wording:**

- Replace: “_Age at enrollment showed a roughly normal distribution_ …”
  **With:** “_Age at enrollment is **right-skewed** with a long tail; most students enroll in their early 20s._”

### 3.2 Data preparation

**3.2.1 Cleaning and preprocessing – label encoding note:**

- You currently include a table mapping Target → numeric codes. Metrics are reported by **class names** (Dropout/Graduate/Enrolled) in your JSONs, so you can **remove the numeric mapping table** to avoid mismatch risk; or explicitly say _“We encoded the target but always report metrics by class name.”_

**3.2.2 Feature engineering – add concrete engineered features used**

- Insert:
  “We engineered and retained several predictors due to observed predictive value: **engagement_index, financial_pressure, grade_change, fail_ratio_1st, fail_ratio_2nd**, alongside the aggregates **overall_approval_rate, total_approved_units, average_grade**.”
  (These show up at the top of SHAP/importance and in your final summary.)

**Class imbalance handling (end of 3.2) – be explicit about what the code did**

- Replace your generic line with:
  “We addressed class imbalance via **class weighting** in Random Forest (**balanced_subsample**) and **per-row sample weights** for XGBoost; we also evaluated **SMOTE** on the training split and selected configurations by cross-validation.”
  _(Metrics files confirm classes reported by name; SMOTE use is consistent with the notebooks’ design.)_

### 3.3 Modeling

**3.3.1 Selection of models – add the ensemble you actually ran**

- Add this sentence at the end:
  “In addition, we evaluated a **soft-voting ensemble** (RF + XGBoost) which ultimately achieved the **best overall accuracy**.”

**3.3.2 Evaluation metrics – no change** (you already describe the right metrics).

---

## 4 Evaluation and interpretation

### 4.1 Model performance

**4.1.1 Overall performance – replace your table with these exact numbers:**

| Metric            | Random Forest |    XGBoost | Ensemble (RF+XGB) |
| ----------------- | ------------: | ---------: | ----------------: |
| **Accuracy**      |    **0.7175** | **0.7322** |        **0.7492** |
| **F1 – Dropout**  |        0.7430 |     0.7436 |            0.7523 |
| **F1 – Graduate** |        0.4183 |     0.4438 |            0.4276 |
| **F1 – Enrolled** |        0.8198 |     0.8392 |            0.8479 |

Sources: RF, XGB, Ensemble metrics JSONs.

**4.1.2 Comparison of RF and XGBoost – adjust the narrative:**

- Replace: “RF performs slightly better overall.”
  **With:**
  “**XGBoost outperforms RF overall** (Accuracy 0.732 vs 0.718; macro-F1 0.676 vs 0.660), and a **soft-voting ensemble is best** with Accuracy **0.749** and macro-F1 **0.676**.”
- Add nuance: “XGBoost improves **Graduate** F1 vs RF (0.444 vs 0.418), while the ensemble boosts **Dropout** and **Enrolled** F1 the most.”

**4.1.3 Confusion patterns – keep your text, but you can add exact recalls**

- Suggested one-liner to pin it down:
  “Graduate recall: **0.46 (RF)**, **0.48 (XGB)**; Enrolled recall: **0.82 (RF)**, **0.83 (XGB)**.”
- (Optional) If you later export it, include an **ensemble confusion matrix** figure and reference it alongside RF/XGB.

**4.1.4 Feature importance and SHAP – update the bullet list to match the evidence**

- Replace your current list with (order doesn’t have to be exact):

  - **engagement_index**
  - **overall_approval_rate**
  - **total_approved_units**
  - **average_grade**
  - **tuition_fees_up_to_date**
  - **financial_pressure**
  - **age_at_enrollment**
    These appear consistently across RF importances, XGB gain importances, and SHAP summaries / CSVs; XGBoost in particular ranks **tuition_fees_up_to_date** and **overall_approval_rate** very highly.

_(Your “Top Predictors” bullets in `final_summary.md` already reflect this direction, so this change harmonizes the report with the figures/CSVs.)_

---

## 4.2 Practical value

- Add a clause calling out the actionable features you now highlight:
  “Signals like **tuition compliance** and **engagement_index** are operationally actionable (financial follow-up, outreach), while **approval rate / grades** point to academic interventions.”

---

## 4.3 Shortcomings

- Keep all current items. Optionally add:
  “**Label fidelity:** ‘Dropout’ reflects ‘not currently enrolled in this program’ and may include transfers/returns; interpret predicted ‘Dropout’ as **near-term non-enrollment**, not permanent exit.” (You already hint this; consider making it explicit.)

---

# Figures & tables to include (so the document matches artifacts)

- **EDA**: `eda_target.png`, `eda_histograms.png`, `eda_corr_heatmap.png`.
- **Performance**: keep RF/XGB confusion matrices; (optional) add an **ensemble** confusion matrix when available.
- **Feature importance**: `rf_feature_importance.png`, `xgb_feature_importance_gain.png`.
- **SHAP**: `shap_summary_rf.png`, `shap_summary_xgb.png`.
- **(Optional nice-to-have)**: bar charts of **Top-10 SHAP** from your CSVs (both models) to accompany the beeswarms.

---

# Tiny “drop-in” sentence swaps (ready to paste)

- **3.1.1:**
  “We use one dataset with **4,424 rows and 35 columns** and **no missing values**. The class distribution is **Graduate 49.9%**, **Dropout 32.1%**, **Enrolled 17.9%**.”

- **3.1.2 (Age):**
  “Age at enrollment is **right-skewed** with a long tail; most students enroll in their early 20s.”

- **3.2 (Imbalance):**
  “RF used **class_weight='balanced_subsample'**; XGBoost used **row sample weights**, and we also evaluated **SMOTE** on the training split.”

- **4.1.2 (Model comparison):**
  “**XGBoost > RF overall** (Acc **0.732** vs **0.718**, macro-F1 **0.676** vs **0.660**), and a **soft-voting ensemble** is best with Acc **0.749** and macro-F1 **0.676**.”

- **4.1.4 (Drivers):**
  “Across models, the strongest drivers are **engagement_index**, **overall_approval_rate**, **total_approved_units**, **average_grade**, **tuition_fees_up_to_date**, **financial_pressure**, and **age_at_enrollment**.”
