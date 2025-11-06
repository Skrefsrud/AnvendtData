# Final Results Summary

**Random Forest**
- Accuracy: 0.718
- Precision (macro): 0.665
- Recall (macro): 0.660
- F1 (macro): 0.660

**XGBoost**
- Accuracy: 0.732
- Precision (macro): 0.677
- Recall (macro): 0.677
- F1 (macro): 0.676

**Ensemble**
- Accuracy: 0.749
- Precision (macro): 0.686
- Recall (macro): 0.670
- F1 (macro): 0.676

**Top Predictors (SHAP) — Random Forest:**
1. engagement_index
2. overall_approval_rate
3. total_approved_units
4. average_grade
5. Scholarship holder

**Top Predictors (SHAP) — XGBoost:**
1. overall_approval_rate
2. total_approved_units
3. Age at enrollment
4. average_grade
5. Scholarship holder

**Interpretation:** Academic performance and tuition compliance dominate student outcome prediction.