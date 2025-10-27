Love it. Here’s a **lean, step-by-step roadmap** you can follow from zero → results, using the multi-notebook layout and a tiny `/src` helpers package. It’s opinionated, practical, and keeps you out of notebook spaghetti.

---

# 🧭 Project Roadmap (step-by-step)

## 0) Scaffold the project

```
/data/raw/                  # original CSV only (read-only)
/data/processed/            # cleaned, engineered, split data
/notebooks/
  01_eda.ipynb
  02_prepare_features.ipynb
  03_train_rf_baseline.ipynb
  04_train_xgb.ipynb
  05_interpretation_shap.ipynb
  06_report_figures.ipynb
/src/
  __init__.py
  config.py                 # seeds, paths, feature lists, label map
  io_utils.py               # load/save helpers
  preprocessing.py          # cleaning + encoding
  features.py               # engineered columns
  eval_utils.py             # metrics, plots, tables
/models/                    # saved .pkl
/figures/                   # png/svg for the report
/reports/results.md
```

Install:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn joblib
```

---

## 1) 01_eda.ipynb — verify assumptions & visualize

**Goal:** lock down data understanding with evidence you can paste into the report.

1. Load `/data/raw/graduation_dataset.csv`.
2. `df.info()`, `df.describe()`, `df.isna().sum()`.
3. Class balance: `df['Target'].value_counts(normalize=True)`.
4. Quick plots:

   - Target countplot
   - Histograms for key continuous vars (grades, age, GDP, inflation, unemployment)
   - Pairwise correlations heatmap (only continuous)

5. Save **figures** to `/figures/eda_*` + write bullets to `/reports/results.md`:

   - “No missing values confirmed”
   - “Class imbalance: X% Graduate, Y% Dropout, Z% Enrolled”
   - “Grades right-skewed”, etc.

**Deliverables:** 3–5 clear plots + bullets confirming your doc’s claims.

---

## 2) 02_prepare_features.ipynb — minimal preprocessing + feature engineering

**Goal:** build the modeling table, consistent with your plan.

1. **Label-encode Target** (Dropout/Enrolled/Graduate → 0/1/2). Persist the mapping in `/src/config.py`.
2. **Feature engineering** (per your spec):

   - `average_grade = mean(grade_sem1, grade_sem2)`
   - `total_approved_units = approved_sem1 + approved_sem2`
   - `overall_approval_rate = total_approved_units / (enrolled_sem1 + enrolled_sem2)` (guard div/0)

3. **Column selection**
   Keep the focused set you defined (academic + financial + key demo + economic). Optionally keep a “full” version for ablation later.
4. **Sanity checks:** value ranges, no NaNs introduced, rates in [0,1], grades in [0,20].
5. **Persist** a single clean dataframe to `/data/processed/modeling.parquet`.

Move reusable bits into `/src`:

- `preprocessing.py`: `encode_target(df)`, `validate_ranges(df)`
- `features.py`: `add_engineered_features(df)`
- `io_utils.py`: `save_df(df, path)`

**Deliverables:** `/data/processed/modeling.parquet` + updated `/src`.

---

## 3) 02_prepare_features.ipynb (end) — stratified split + save folds

**Goal:** leakage-safe splits you’ll reuse everywhere.

1. `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`
2. Save:

   - `/data/processed/X_train.parquet`, `/data/processed/y_train.parquet`
   - `/data/processed/X_test.parquet`, `/data/processed/y_test.parquet`

3. Log class ratios in each split to `/reports/results.md`.

---

## 4) 03_train_rf_baseline.ipynb — baseline that “just works”

**Goal:** a strong, simple benchmark and the first pass at “what matters”.

1. Load split data.
2. Train:

   ```python
   from sklearn.ensemble import RandomForestClassifier
   rf = RandomForestClassifier(
       n_estimators=300, max_depth=20,
       min_samples_split=4, min_samples_leaf=2,
       max_features='sqrt',
       class_weight='balanced',
       random_state=42, n_jobs=-1
   ).fit(X_train, y_train)
   ```

3. Evaluate:

   - Accuracy, macro F1, per-class precision/recall/F1
   - Confusion matrix

4. **Feature importance** (top 10 barh). Expect academic signals to dominate.
5. Persist:

   - `joblib.dump(rf, 'models/rf_baseline.pkl')`
   - Save metrics JSON to `/reports/rf_metrics.json`
   - Save plots to `/figures/rf_*`

**Deliverables:** metrics + importances you can already discuss.

---

## 5) 04_train_xgb.ipynb — primary model + light tuning

**Goal:** beat (or match) RF with explainable tree-boosting.

1. Train starter config:

   ```python
   from xgboost import XGBClassifier
   xgb = XGBClassifier(
       n_estimators=500, learning_rate=0.05,
       max_depth=6, subsample=0.8, colsample_bytree=0.8,
       eval_metric='mlogloss', random_state=42, n_jobs=-1
   ).fit(X_train, y_train)
   ```

2. Evaluate same metrics/plots as RF.
3. (Optional) Light hyperparam sweep (3–5 combos) with `StratifiedKFold` (no grid-hell).
4. Persist best model + metrics/figures.

**Deliverables:** side-by-side comparison table (RF vs XGB) saved to `/reports/results.md`.

---

## 6) 05_interpretation_shap.ipynb — actionable insights

**Goal:** produce **trustworthy** explanations for interventions.

1. SHAP on the **best model** (likely XGB, but RF works too):

   ```python
   import shap
   explainer = shap.TreeExplainer(xgb)
   sv = explainer.shap_values(X_test)
   shap.summary_plot(sv, X_test, show=False); plt.savefig('figures/shap_summary.png')
   ```

2. **Class-specific** insights (e.g., Dropout vs Graduate):

   - SHAP class index: plot beeswarm for the dropout class only.

3. **Partial dependence / SHAP dependence** for top features:

   - `overall_approval_rate`, `average_grade`, `tuition_up_to_date`, `debtor`

4. Write concrete takeaways in `/reports/results.md`:

   - “Low approval rate and unpaid tuition are the strongest dropout drivers”
   - “Thresholds where risk spikes (e.g., avg_grade < 11)”

**Deliverables:** SHAP summary + 2–3 dependence plots with bullet insights.

---

## 7) 06_report_figures.ipynb — clean, publication-ready outputs

**Goal:** final plots/tables sized and styled for the report.

- Confusion matrices (normalized + counts)
- Metrics table (RF vs XGB)
- Top features bar chart
- SHAP summary and 2 dependence plots
- Save in vector (SVG/PDF) + PNG

---

## 8) Robustness & ablations (quick, valuable)

**Goal:** prove choices weren’t arbitrary.

- **Ablation A:** Full vs reduced feature set → does performance drop with fewer features?
- **Ablation B:** `class_weight=None` vs `'balanced'`
- **Ablation C:** Keep semester-specific features vs aggregated ones
- **Ablation D (optional):** Stratified 5-fold CV performance variance

Write 3–5 bullets on what changed (or didn’t).

---

## 9) Finalize narrative (CRISP-DM hooks)

**Goal:** make the report write itself from artifacts.

- **Business understanding →** “Early identification of at-risk students for targeted interventions”
- **Data understanding →** EDA figures + class imbalance
- **Preparation →** feature engineering table + selected features rationale
- **Modeling →** RF baseline, XGB primary, hyperparams (brief)
- **Evaluation →** metrics table + confusion matrix
- **Deployment/Next steps →** monitoring, fairness checks, periodic re-train

---

## 10) (Optional) Repro/automation

- A tiny `00_run_all.ipynb` that `%run`s notebooks in order
- Or a `Makefile`:

  ```makefile
  all: eda prep rf xgb shap figs
  eda:  notebook-run notebooks/01_eda.ipynb
  prep: notebook-run notebooks/02_prepare_features.ipynb
  rf:   notebook-run notebooks/03_train_rf_baseline.ipynb
  xgb:  notebook-run notebooks/04_train_xgb.ipynb
  shap: notebook-run notebooks/05_interpretation_shap.ipynb
  figs: notebook-run notebooks/06_report_figures.ipynb
  ```

---

## Heads-up vs our chat (differences to watch)

- **Scaling:** still **not needed** (trees). Don’t add it.
- **One-hot encoding:** still **not needed** for trees given integer codes.
- **Class imbalance:** prefer `class_weight='balanced'` (no SMOTE unless metrics demand it).
- **Feature pruning:** your plan drops weakly correlated demo vars — fine, but keep one ablation where you **don’t** drop them to prove it doesn’t help.
- **Interpretability:** add **SHAP** (stronger than plain feature importance) — this is an upgrade from the doc.

---

## Done-right checklist

- [ ] Modeling dataset saved once → reused everywhere
- [ ] Stratified split with fixed seed
- [ ] RF baseline persisted + metrics JSON
- [ ] XGB model persisted + metrics JSON
- [ ] Side-by-side metrics table
- [ ] SHAP summary + dependence plots with written takeaways
- [ ] Ablation notes (2–3 quick runs)
- [ ] Figures exported (PNG + SVG/PDF)
- [ ] Results woven into CRISP-DM report

If you want, I can spit out **starter code files** for `/src/*.py` and empty notebook headers so you can just drop them in and go.
