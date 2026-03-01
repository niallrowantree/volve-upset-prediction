# Databricks notebook source
# MAGIC %pip install "scipy==1.10.1" "xgboost<2.0" shap

# COMMAND ----------
# MAGIC %md
# MAGIC # Volve PI Historian — Notebook 04: Train, Evaluate & Explain
# MAGIC
# MAGIC Trains an XGBoost classifier to predict production upsets from PI sensor features.
# MAGIC Features are restricted to **independent signals only** (downhole PT/TT, topside
# MAGIC machinery speed/vibration, calendar) — all surface flow meters (FE-315/914/944) and
# MAGIC derived surface-flow features are excluded to avoid circular label leakage.
# MAGIC
# MAGIC **Train period**: Nov 2019 – Dec 2020
# MAGIC **Test period**:  Jan 2021 – Apr 2021

# COMMAND ----------

FEATURE_TABLE  = "workspace.volve_ml.feature_matrix"
TARGET_LABEL   = "upset_12h"        # change to upset_4h / upset_24h to compare horizons

TRAIN_END = "2020-12-31 23:59:59"
TEST_START = "2021-01-01 00:00:00"

# XGBoost hyperparameters (reasonable defaults; tune with hyperopt in future)
XGB_PARAMS = {
    "n_estimators":     400,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 10,   # class imbalance weight (adjust after seeing label rate)
    "random_state":     42,
    "tree_method":      "hist",
    "eval_metric":      "aucpr",
}

# COMMAND ----------

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix
)
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# COMMAND ----------
# MAGIC %md ## Load feature matrix

# COMMAND ----------

df = spark.table(FEATURE_TABLE)

# Columns that are not features
NON_FEATURE_COLS = {"ts", "is_upset", "upset_4h", "upset_12h", "upset_24h", "baseline_7d"}

# Exclude ALL surface flow signals — they all drop simultaneously during an upset
# so any of them just tells the model "an upset is already happening".
# Keep: downhole PT/TT, topside machinery (mach_*), dhp_spread, sensor quality, calendar
CIRCULAR_PREFIXES = (
    "flow_hc_315",      # primary label tag and all derived features
    "flow_hc_315_corr", # corrected mass from same meter
    "flow_hc_914",      # secondary surface flow meter — co-moves with upset
    "flow_hc_944",      # tertiary surface flow meter — co-moves with upset
    "flow_wat_315",     # water flow on same separator — co-moves with upset
    "flow_imbalance",   # ratio of surface meters
    "watercut_proxy",   # surface water fraction
    "flow_vs_baseline", # 315 deviation from baseline
)

FEATURE_COLS = [
    c for c in df.columns
    if c not in NON_FEATURE_COLS
    and not any(c.startswith(p) or c == p for p in CIRCULAR_PREFIXES)
]

print(f"Target: {TARGET_LABEL}")
print(f"Feature columns after excluding circular flow signals: {len(FEATURE_COLS)}")
print(f"Positive rate: {df.filter(f'{TARGET_LABEL} = 1').count() / df.count():.3f}")
print("\nFeatures retained:")
for c in sorted(FEATURE_COLS): print(f"  {c}")

# COMMAND ----------
# MAGIC %md ## Train / test split (time-based — critical for avoiding leakage)

# COMMAND ----------

train_df = df.filter(f"ts <= '{TRAIN_END}'").toPandas()
test_df  = df.filter(f"ts >= '{TEST_START}'").toPandas()

X_train = train_df[FEATURE_COLS].astype(float)
y_train = train_df[TARGET_LABEL].astype(int)
X_test  = test_df[FEATURE_COLS].astype(float)
y_test  = test_df[TARGET_LABEL].astype(int)

print(f"Train: {len(X_train):,} rows  |  positive rate: {y_train.mean():.3f}")
print(f"Test:  {len(X_test):,} rows   |  positive rate: {y_test.mean():.3f}")

# COMMAND ----------
# MAGIC %md ## Train XGBoost

# COMMAND ----------

# Adjust scale_pos_weight from actual class balance
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
XGB_PARAMS["scale_pos_weight"] = neg_count / max(pos_count, 1)
print(f"scale_pos_weight = {XGB_PARAMS['scale_pos_weight']:.2f}")

# ── Train ──────────────────────────────────────────────────────────────────
model = XGBClassifier(**XGB_PARAMS)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# COMMAND ----------
# MAGIC %md ## Evaluate

# COMMAND ----------

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)

print(f"ROC-AUC:  {roc_auc:.4f}")
print(f"PR-AUC:   {pr_auc:.4f}")
print(classification_report(y_test, y_pred, target_names=["Normal", "Upset"]))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"True upsets caught: {tp}/{tp+fn}  ({100*tp/(tp+fn):.0f}%)")
print(f"False alarms:       {fp}/{fp+tn}")

# COMMAND ----------
# MAGIC %md ## Precision-Recall curve + timeline

# COMMAND ----------

prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(rec, prec, lw=2)
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title(f"Precision-Recall Curve (PR-AUC = {pr_auc:.3f})")
axes[0].grid(True, alpha=0.3)

ts_test = test_df["ts"].values
axes[1].plot(ts_test, y_prob, alpha=0.6, lw=0.8, label="Upset probability")
axes[1].fill_between(ts_test, 0, y_test * 0.5, alpha=0.3, color="red", label="Actual upset")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("P(upset)")
axes[1].set_title("Model score vs actual upsets (test period)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
fig.autofmt_xdate()
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------
# MAGIC %md ## SHAP explainability

# COMMAND ----------

# Patch XGBoost 2.x base_score format ([5E-1] → 5E-1) for SHAP compatibility
import json
booster = model.get_booster()
cfg = json.loads(booster.save_config())
bs = cfg['learner']['learner_model_param']['base_score']
if bs.startswith('[') and bs.endswith(']'):
    cfg['learner']['learner_model_param']['base_score'] = bs[1:-1]
    booster.load_config(json.dumps(cfg))

explainer = shap.TreeExplainer(model)
sample_idx = np.random.choice(len(X_test), min(2000, len(X_test)), replace=False)
X_sample = X_test.iloc[sample_idx]
shap_values = explainer.shap_values(X_sample)

# Beeswarm summary
fig_shap = plt.figure(figsize=(10, 12))
shap.summary_plot(shap_values, X_sample, show=False, max_display=25)
plt.title(f"SHAP Feature Importance — {TARGET_LABEL}")
plt.tight_layout()
display(fig_shap)
plt.close()

# Bar chart of mean |SHAP|
mean_shap = pd.DataFrame({
    "feature": FEATURE_COLS,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False).head(25)

fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
ax_bar.barh(mean_shap["feature"][::-1], mean_shap["mean_abs_shap"][::-1])
ax_bar.set_xlabel("Mean |SHAP value|")
ax_bar.set_title(f"Top 25 predictors for {TARGET_LABEL}")
plt.tight_layout()
display(fig_bar)
plt.close()

print("\nTop 10 most predictive features:")
print(mean_shap.head(10).to_string(index=False))

# COMMAND ----------
# MAGIC %md ## Operational threshold analysis
# MAGIC
# MAGIC In production, we don't just want AUC — we want to find the probability threshold
# MAGIC that gives acceptable precision (false alarm rate) while maximising recall (catches).

# COMMAND ----------

results = []
for threshold in np.arange(0.1, 0.9, 0.05):
    y_at_t = (y_prob >= threshold).astype(int)
    tp_t = ((y_at_t == 1) & (y_test == 1)).sum()
    fp_t = ((y_at_t == 1) & (y_test == 0)).sum()
    fn_t = ((y_at_t == 0) & (y_test == 1)).sum()
    recall_t    = tp_t / max(tp_t + fn_t, 1)
    precision_t = tp_t / max(tp_t + fp_t, 1)
    results.append({
        "threshold": round(threshold, 2),
        "recall": round(recall_t, 3),
        "precision": round(precision_t, 3),
        "false_alarms": int(fp_t),
        "upsets_caught": int(tp_t),
        "upsets_missed": int(fn_t),
    })

print("\n=== Operational threshold table ===")
print(pd.DataFrame(results).to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Interpretation guide
# MAGIC
# MAGIC **How to read SHAP summary plot**:
# MAGIC - Each dot is one prediction. Position on x-axis = SHAP value (impact on model output)
# MAGIC - Colour = feature value (red=high, blue=low)
# MAGIC - Features ranked by mean |SHAP| — top features are most influential
# MAGIC
# MAGIC **Operational guidance**:
# MAGIC - Choose a threshold from the table above based on acceptable false alarm rate
# MAGIC - Upsets caught at 12h horizon gives operators time to take preventive action
# MAGIC - SHAP top features tell maintenance teams which sensors to watch
