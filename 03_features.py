# Databricks notebook source
# MAGIC %md
# MAGIC # Volve PI Historian — Notebook 03: Feature Engineering
# MAGIC
# MAGIC Builds the ML feature matrix by joining sensor readings with upset labels and
# MAGIC computing rolling window statistics.
# MAGIC
# MAGIC **Implementation note**: All rolling computations are done in **pandas** after
# MAGIC pulling the joined table (~158K rows). Spark global window functions (no partition key)
# MAGIC move all data to a single executor and cause OOM/timeout on serverless clusters.
# MAGIC Pandas rolling is faster, simpler, and has no partition overhead for a single time series.
# MAGIC
# MAGIC **Feature groups**:
# MAGIC 1. **Rolling stats** (mean, std, min, max, range) over 30-min, 2h, 6h, 24h windows
# MAGIC 2. **Rate of change** (relative, per window) for flow, pressure, temp and machinery tags
# MAGIC 3. **Cross-tag ratios** (pressure spread, flow imbalance, watercut proxy)
# MAGIC 4. **Sensor quality score** (fraction of gap-filled or bad-quality readings)
# MAGIC 5. **Calendar features** (hour of day, day of week) — maintenance patterns

# COMMAND ----------

SENSOR_TABLE = "workspace.volve_ml.pi_sensors_5min"
LABEL_TABLE  = "workspace.volve_ml.upset_labels"
OUTPUT_TABLE = "workspace.volve_ml.feature_matrix"

# Target label to use for training (can change to upset_4h or upset_24h)
TARGET_LABEL = "upset_12h"

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import functions as F

# COMMAND ----------
# MAGIC %md ## Load and join tables

# COMMAND ----------

sensors = spark.table(SENSOR_TABLE)
labels  = spark.table(LABEL_TABLE).select(
    "ts", "is_upset", "upset_4h", "upset_12h", "upset_24h", "baseline_7d"
)

df_spark = sensors.join(labels, on="ts", how="inner")
print(f"Joined rows: {df_spark.count():,}")

# Pull to pandas — ~158K rows, well within memory limits
pdf = df_spark.orderBy("ts").toPandas()
pdf = pdf.sort_values("ts").reset_index(drop=True)
print(f"Pulled to pandas: {len(pdf):,} rows × {len(pdf.columns)} columns")

# COMMAND ----------
# MAGIC %md ## Define feature columns and rolling windows

# COMMAND ----------

# Continuous sensor columns to compute rolling stats for
SENSOR_COLS = [
    "flow_hc_315", "flow_hc_914", "flow_hc_944",
    "flow_hc_315_corr", "flow_wat_315",
    "dhp_w12_002", "dhp_w12_003", "dhp_w12_004", "dhp_w12_005",
    "dhp_w12_006", "dhp_w12_010", "dhp_w12_013", "dhp_w12_016",
    "dht_w12_002", "dht_w12_003", "dht_w12_004", "dht_w12_005",
    "dht_w12_006", "dht_w12_010", "dht_w12_013", "dht_w12_016",
    # Topside machinery — G-21 rotating equipment (speed + vibration)
    "mach_spd_g21a", "mach_spd_g21b",
    "mach_vib_183", "mach_vib_186", "mach_vib_190", "mach_vib_191",
    "mach_vib_283", "mach_vib_286", "mach_vib_290", "mach_vib_291",
]

# Quality columns (only those that exist in the data)
QUALITY_COLS = [f"{c}_q" for c in SENSOR_COLS if f"{c}_q" in pdf.columns]

# Rolling windows in 5-min steps
WINDOWS = {
    "30m":  6,
    "2h":   24,
    "6h":   72,
    "24h":  288,
}

# Restrict to columns that actually exist (machinery tags may be absent if ingest failed)
SENSOR_COLS = [c for c in SENSOR_COLS if c in pdf.columns]
print(f"Sensor columns present: {len(SENSOR_COLS)}")

# COMMAND ----------
# MAGIC %md ## Rolling statistics (pandas)
# MAGIC
# MAGIC Uses a trailing window (min_periods=1 so early rows get partial stats rather than NaN).
# MAGIC The first 24h of data is dropped at the end to remove partially-initialised rows.

# COMMAND ----------

for col in SENSOR_COLS:
    for win_name, win_steps in WINDOWS.items():
        w = pdf[col].rolling(window=win_steps, min_periods=1)
        pdf[f"{col}_{win_name}_mean"]  = w.mean()
        pdf[f"{col}_{win_name}_std"]   = w.std()
        pdf[f"{col}_{win_name}_min"]   = w.min()
        pdf[f"{col}_{win_name}_max"]   = w.max()
        pdf[f"{col}_{win_name}_range"] = (
            pdf[f"{col}_{win_name}_max"] - pdf[f"{col}_{win_name}_min"]
        )

print(f"After rolling stats: {len(pdf.columns)} columns")

# COMMAND ----------
# MAGIC %md ## Rate of change features (pandas)

# COMMAND ----------

ROC_COLS = [
    "flow_hc_315", "flow_hc_914", "flow_hc_944",
    "dhp_w12_002", "dhp_w12_004", "dhp_w12_006",
    "dht_w12_002", "dht_w12_004",
    # Machinery: speed rate-of-change captures acceleration/instability events
    "mach_spd_g21a", "mach_spd_g21b",
    # Vibration rate-of-change captures sudden bearing deterioration
    "mach_vib_183", "mach_vib_283",
]

ROC_COLS = [c for c in ROC_COLS if c in pdf.columns]

for col in ROC_COLS:
    for win_name, win_steps in [("30m", 6), ("2h", 24), ("6h", 72)]:
        lag_val = pdf[col].shift(win_steps)
        pdf[f"{col}_roc_{win_name}"] = (pdf[col] - lag_val) / (lag_val.abs() + 1e-6)

print(f"After rate-of-change: {len(pdf.columns)} columns")

# COMMAND ----------
# MAGIC %md ## Cross-tag ratio features

# COMMAND ----------

# Water cut proxy: water flow / total HC flow
if "flow_wat_315" in pdf.columns and "flow_hc_315" in pdf.columns:
    pdf["watercut_proxy"] = (
        pdf["flow_wat_315"] / (pdf["flow_hc_315"] + pdf["flow_wat_315"] + 1e-6)
    )

# Pressure spread across downhole sensors (max - min)
dhp_cols = [c for c in SENSOR_COLS if c.startswith("dhp_")]
if len(dhp_cols) >= 2:
    pdf["dhp_spread"] = pdf[dhp_cols].max(axis=1) - pdf[dhp_cols].min(axis=1)

# Flow imbalance between the two primary surface meters
if "flow_hc_315" in pdf.columns and "flow_hc_914" in pdf.columns:
    pdf["flow_imbalance"] = (
        (pdf["flow_hc_315"] - pdf["flow_hc_914"]).abs()
        / (pdf["flow_hc_315"] + pdf["flow_hc_914"] + 1e-6)
    )

# Machinery speed ratio (train A / train B) — divergence may indicate problem
if "mach_spd_g21a" in pdf.columns and "mach_spd_g21b" in pdf.columns:
    pdf["mach_spd_ratio"] = (
        pdf["mach_spd_g21a"] / (pdf["mach_spd_g21b"] + 1e-6)
    )

# COMMAND ----------
# MAGIC %md ## Sensor quality aggregate feature

# COMMAND ----------

if QUALITY_COLS:
    pdf["overall_quality"] = pdf[QUALITY_COLS].mean(axis=1)
else:
    pdf["overall_quality"] = 1.0

# COMMAND ----------
# MAGIC %md ## Calendar features

# COMMAND ----------

ts = pd.to_datetime(pdf["ts"])
pdf["hour_of_day"] = ts.dt.hour
pdf["day_of_week"]  = ts.dt.dayofweek   # 0=Mon, 6=Sun (pandas convention)
pdf["month"]        = ts.dt.month
pdf["is_weekend"]   = ts.dt.dayofweek.isin([5, 6]).astype(int)

# COMMAND ----------
# MAGIC %md ## Deviation from rolling baseline

# COMMAND ----------

if "baseline_7d" in pdf.columns:
    pdf["flow_vs_baseline"] = (
        (pdf["flow_hc_315"] - pdf["baseline_7d"]) / (pdf["baseline_7d"] + 1e-6)
    )

# COMMAND ----------
# MAGIC %md ## Drop warm-up rows and raw sensor columns

# COMMAND ----------

# Drop the first 24h where rolling features are not yet fully populated
warmup_cutoff = pd.Timestamp("2019-12-11")
pdf = pdf[pd.to_datetime(pdf["ts"]) >= warmup_cutoff].copy()

# Drop rows where the target label is null
pdf = pdf.dropna(subset=[TARGET_LABEL])

# Drop raw sensor columns — model only sees engineered features
raw_cols_to_drop = [c for c in SENSOR_COLS + QUALITY_COLS if c in pdf.columns]
pdf = pdf.drop(columns=raw_cols_to_drop)

print(f"Feature matrix: {len(pdf):,} rows × {len(pdf.columns)} columns")
print(f"Label positive rate ({TARGET_LABEL}): {pdf[TARGET_LABEL].mean():.3f}")

# COMMAND ----------
# MAGIC %md ## Write feature matrix

# COMMAND ----------

feat_df = spark.createDataFrame(pdf)

(
    feat_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE)
)

print(f"Written to {OUTPUT_TABLE}")

# Quick label balance check
spark.sql(f"""
SELECT
  SUM(upset_4h)  / COUNT(*) as pct_upset_4h,
  SUM(upset_12h) / COUNT(*) as pct_upset_12h,
  SUM(upset_24h) / COUNT(*) as pct_upset_24h
FROM {OUTPUT_TABLE}
""").show()
