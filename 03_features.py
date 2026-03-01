# Databricks notebook source
# MAGIC %md
# MAGIC # Volve PI Historian — Notebook 03: Feature Engineering
# MAGIC
# MAGIC Builds the ML feature matrix by joining sensor readings with upset labels and
# MAGIC computing rolling window statistics.
# MAGIC
# MAGIC **Feature groups**:
# MAGIC 1. **Rolling stats** (mean, std, min, max) over 30-min, 2h, 6h, 24h windows
# MAGIC 2. **Rate of change** (first difference per window) for flow and pressure tags
# MAGIC 3. **Cross-tag ratios** (e.g., surface-to-downhole pressure gradient)
# MAGIC 4. **Sensor quality score** (fraction of gap-filled or bad-quality readings)
# MAGIC 5. **Calendar features** (hour of day, day of week) — maintenance patterns

# COMMAND ----------

SENSOR_TABLE = "workspace.volve_ml.pi_sensors_5min"
LABEL_TABLE  = "workspace.volve_ml.upset_labels"
OUTPUT_TABLE = "workspace.volve_ml.feature_matrix"

# Target label to use for training (can change to upset_4h or upset_24h)
TARGET_LABEL = "upset_12h"

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pyspark.sql.functions as F

sensors = spark.table(SENSOR_TABLE)
labels  = spark.table(LABEL_TABLE).select("ts", "is_upset", "upset_4h", "upset_12h", "upset_24h", "baseline_7d")

# Join on timestamp
df = sensors.join(labels, on="ts", how="inner")

print(f"Joined rows: {df.count():,}")

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
]

# Quality columns
QUALITY_COLS = [f"{c}_q" for c in SENSOR_COLS if f"{c}_q" in sensors.columns]

# Rolling windows in 5-min steps
WINDOWS = {
    "30m":  6,
    "2h":   24,
    "6h":   72,
    "24h":  288,
}

# COMMAND ----------
# MAGIC %md ## Compute rolling statistics

# COMMAND ----------

w_base = Window.orderBy("ts")

feat_df = df

for col in SENSOR_COLS:
    if col not in feat_df.columns:
        continue
    for win_name, win_steps in WINDOWS.items():
        w = w_base.rowsBetween(-win_steps, 0)
        feat_df = (
            feat_df
            .withColumn(f"{col}_{win_name}_mean", F.mean(col).over(w))
            .withColumn(f"{col}_{win_name}_std",  F.stddev(col).over(w))
            .withColumn(f"{col}_{win_name}_min",  F.min(col).over(w))
            .withColumn(f"{col}_{win_name}_max",  F.max(col).over(w))
        )
        # Range (max - min) as a volatility measure
        feat_df = feat_df.withColumn(
            f"{col}_{win_name}_range",
            F.col(f"{col}_{win_name}_max") - F.col(f"{col}_{win_name}_min")
        )

# COMMAND ----------
# MAGIC %md ## Rate of change features
# MAGIC
# MAGIC For flow and pressure, a sudden change rate is often the earliest observable signal.

# COMMAND ----------

ROC_COLS = [
    "flow_hc_315", "flow_hc_914", "flow_hc_944",
    "dhp_w12_002", "dhp_w12_004", "dhp_w12_006",
    "dht_w12_002", "dht_w12_004",
]

for col in ROC_COLS:
    if col not in feat_df.columns:
        continue
    for win_name, win_steps in [("30m", 6), ("2h", 24), ("6h", 72)]:
        lag_col = F.lag(col, win_steps).over(w_base)
        feat_df = feat_df.withColumn(
            f"{col}_roc_{win_name}",
            (F.col(col) - lag_col) / (F.abs(lag_col) + 1e-6)   # relative rate of change
        )

# COMMAND ----------
# MAGIC %md ## Cross-tag ratio features

# COMMAND ----------

# Water cut proxy: water flow / total HC flow
feat_df = feat_df.withColumn(
    "watercut_proxy",
    F.col("flow_wat_315") / (F.col("flow_hc_315") + F.col("flow_wat_315") + 1e-6)
)

# Pressure spread across downhole sensors (max - min)
dhp_cols = [c for c in SENSOR_COLS if c.startswith("dhp_")]
if len(dhp_cols) >= 2:
    feat_df = (
        feat_df
        .withColumn("dhp_spread", F.greatest(*dhp_cols) - F.least(*dhp_cols))
    )

# Flow imbalance between the three surface flow meters
feat_df = feat_df.withColumn(
    "flow_imbalance",
    F.abs(F.col("flow_hc_315") - F.col("flow_hc_914")) /
    (F.col("flow_hc_315") + F.col("flow_hc_914") + 1e-6)
)

# COMMAND ----------
# MAGIC %md ## Sensor quality aggregate features

# COMMAND ----------

# Average quality fraction across all sensor groups
feat_df = feat_df.withColumn(
    "overall_quality",
    sum(F.col(c) for c in QUALITY_COLS if c in feat_df.columns) / len(QUALITY_COLS)
    if QUALITY_COLS else F.lit(1.0)
)

# COMMAND ----------
# MAGIC %md ## Calendar features (captures scheduled maintenance patterns)

# COMMAND ----------

feat_df = (
    feat_df
    .withColumn("hour_of_day",   F.hour("ts"))
    .withColumn("day_of_week",   F.dayofweek("ts"))     # 1=Sun, 7=Sat
    .withColumn("month",         F.month("ts"))
    .withColumn("is_weekend",    (F.dayofweek("ts").isin([1, 7])).cast("int"))
)

# COMMAND ----------
# MAGIC %md ## Deviation from rolling baseline (normalised)

# COMMAND ----------

feat_df = feat_df.withColumn(
    "flow_vs_baseline",
    (F.col("flow_hc_315") - F.col("baseline_7d")) / (F.col("baseline_7d") + 1e-6)
)

# COMMAND ----------
# MAGIC %md ## Drop early rows lacking sufficient history, remove nulls

# COMMAND ----------

# Drop the first 24h where rolling features are not yet fully populated
feat_df = feat_df.filter(F.col("ts") >= F.lit("2019-12-11").cast("timestamp"))

# Drop rows where the target label is null
feat_df = feat_df.dropna(subset=[TARGET_LABEL])

# Drop raw sensor columns — model only sees engineered features
# (keep ts and labels for downstream joining)
raw_cols_to_drop = SENSOR_COLS + QUALITY_COLS
feat_df = feat_df.drop(*[c for c in raw_cols_to_drop if c in feat_df.columns])

print(f"Feature matrix rows: {feat_df.count():,}")
print(f"Columns: {len(feat_df.columns)}")
feat_df.printSchema()

# COMMAND ----------
# MAGIC %md ## Write feature matrix

# COMMAND ----------

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
