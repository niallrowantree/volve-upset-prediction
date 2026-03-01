# Databricks notebook source
# MAGIC %md
# MAGIC # Volve PI Historian — Notebook 02: Label Upset Events
# MAGIC
# MAGIC Defines "production upset" labels from the primary HC flow tag.
# MAGIC
# MAGIC **Strategy**: A period is labelled an upset when the flow rate drops >20% relative
# MAGIC to a 7-day rolling baseline.  We create three forward-looking binary labels:
# MAGIC - `upset_4h`  — will an upset occur in the next 4 hours?
# MAGIC - `upset_12h` — will an upset occur in the next 12 hours?
# MAGIC - `upset_24h` — will an upset occur in the next 24 hours?
# MAGIC
# MAGIC This framing lets us ask "how far ahead can we predict?"

# COMMAND ----------

INPUT_TABLE  = "workspace.volve_ml.pi_sensors_5min"
OUTPUT_TABLE = "workspace.volve_ml.upset_labels"

# Upset definition thresholds
DROP_THRESHOLD_PCT = 0.20   # flow must fall >20% below 7-day rolling baseline
BASELINE_WINDOW_HOURS = 7 * 24  # 7 days in hours
MIN_BASELINE_HOURS = 24         # need at least 24h of data before labelling starts

# Forward-look windows (in 5-min steps)
LOOKAHEAD = {
    "upset_4h":  4  * 12,   # 48 steps
    "upset_12h": 12 * 12,   # 144 steps
    "upset_24h": 24 * 12,   # 288 steps
}

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

df = spark.table(INPUT_TABLE).select("ts", "flow_hc_315", "flow_hc_315_q")

# COMMAND ----------
# MAGIC %md ## Step 1: Compute 7-day rolling baseline for HC flow

# COMMAND ----------

# Rows in a 7-day window at 5-min resolution
steps_7d = BASELINE_WINDOW_HOURS * 12  # 7*24*12 = 2016
steps_min = MIN_BASELINE_HOURS * 12    # 288

w_baseline = (
    Window.orderBy("ts")
    .rowsBetween(-steps_7d, -1)   # look BACK only (no future leakage)
)

labelled = (
    df
    .withColumn("baseline_7d",   F.mean("flow_hc_315").over(w_baseline))
    .withColumn("baseline_count", F.count("flow_hc_315").over(w_baseline))
    # Only label rows where we have enough history
    .filter(F.col("baseline_count") >= steps_min)
    # Raw upset flag: flow < (1 - threshold) * baseline, and baseline is positive
    .withColumn(
        "is_upset_raw",
        (
            (F.col("baseline_7d") > 10) &    # exclude periods when baseline is already near zero
            (F.col("flow_hc_315") < F.col("baseline_7d") * (1 - DROP_THRESHOLD_PCT))
        ).cast("int")
    )
)

# COMMAND ----------
# MAGIC %md ## Step 2: Clean up label — merge upsets within 2 hours into a single event
# MAGIC
# MAGIC Short blips (brief flow dips) should not be treated as separate events.
# MAGIC We use a rolling max over ±2h to merge nearby upset flags.

# COMMAND ----------

w_merge = Window.orderBy("ts").rowsBetween(-24, 0)   # 24 steps = 2 hours look-back

labelled = (
    labelled
    .withColumn(
        "is_upset",
        F.max("is_upset_raw").over(w_merge)   # sticky — once in upset, stays for 2h
    )
)

# COMMAND ----------
# MAGIC %md ## Step 3: Create forward-looking labels
# MAGIC
# MAGIC For each timestamp t, `upset_Xh = 1` if any row in [t+1, t+X] has `is_upset = 1`.
# MAGIC We use a **lead window** (look forward) — no data leakage risk since labels are
# MAGIC derived from the same tag used only as the target, not as a feature.

# COMMAND ----------

for label_col, n_steps in LOOKAHEAD.items():
    w_forward = Window.orderBy("ts").rowsBetween(1, n_steps)
    labelled = labelled.withColumn(
        label_col,
        F.max("is_upset").over(w_forward)
    )

# Drop rows where forward windows extend beyond our data (nulls at the end)
labelled = labelled.dropna(subset=list(LOOKAHEAD.keys()))

# COMMAND ----------
# MAGIC %md ## Step 4: Event summary — how many distinct upset events?

# COMMAND ----------

# Identify event boundaries (transition from 0→1)
w_prev = Window.orderBy("ts").rowsBetween(-1, -1)
event_summary = (
    labelled
    .withColumn("prev_upset", F.lag("is_upset", 1).over(Window.orderBy("ts")))
    .filter((F.col("is_upset") == 1) & ((F.col("prev_upset") == 0) | F.col("prev_upset").isNull()))
    .agg(F.count("*").alias("n_distinct_upset_events"))
)

print("=== Upset event summary ===")
event_summary.show()

labelled.groupBy("is_upset").count().show()

label_balance = (
    labelled
    .agg(
        F.mean("upset_4h").alias("upset_4h_rate"),
        F.mean("upset_12h").alias("upset_12h_rate"),
        F.mean("upset_24h").alias("upset_24h_rate"),
    )
)
print("Label positive rates:")
label_balance.show()

# COMMAND ----------
# MAGIC %md ## Step 5: Write output

# COMMAND ----------

(
    labelled
    .select("ts", "flow_hc_315", "baseline_7d", "is_upset", *LOOKAHEAD.keys())
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE)
)

print(f"Written to {OUTPUT_TABLE}")
spark.sql(f"SELECT COUNT(*) as rows FROM {OUTPUT_TABLE}").show()

# COMMAND ----------
# MAGIC %md ## Step 6: Visual check — spot the known upsets

# COMMAND ----------

spark.sql(f"""
SELECT ts, flow_hc_315, baseline_7d, is_upset, upset_24h
FROM {OUTPUT_TABLE}
WHERE ts BETWEEN '2020-03-03' AND '2020-03-10'
ORDER BY ts
LIMIT 50
""").show(50, truncate=False)
