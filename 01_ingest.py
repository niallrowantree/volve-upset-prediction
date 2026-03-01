# Databricks notebook source
# MAGIC %md
# MAGIC # Volve PI Historian — Notebook 01: Ingest & Resample
# MAGIC
# MAGIC Reads raw per-tag parquet files from the Equinor Volve Volume, unions them into a
# MAGIC single long-format table, resamples from irregular PI-compressed timestamps to a
# MAGIC regular 5-minute grid, then pivots wide and writes a Delta table.
# MAGIC
# MAGIC **Coverage**: Nov 2019 – Apr 2021
# MAGIC **Tags ingested**: 3 surface flow (SLT), 8 downhole pressure (SLB), 8 downhole temperature (SLB),
# MAGIC 2 topside speed transmitters (SLT G-21 SIT), up to 8 topside vibration transmitters (SLT G-21 ZIT)

# COMMAND ----------

# Configuration — adjust catalog/schema if needed
CATALOG    = "equinor_asa_volve_data_village"
SCHEMA     = "public"
VOLUME     = "volve"
SENSOR_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/PI System Manager Sleipner/sensordata"

OUTPUT_CATALOG = "workspace"   # change to your target catalog
OUTPUT_SCHEMA  = "volve_ml"
OUTPUT_TABLE   = f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.pi_sensors_5min"

RESAMPLE_FREQ  = "5 minutes"   # target grid interval

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, IntegerType
import pyspark.sql.functions as F

# Schema for every parquet file
PI_SCHEMA = StructType([
    StructField("Time",   TimestampType(), True),
    StructField("Value",  DoubleType(),    True),
    StructField("Status", IntegerType(),   True),
])

# COMMAND ----------
# MAGIC %md ## Tag catalogue
# MAGIC
# MAGIC We select a focused set of tags that cover:
# MAGIC - **Surface HC flow** (primary label source + correlated flows)
# MAGIC - **Downhole pressure** (early indicator of reservoir/wellbore changes)
# MAGIC - **Downhole temperature** (thermal signature of flow regime changes)
# MAGIC - **Topside machinery** (G-21 pump/compressor speed and bearing vibration — independent leading indicators)
# MAGIC
# MAGIC Each entry is (tag_name, volume_sub_path).

# COMMAND ----------

TAGS = {
    # ── Surface flow (SLT) ──────────────────────────────────────────────────
    "flow_hc_315":     "1141-SLT/G-18-FE-315.TotalHCMassFlowRate",
    "flow_hc_914":     "1141-SLT/G-18-FE-914.TotalHCMassFlowRate",
    "flow_hc_944":     "1141-SLT/G-18-FE-944.TotalHCMassFlowRate",
    "flow_hc_315_corr":"1141-SLT/G-18-FE-315.HC.PD1.CorrMass",
    "flow_wat_315":    "1141-SLT/G-18-FE-315.Water.PD1.CorrMass",

    # ── Downhole pressure (SLB) ─────────────────────────────────────────────
    "dhp_w12_002":  "1142-SLB/W-12-PT___002_",
    "dhp_w12_003":  "1142-SLB/W-12-PT___003_",
    "dhp_w12_004":  "1142-SLB/W-12-PT___004_",
    "dhp_w12_005":  "1142-SLB/W-12-PT___005_",
    "dhp_w12_006":  "1142-SLB/W-12-PT___006_",
    "dhp_w12_010":  "1142-SLB/W-12-PT___010_",
    "dhp_w12_013":  "1142-SLB/W-12-PT___013_",
    "dhp_w12_016":  "1142-SLB/W-12-PT___016_",

    # ── Downhole temperature (SLB) ──────────────────────────────────────────
    "dht_w12_002":  "1142-SLB/W-12-TT___002_",
    "dht_w12_003":  "1142-SLB/W-12-TT___003_",
    "dht_w12_004":  "1142-SLB/W-12-TT___004_",
    "dht_w12_005":  "1142-SLB/W-12-TT___005_",
    "dht_w12_006":  "1142-SLB/W-12-TT___006_",
    "dht_w12_010":  "1142-SLB/W-12-TT___010_",
    "dht_w12_013":  "1142-SLB/W-12-TT___013_",
    "dht_w12_016":  "1142-SLB/W-12-TT___016_",

    # ── GDR alarm counts (Mar–Apr 2021 only, used for validation) ────────────
    "alarm_active":    "1163-GDR/B00.Alarm.ActiveCount",
    "alarm_critical":  "1163-GDR/B00.Alarm.CriticalActiveCount",
    "alarm_p20":       "1163-GDR/P-20-B03.Alarm.ActiveCount",
    "alarm_p21":       "1163-GDR/P-21-B02.Alarm.ActiveCount",

    # ── Topside machinery — G-21 rotating equipment (SLT) ────────────────────
    # Speed transmitters (SIT): pump/compressor RPM for two parallel trains
    "mach_spd_g21a":  "1141-SLT/G-21-SIT__187_",
    "mach_spd_g21b":  "1141-SLT/G-21-SIT__287_",
    # Vibration transmitters (ZIT): shaft/bearing displacement, both trains
    # Tags that don't exist in the Volume will be skipped by the error handler below
    "mach_vib_183":   "1141-SLT/G-21-ZIT__183_",
    "mach_vib_186":   "1141-SLT/G-21-ZIT__186_",
    "mach_vib_190":   "1141-SLT/G-21-ZIT__190_",
    "mach_vib_191":   "1141-SLT/G-21-ZIT__191_",
    "mach_vib_283":   "1141-SLT/G-21-ZIT__283_",
    "mach_vib_286":   "1141-SLT/G-21-ZIT__286_",
    "mach_vib_290":   "1141-SLT/G-21-ZIT__290_",
    "mach_vib_291":   "1141-SLT/G-21-ZIT__291_",
}

# COMMAND ----------
# MAGIC %md ## Read and union all tags

# COMMAND ----------

from functools import reduce

tag_dfs = []
failed_tags = []

for tag_name, sub_path in TAGS.items():
    parquet_path = f"{SENSOR_ROOT}/{sub_path}/parquet/"
    try:
        df = (
            spark.read
            .schema(PI_SCHEMA)
            .parquet(parquet_path)
            .withColumn("tag", F.lit(tag_name))
            # PI Status 192 = Good, anything else is degraded/bad quality
            .withColumn("good_quality", (F.col("Status") == 192).cast("int"))
            .select("Time", "tag", "Value", "good_quality")
        )
        tag_dfs.append(df)
        print(f"  OK  {tag_name}")
    except Exception as e:
        failed_tags.append(tag_name)
        print(f"  FAIL {tag_name}: {e}")

print(f"\nLoaded {len(tag_dfs)} tags, {len(failed_tags)} failed: {failed_tags}")
all_tags_df = reduce(lambda a, b: a.union(b), tag_dfs)

# COMMAND ----------
# MAGIC %md ## Resample to regular 5-minute grid
# MAGIC
# MAGIC PI historian uses exception-reporting (step interpolation) — values are only written
# MAGIC when they change beyond a deadband. We resample by:
# MAGIC 1. Snapping each timestamp to the nearest 5-min bucket (floor)
# MAGIC 2. Taking the mean value and fraction of good-quality readings within each bucket
# MAGIC 3. Forward-filling small gaps (up to 1 hour) per tag

# COMMAND ----------

from pyspark.sql.window import Window

# Step 1: snap to 5-min buckets
resampled = (
    all_tags_df
    .withColumn(
        "ts",
        F.date_trunc("minute",
            F.from_unixtime(
                (F.unix_timestamp("Time") / 300).cast("long") * 300
            )
        )
    )
    .groupBy("ts", "tag")
    .agg(
        F.mean("Value").alias("value"),
        F.mean("good_quality").alias("quality_frac"),  # 1.0 = all readings good
        F.count("*").alias("n_raw_readings"),
    )
)

# Step 2: generate a complete time grid for each tag then left-join
# (this fills gaps so forward-fill works correctly)
min_ts, max_ts = resampled.selectExpr("min(ts)", "max(ts)").first()

time_grid = spark.range(
    int(min_ts.timestamp()),
    int(max_ts.timestamp()) + 1,
    5 * 60   # 5 minutes in seconds
).select(
    F.from_unixtime(F.col("id")).cast("timestamp").alias("ts")
)

tag_list = spark.createDataFrame(
    [(t,) for t in TAGS.keys()], ["tag"]
)

full_grid = time_grid.crossJoin(tag_list)

# Step 3: join actuals onto full grid
joined = (
    full_grid
    .join(resampled, on=["ts", "tag"], how="left")
)

# Step 4: forward fill within each tag (up to 12 steps = 1 hour)
w_ffill = (
    Window.partitionBy("tag")
    .orderBy("ts")
    .rowsBetween(Window.unboundedPreceding, 0)
)

filled = (
    joined
    .withColumn("value_ffill",        F.last("value",        ignorenulls=True).over(w_ffill))
    .withColumn("quality_frac_ffill", F.last("quality_frac", ignorenulls=True).over(w_ffill))
    # Mark rows that are gap-filled (no raw reading)
    .withColumn("is_gap_filled", F.col("value").isNull().cast("int"))
    .select(
        "ts",
        "tag",
        F.col("value_ffill").alias("value"),
        F.col("quality_frac_ffill").alias("quality_frac"),
        "is_gap_filled",
    )
)

# COMMAND ----------
# MAGIC %md ## Pivot to wide format (one column per tag)

# COMMAND ----------

# Pivot value
wide_value = (
    filled
    .groupBy("ts")
    .pivot("tag")
    .agg(F.first("value"))
)

# Pivot quality fraction (suffix _q)
wide_quality = (
    filled
    .select("ts", F.col("tag").alias("tag_q"), "quality_frac")
    .groupBy("ts")
    .pivot("tag_q")
    .agg(F.first("quality_frac"))
)
# rename quality columns
for c in wide_quality.columns:
    if c != "ts":
        wide_quality = wide_quality.withColumnRenamed(c, f"{c}_q")

wide = wide_value.join(wide_quality, on="ts", how="left")

# COMMAND ----------
# MAGIC %md ## Write Delta table

# COMMAND ----------

(
    wide
    .withColumn("year", F.year("ts"))
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("year")
    .saveAsTable(OUTPUT_TABLE)
)

print(f"Written to {OUTPUT_TABLE}")
spark.sql(f"SELECT COUNT(*) as rows, MIN(ts) as earliest, MAX(ts) as latest FROM {OUTPUT_TABLE}").show()

# COMMAND ----------
# MAGIC %md ## Quick sanity check

# COMMAND ----------

spark.sql(f"""
SELECT
  ts,
  flow_hc_315,
  flow_hc_914,
  flow_hc_944,
  alarm_active,
  alarm_critical,
  mach_spd_g21a,
  mach_vib_183
FROM {OUTPUT_TABLE}
WHERE ts BETWEEN '2020-03-04' AND '2020-03-10'
ORDER BY ts
""").show(100, truncate=False)
