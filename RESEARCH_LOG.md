# Volve Upset Prediction — Research Log

A running record of what we built, what we learned, and how the project evolved.

---

## Phase 1: Data and framing

### Starting point

The Equinor Volve Data Village catalog on Databricks contains 18 months of PI historian data (Nov 2019 – Apr 2021) from a North Sea production platform. The goal: build an early warning model for production upsets — periods where hydrocarbon flow drops significantly — with enough lead time for operators to intervene.

### Field identity correction

The project was initially framed as using data from the **Volve** field. Late in the work we identified this is incorrect: the Volve field reached COP in September 2016 and cannot have produced sensor data from 2019–2021. Examining the data path in notebook 01 (`PI System Manager Sleipner/sensordata`) makes clear the PI historian data originates from the **Sleipner** field complex — an active Equinor gas/condensate operation on the Norwegian shelf. The Databricks catalog bundles Sleipner PI data alongside the original Volve well and production documents. The operational events modelled here (including the Feb–Mar 2021 12-day shutdown) are real Sleipner events. The "Volve" naming is retained throughout as it matches the catalog convention.

### Choosing a label

The primary flow tag `G-18-FE-315.TotalHCMassFlowRate` (surface HC mass flow rate) was selected as the ground truth. An upset is defined as flow dropping more than **20% below its 7-day rolling baseline**, with the baseline requiring at least 24 hours of history before labelling begins.

Three forward-looking binary labels were created:
- `upset_4h` — will an upset occur in the next 4 hours?
- `upset_12h` — in the next 12 hours?
- `upset_24h` — in the next 24 hours?

Key upset events in the dataset:
- Mar 2020: 3-day full shutdown
- Jun 2020: 2-week extended partial upset
- Sep 2020: 6-day full shutdown
- Feb–Mar 2021: 12-day extended shutdown (the largest event)

### Why not use export flow as the label?

We explored `G-23-FT___103_` and `G-23-FT___203_` as potential plant-wide export labels, reasoning that a meter further downstream might better capture whole-platform upsets. Querying the raw data revealed a 13× step increase in G-23-FT on 7 March 2020 — exactly when `G-18-FE-315` drops to zero for the March 2020 shutdown. G-23-FT is **gas injection/reinjection flow**: it goes up when oil production stops, not down. Unsuitable as a label. `G-18-FE-315` remains the correct choice.

---

## Phase 2: Initial feature set and first model

### Features

The first model used only downhole sensors as features:
- 8 downhole pressure sensors: `W-12-PT___002_` through `W-12-PT___016_` (SLB sub-system)
- 8 downhole temperature sensors: `W-12-TT___002_` through `W-12-TT___016_` (SLB sub-system)

Rolling statistics (mean, std, min, max, range) over 30-minute, 2h, 6h, and 24h windows were computed for each sensor. Surface flow tags were deliberately excluded — all surface meters drop simultaneously during an upset, so including them would teach the model to detect ongoing events rather than predict future ones.

### Key finding: the lookahead gap problem

The initial label window ran from `t+0` to `t+Xh`. This meant that once an upset started, the label was already 1 at `t=0`, and the model could score high simply by detecting that an upset was already in progress. The SHAP values for that run pointed strongly to downhole temperature (`dht_w12_004`, `dht_w12_016`) which do respond immediately to flow changes — a hint the model was detecting rather than predicting.

**Fix**: shifted all label windows to start at `t+2h` (24 steps at 5-min resolution) instead of `t+0`. The label window for `upset_12h` became `[t+2h, t+14h]`. The model must now find precursor patterns — signals that change before the upset — not just correlates of the upset itself.

---

## Phase 3: Adding topside machinery

### Exploration

We asked whether the PI historian contained topside equipment signals that could act as independent leading indicators. The SLT sub-system (1141-SLT) was explored and found to contain G-21 rotating equipment tags:

- **Speed transmitters (SIT)**: `G-21-SIT__187_` and `G-21-SIT__287_` — pump/compressor RPM for two parallel trains (A and B), range 0–3,300 RPM
- **Vibration transmitters (ZIT)**: up to 8 shaft/bearing displacement sensors across both trains

The amine gas sweetening system, seawater cooling, and P-20/P-63 gas compression trains (GDR sub-system) were also examined. The GDR tags cover Mar–Apr 2021 only and are too sparse for training. Amine and cooling water instrumentation are absent from the PI data entirely.

### Why add machinery?

Production upsets can originate topside rather than downhole. A compressor surge, pump trip, or bearing failure in the G-21 train may cause a surface flow dropout with no warning from wellbore sensors. Speed instability or vibration growth in the hours before a trip could give earlier warning than downhole sensors alone. Crucially, machinery signals are **independent** of wellbore conditions — they can degrade while downhole pressure and temperature appear normal.

### Changes made

- Notebook 01: added 10 machinery tags (`mach_spd_g21a/b`, `mach_vib_183/186/190/191/283/286/290/291`)
- Notebook 03: added machinery to `SENSOR_COLS` (rolling stats), `ROC_COLS` (rate of change), and added `mach_spd_ratio` as a cross-tag feature (speed divergence between trains A and B)

---

## Phase 4: Engineering fixes

### Spark global window timeout

Notebook 03 originally computed rolling features using Spark Window functions without a `partitionBy` key. On a Databricks serverless cluster, a global window forces all ~158K rows onto a single executor partition, causing `RETRIES_EXCEEDED` and `SparkConnectGrpcException` timeouts.

**Fix**: rewrote notebook 03 to pull the joined table to pandas (`toPandas()`) and compute all rolling features using `pandas.Series.rolling()` and `.shift()`. At ~158K rows the dataset is trivially small for pandas; all rolling computation finishes in seconds with no partition overhead. The result is then written back via `spark.createDataFrame()`.

### DataFrame fragmentation warning

Adding ~700 new columns one at a time inside a loop (`pdf[col] = series`) triggers repeated internal DataFrame copies and a `PerformanceWarning: DataFrame is highly fragmented`.

**Fix**: collect all new Series in a dict (`rolling_new`, `roc_new`), then do one `pd.concat([pdf, pd.DataFrame(new_cols, index=pdf.index)], axis=1)` per feature group.

---

## Phase 5: SHAP analysis — removing month

After re-running with the corrected label window and machinery features, the SHAP bar chart showed `month` as the dominant predictor by a large margin (~1.8 mean |SHAP| vs ~0.7 for the next feature).

**Assessment**: with only one year of training data and the largest upset event (Feb–Mar 2021) happening to occur in winter months, the model has almost certainly learned a calendar artefact rather than a genuine seasonal causal signal. There is no physical mechanism by which calendar month alone determines upset risk on an offshore platform.

**Decision**: remove `month` from calendar features. `hour_of_day`, `day_of_week`, and `is_weekend` were retained — shift-change patterns and maintenance scheduling are plausibly real.

After removing month, the top SHAP features became physically interpretable:
- `dhp_w12_004_24h_min` — minimum downhole pressure over 24h
- `mach_spd_g21b_6h_min` — minimum compressor B speed over 6h
- `dhp_w12_004_24h_mean` — mean downhole pressure over 24h
- `mach_vib_183_24h_min` — minimum vibration over 24h
- Multiple other machinery speed/vibration rolling stats

---

## Phase 6: False alarm diagnosis — sensor failure

### The problem

After removing month, the model generated massive false alarms in the test period (Jan–Apr 2021): 16,176 false alarms against 24,926 normal periods at the 0.5 threshold. Raising the threshold to 0.85 barely helped — precision only improved from 15.5% to 20.9%, and false alarms remained above 9,000. The time series plot showed near-1.0 probability for the entire test window.

### Finding the root cause

A monthly summary query of `dhp_w12_004` and `mach_spd_g21b` from the raw sensor table revealed:

| Month | dhp_004 mean | dhp_004 min |
|---|---|---|
| Nov 2019 – Dec 2020 | 35–72 bar | physical values |
| Jan 2021 | **-13** | **-999** |
| Feb 2021 | **-926** | **-999** |
| Mar–Apr 2021 | **-999** | **-999** |

`-999` is the PI historian sentinel value for a sensor that is offline or returning bad-quality data. `dhp_w12_004` failed progressively from January 2021 and was completely dead from February onwards.

### How it got through

In notebook 01, the resampling aggregation was:
```python
F.mean("Value").alias("value")
```
Bad-quality readings (`Status != 192`) were flagged with `good_quality = 0` but their raw `-999` Value was still included in the mean. The forward-fill step only fills null buckets, not buckets where `-999` was explicitly averaged in. So `-999` flowed into the feature matrix as if it were a real pressure reading.

The model trained on pressures of 35–72 bar and learned low values correlate with upsets. Seeing `-999` in the test period, it scored maximum upset risk continuously.

### Fix

In notebook 01, restrict the mean to good-quality readings only:
```python
F.mean(F.when(F.col("good_quality") == 1, F.col("Value"))).alias("value")
```
Bad-quality readings now contribute null to the bucket, which the forward-fill then handles correctly — propagating the last genuine reading or leaving NaN if the sensor has been dead too long.

The full pipeline must be re-run (01 → 02 → 03 → 04) after this change.

---

## Current state

| Notebook | Status |
|---|---|
| 01_ingest.py | Fixed: bad-quality readings excluded from mean |
| 02_labels.py | Fixed: 2h minimum lookahead gap (MIN_LEAD_STEPS = 24) |
| 03_features.py | Fixed: pandas rolling, machinery features, month removed |
| 04_train_evaluate.py | Pending re-run after pipeline fix |

---

## Phase 7: SLB downhole sensor audit — most sensors are dead

Re-running after the quality fix still produced massive false alarms (18,433 / 24,929 normal periods at 0.5 threshold), with ROC-AUC actually dropping to 0.536 vs 0.601 previously. Something else was wrong.

### Querying all downhole sensors by month

A monthly summary query across all 8 downhole pressure and 8 downhole temperature sensors revealed:

**Downhole pressure (dhp):**

| Sensor | Status |
|---|---|
| dhp_002 | Stuck at **31.25** for every month — constant, never varies |
| dhp_003, 005, 006, 010, 013 | All **0.0** throughout — completely dead |
| dhp_004 | Only active sensor, but fails from Jan 2021 with -999 sentinel |
| dhp_016 | Real readings through Jan–Feb 2021; fails Mar 2021 |

**Downhole temperature (dht):**

| Sensor | Status |
|---|---|
| dht_002 | Stuck at **62.5** — constant |
| dht_003, 005, 006, 010, 013 | All **0.0** — dead |
| dht_004 | Same physical gauge as dhp_004 — also fails from Jan 2021 |
| dht_016 | Consistently **negative** values (-17 to -152°C) — broken throughout |

Out of 16 downhole sensors, only **two** ever had real varying data (dhp_004 and dht_004, likely the same physical gauge). Both are dead in the test period. dhp_016 has real readings in Jan–Feb 2021 but fails in March.

### Why the quality fix made things worse

With the quality fix applied, dhp_004 values in the test period are no longer -999 but are forward-filled from the last good reading (~35 bar, Dec 2020). This is a problem: in training, 35 bar corresponds to the late-2020 low-flow / pre-shutdown period. A forward-filled stale constant at 35 bar throughout Jan–Apr 2021 looks like "sustained pre-shutdown conditions" to the model, driving continuous high upset probability — arguably worse than the -999 which at least was out-of-range.

### Fix

Removed all dead, stuck, and failing sensors from SENSOR_COLS. The model now uses only:
- **dhp_016** — the one pressure sensor with real test-period data
- **G-21 machinery** (speed and vibration) — the only reliable signals in the dataset

This means the model is effectively a **pure topside machinery predictor** with one downhole pressure input. That is a more honest reflection of what the SLB sensor suite actually delivers in this dataset.

**Open questions / next steps**
- Re-run notebooks 03 and 04 with cleaned sensor list and assess false alarm rate
- If false alarms persist, the issue is likely the machinery signals themselves reflecting a genuinely degraded operating regime in Jan–Apr 2021 (end of field life), rather than a data quality problem
- Consider whether temporal smoothing / debounce on the alert output is the appropriate operational fix rather than further feature engineering

---

*Last updated: March 2026*
