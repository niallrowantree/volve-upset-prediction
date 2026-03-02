# Volve Production Upset Prediction

Early warning model for production upsets using PI historian sensor data from the Equinor Volve Data Village on Databricks Unity Catalog.

## Overview

Uses 18 months of PI System time-series data (Nov 2019 – Apr 2021) to train an XGBoost classifier that predicts production upsets 2–14 hours ahead. The model is restricted to **independent signals** — downhole pressure sensors and topside rotating machinery — to provide genuine early warning rather than detection of already-in-progress events.

**Data source**: `equinor_asa_volve_data_village` catalog — Equinor's open dataset available on Databricks via the Data Marketplace.

**Note on field identity**: Despite the catalog name, the PI historian data (path: `PI System Manager Sleipner/sensordata`) originates from the **Sleipner** field complex, not the Volve field. Volve reached COP in September 2016 and could not have produced data from 2019–2021. Sleipner A/B/T is an active Equinor gas/condensate field on the Norwegian shelf; the sensor timestamps and operational events (including the Feb–Mar 2021 12-day shutdown) reflect real Sleipner operations. The project retains the "Volve" naming convention used by the Databricks catalog.

## Project structure

| Notebook | Purpose |
|---|---|
| `01_ingest.py` | Reads ~35 PI sensor tags from Volume parquet files, resamples to 5-min grid, pivots wide, writes `pi_sensors_5min` Delta table |
| `02_labels.py` | Defines upset events as >20% flow drop vs 7-day rolling baseline; creates `upset_4h`, `upset_12h`, `upset_24h` forward-looking binary labels with 2h minimum lead time |
| `03_features.py` | Rolling stats (mean/std/min/max/range) over 30min/2h/6h/24h windows, rate-of-change, cross-tag ratios, sensor quality scores, calendar features |
| `04_train_evaluate.py` | XGBoost classifier, PR-AUC evaluation, SHAP explainability plots, operational threshold table |

## Sensor tags used

**Surface flow (SLT)** — used for label generation only, excluded from model features:
- `G-18-FE-315.TotalHCMassFlowRate` (primary label tag)
- `G-18-FE-914`, `G-18-FE-944` (secondary meters)

**Downhole pressure (SLB)** — model features:
- `W-12-PT___002_` through `W-12-PT___016_` (8 sensors)

**Downhole temperature (SLB)** — model features:
- `W-12-TT___002_` through `W-12-TT___016_` (8 sensors)

**Topside rotating machinery (SLT G-21)** — model features:
- `G-21-SIT__187_`, `G-21-SIT__287_` (pump/compressor speed, trains A and B, 0–3,300 RPM)
- `G-21-ZIT__183_` through `G-21-ZIT__291_` (up to 8 shaft/bearing vibration transmitters)

**GDR alarm counts** — included in Delta table, used for validation only (Mar–Apr 2021 coverage):
- `B00.Alarm.ActiveCount`, `B00.Alarm.CriticalActiveCount`

## Label definition

A timestep is labelled an upset if the HC flow rate drops more than 20% below its 7-day rolling baseline. Forward-looking labels flag any timestep where an upset occurs within a future window, with a **2-hour minimum lead time** — the label window starts at t+2h, not t+0, so the model cannot simply detect an ongoing upset.

| Label | Prediction window | Minimum warning |
|---|---|---|
| `upset_4h` | t+2h to t+6h | 2 hours |
| `upset_12h` | t+2h to t+14h | 2 hours |
| `upset_24h` | t+2h to t+26h | 2 hours |

Key upset events in the dataset:
- Mar 2020: 3-day full shutdown
- Jun 2020: 2-week extended partial upset
- Sep 2020: 6-day full shutdown
- Feb–Mar 2021: 12-day extended shutdown

## PI data exploration findings

The Volve PI historian (4 sub-systems: SLT, SLB, GDR, GKR) was explored to assess which topside instrumentation could act as independent leading indicators:

**What's available**:
- G-21 rotating equipment (SLT): speed transmitters (0–3,300 RPM) and bearing vibration for two parallel machinery trains — these are genuinely independent of wellbore flow
- G-63 gas compression trains (GDR): digital run/stop status for compressors KA01A/B/C, plus stage pressures and temperatures (but sparse coverage)
- G-23 gas flow meters (SLT): dense continuous flow (~13M readings), but these measure gas injection/reinjection and move counter to oil production — unsuitable as labels

**What's absent**:
- Amine gas sweetening system — no amine temperature, level, or foam-detection tags in any sub-system
- Seawater cooling / marine fouling — no cooling water instrumentation
- A clean export meter — G-23-FT moves inversely to production (injection mode), so G-18-FE-315 remains the most reliable upset label

## Setup

### Prerequisites
- Databricks workspace with access to `equinor_asa_volve_data_village` catalog
- Cluster with Python 3.11+
- Notebook 04 installs its own dependencies via `%pip`

### Configuration
In `01_ingest.py`, set your target catalog and schema:
```python
OUTPUT_CATALOG = "workspace"   # your catalog
OUTPUT_SCHEMA  = "volve_ml"    # will be created if it doesn't exist
```
The same values must match in notebooks 02–04.

### Run order
Run notebooks in sequence: `01` → `02` → `03` → `04`. Each writes a Delta table consumed by the next.

### Import into Databricks
Each `.py` file is in Databricks notebook source format. Import via:
**Workspace → (folder) → Import → File → select .py**

## Model design decisions

**Why exclude surface flow features?** All surface flow meters (FE-315, FE-914, FE-944) drop simultaneously during a production upset. Including them makes the model learn "upsets continue once started" rather than predicting them ahead of time. Only downhole sensors, topside machinery, quality metrics, and calendar features are used as model inputs.

**Why add topside machinery?** Production upsets can originate topside (compressor or pump failure) rather than downhole. Speed instability or vibration changes in the G-21 machinery train may precede the surface flow dropout, giving earlier warning than wellbore sensors alone. These signals are independent — machinery can degrade while downhole conditions appear normal.

**Why a 2-hour minimum label lookahead?** The original label window (t+0 to t+12h) meant that when an upset began, the label was already 1 at the onset timestep. The model could score high simply by detecting the start of an event rather than predicting it. Shifting the window to start at t+2h forces the model to find genuine precursors — patterns that appear before the event, not simultaneously with it.

**Why XGBoost?** Interpretability via SHAP is critical for production engineers to act on alerts. SHAP feature attribution shows exactly which sensor is driving each prediction.

**Why time-based train/test split?** Random splits would leak future information across the temporal autocorrelation in the time series. Train: Nov 2019 – Dec 2020. Test: Jan 2021 – Apr 2021.

## Key findings

The strongest independent predictors of upsets (initial run, without minimum lookahead gap) were downhole temperature sensors, particularly `dht_w12_004` and `dht_w12_016`, alongside downhole pressure variability (`dhp_w12_004_24h_std`, `dhp_spread`). The contribution of topside machinery features (mach_spd_g21a, mach_vib_*) is to be evaluated after re-running with the corrected label window.

## The biggest takeaway

This was an interesting exercise — but the clearest conclusion from it is that you need a production engineer to make a dataset like this genuinely useful. The data captures what happened: sensor readings, alarm counts, flow rates. It does not capture why a shutdown was delayed, which sensor everyone on the platform knew to ignore, or what the operations team observed before the event showed up in the historian. The Volve dataset is to an asset what a cave painting is to a hunt — a record of what happened, silent on why.

That gap matters more on a real asset. Volve is a decommissioned training dataset with reasonable documentation. On a live field, the stakes are higher, the data is messier, and the decisions are consequential. GenAI can absorb PI tag catalogues, P&IDs, and alarm logs quickly, but it is not a substitute for the process engineer who has watched a compressor degrade. The tacit knowledge simply does not exist in any dataset.

---

## Data licence

Volve data is released by Equinor under the Equinor Open Data Licence. See `HRS and Terms and conditions for license to data - Volve.pdf` in the Volume for full terms.
