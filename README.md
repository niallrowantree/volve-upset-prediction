# Volve Production Upset Prediction

Early warning model for production upsets on the Equinor Volve oil field, using PI historian sensor data from Databricks Unity Catalog.

## Overview

Uses 18 months of PI System time-series data (Nov 2019 – Apr 2021) from the Volve field to train an XGBoost classifier that predicts production upsets up to 12 hours ahead. The model is deliberately restricted to **independent wellbore signals** (downhole pressure and temperature sensors) to provide genuine early warning rather than detection of already-in-progress events.

**Data source**: `equinor_asa_volve_data_village` catalog — Equinor's open Volve dataset, available on Databricks via the Data Marketplace.

## Project structure

| Notebook | Purpose |
|---|---|
| `01_ingest.py` | Reads ~25 PI sensor tags from Volume parquet files, resamples to 5-min grid, pivots wide, writes `pi_sensors_5min` Delta table |
| `02_labels.py` | Defines upset events as >20% flow drop vs 7-day rolling baseline; creates `upset_4h`, `upset_12h`, `upset_24h` forward-looking binary labels |
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

**GDR alarm counts** — included in Delta table, used for validation only (Mar–Apr 2021 coverage):
- `B00.Alarm.ActiveCount`, `B00.Alarm.CriticalActiveCount`

## Label definition

A timestep is labelled an upset if the HC flow rate drops more than 20% below its 7-day rolling baseline. Forward-looking labels (`upset_Xh = 1`) flag any timestep where an upset occurs within the next X hours, enabling the model to predict ahead.

Key upset events in the dataset:
- Mar 2020: 3-day full shutdown
- Jun 2020: 2-week extended partial upset
- Sep 2020: 6-day full shutdown
- Feb–Mar 2021: 12-day extended shutdown

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

**Why exclude surface flow features?** All surface flow meters (FE-315, FE-914, FE-944) drop simultaneously during a production upset. Including them makes the model learn "upsets continue once started" rather than predicting them ahead of time. Only downhole sensors, quality metrics, and calendar features are used as model inputs.

**Why XGBoost?** Interpretability via SHAP is critical for production engineers to act on alerts. SHAP feature attribution shows exactly which downhole sensor is driving each prediction.

**Why time-based train/test split?** Random splits would leak future information across the temporal autocorrelation in the time series. Train: Nov 2019 – Dec 2020. Test: Jan 2021 – Apr 2021.

## Key findings

The strongest independent predictors of upsets 12h ahead are downhole temperature sensors, particularly `dht_w12_004` (multiple window lengths) and `dht_w12_016`, alongside downhole pressure variability (`dhp_w12_004_24h_std`, `dhp_spread`). This suggests a thermal/pressure signature in the wellbore precedes surface production upsets.

## Data licence

Volve data is released by Equinor under the Equinor Open Data Licence. See `HRS and Terms and conditions for license to data - Volve.pdf` in the Volume for full terms.
