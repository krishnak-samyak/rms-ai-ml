# Project Index: energy_forecast

Generated: 2026-06-23 | Pipeline version: 2.4

---

## Project Structure

```
IOT-AI-Implementation/
├── energy_forecast/          ← Python package (this project)
│   ├── __init__.py           ← exports RunResult, run_pipeline
│   ├── server.py             ← FastAPI app (entry point)
│   ├── pipeline.py           ← end-to-end orchestrator
│   ├── train_phase.py        ← model training
│   ├── infer_phase.py        ← 7-day forecast generation
│   ├── model_registry.py     ← artifact persistence + loading
│   ├── config.py             ← env-based Settings dataclass
│   ├── constants.py          ← thresholds, feature lists
│   ├── data.py               ← MongoDB loader
│   ├── preprocess.py         ← raw → hourly consumption
│   ├── features.py           ← calendar/cyclical/lag features
│   ├── daily.py              ← daily agg + XGBoost two-stage model
│   └── profiles.py           ← DOW/daytype hourly fraction profiles
├── Scripts/                  ← research Jupyter notebooks (origin: energy_forecast_v2.ipynb)
├── requirements.txt          ← package dependencies
├── models/                   ← trained artifact store (gitignored)
└── DOCS/                     ← project documentation
```

---

## Entry Points

| Entry | Path | Purpose |
|-------|------|---------|
| HTTP server | [server.py](server.py) | FastAPI app — dashboard + REST API |
| Full pipeline | [pipeline.py](pipeline.py) | `run_pipeline()` — train + infer end-to-end |
| Package root | [__init__.py](__init__.py) | `from energy_forecast import run_pipeline, RunResult` |

---

## API Endpoints (server.py)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Dashboard (serves `static/index.html`) |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/run` | Full pipeline: train + 7-day forecast (~1 min) |
| `POST` | `/api/train` | Train only; body: `{"train_days": 365\|"full"\|null}` |
| `POST` | `/api/forecast` | Inference only (requires saved models) |
| `GET` | `/api/model-status` | Artifact presence + metadata |
| `GET` | `/api/config/{train_days}` | Current config dump |

---

## Core Modules

### [config.py](config.py)
`Settings` frozen dataclass loaded from env vars.

| Env Var | Default | Purpose |
|---------|---------|---------|
| `MONGO_URI` | internal URI | MongoDB connection |
| `MONGO_DB` | `IOTDeviceMonitor` | database name |
| `MONGO_COLLECTION` | `FUTU00_DataMonitor` | collection |
| `METER_ID` | `FUTU0000000004000002` | target meter |
| `ENERGY_MODEL_DIR` | `./models/energy_forecast` | artifact store root |
| `TRAIN_RAW_LOOKBACK_DAYS` | 365 | training data window |
| `INFER_RAW_LOOKBACK_DAYS` | 45 | inference data window (≥21 needed for 168h lags) |

### [constants.py](constants.py)
Thresholds and versioned feature sets.

| Constant | Value | Purpose |
|----------|-------|---------|
| `SHUTDOWN_THRESH` | 50 kWh/day | below → day is "shutdown" |
| `PROB_THRESH` | 0.15 | classifier probability cutoff |
| `DAILY_FEATURES_V22` | 26 features | current active feature set (v2.2) |
| `HOURLY_FEATURES` | 37 features | full hourly feature set (disabled) |
| `DAYAHEAD_HOURLY_FEATURES` | subset | safe for day-ahead (no intra-day lags) |

### [data.py](data.py)
- `load_raw_dataframe(settings, rtc_gte, rtc_lte)` — queries MongoDB, returns raw DataFrame
- `train_raw_rtc_bounds(settings)` / `infer_raw_rtc_bounds(settings)` — UTC window helpers
- Full-data mode: pass `train_raw_lookback_days=-1`

### [preprocess.py](preprocess.py)
- `preprocess_hourly(df_raw)` — renames cols (`A1→total_kwh`, `RTC→rtc_timestamp`), resamples to 1h, computes `consumption_kwh` = diff of cumulative meter reading

### [features.py](features.py)
- `build_features(frame, _holidays)` — adds calendar flags (`is_holiday`, `is_weekend`, `is_working_day`) and cyclical encodings
- `dubai_holiday_calendar()` — UAE holidays (currently active)
- `gj_holiday_calendar()` — Gujarat/India holidays (available, not used)

### [daily.py](daily.py)
Core ML logic for daily-level forecasting.
- `build_daily_agg(df)` — hourly → daily with lag/rolling features (v2.1/2.2)
- `train_two_stage(train_daily, daily_features)` — fits `XGBClassifier` + `XGBRegressor`
- `predict_daily_two_stage(...)` — classifier gates regressor
- `tune_threshold_total_error(val_daily, ...)` — optimizes F1/total-error threshold
- `select_daily_calibration(val_daily)` — picks affine vs isotonic post-calibration
- `apply_daily_postcalibration(preds, mode, ...)` — applies calibration
- `validation_metrics(val_daily, df)` — returns dict with MAPE, MAE, total kWh error %
- `build_future_daily_rows(last_date, n_days, holidays)` — feature rows for future dates

### [profiles.py](profiles.py)
- `build_profiles(df, holidays)` — returns `(dow_profiles, daytype_profiles, prof_meta)`
- Currently returns **flat stub profiles** (1/24 per hour); data-driven implementation is commented out pending hourly model re-enablement

### [train_phase.py](train_phase.py)
- `run_train_phase(settings) → TrainPhaseResult` — full training orchestration:
  1. Load raw Mongo data
  2. Preprocess + feature engineering
  3. Build profiles + daily aggregation
  4. Train/val split
  5. Train two-stage model, tune threshold
  6. Select and apply daily post-calibration
  7. Retrain on full dataset (`xgb_clf_full`, `xgb_reg_full`)
- `TrainPhaseResult` dataclass — carries all trained models, metrics, and calibration params

### [infer_phase.py](infer_phase.py)
- `run_infer_phase(settings, tr: TrainPhaseResult) → InferPhaseResult`
- Generates 7-day forecast autoregressively: each future day appends to history
- Applies post-calibration if `daily_calib_enabled`
- Hourly decomposition and hybrid 48h forecast are commented out (disabled in v2.4)

### [model_registry.py](model_registry.py)
Versioned artifact layout under `{model_dir}/{meter_id}/{version}/`.

- `persist_training_artifacts(tr, settings, inf)` — saves pickles, CSVs, `model_metadata.json`, updates `active_model.json` pointer
- `load_train_phase_for_inference(settings) → TrainPhaseResult` — reloads models + fresh Mongo data
- `resolve_artifact_dir(settings)` — resolves active version (with legacy fallback)
- `artifact_paths_ok(settings)` — presence check for `xgb_clf.pkl`, `xgb_reg.pkl`, profiles, metadata
- `build_forecast_api_payload(tr, meta, future_daily_df)` — shapes JSON for dashboard

---

## Artifact Layout

```
{ENERGY_MODEL_DIR}/
└── {meter_id}/
    ├── active_model.json          ← {"artifact_version": "2025-06-23T10-15-00Z"}
    └── {artifact_version}/
        ├── model_metadata.json    ← full training metadata + val metrics
        ├── xgb_clf.pkl            ← active/shutdown classifier
        ├── xgb_reg.pkl            ← energy regressor (active days only)
        ├── dow_profiles.csv       ← DOW hourly fraction profiles
        ├── daytype_profiles.csv   ← working/weekend/holiday profiles
        └── future_daily.csv       ← persisted 7-day forecast
```

---

## ML Architecture

```
Raw Mongo rows
    → preprocess_hourly()          resample 1h, diff cumulative kWh
    → build_features()             calendar + cyclical + lag features
    → build_daily_agg()            hourly → daily totals + v2.2 lag features
    ↓
 TWO-STAGE DAILY MODEL
    XGBClassifier  →  P(active|features)  →  threshold tune
    XGBRegressor   →  kWh (on active days only)
    ↓
 POST-CALIBRATION (affine or isotonic, val-selected)
    ↓
 7-DAY AUTOREGRESSIVE FORECAST
    (each day's prediction feeds next day's lag features)
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `xgboost>=2.0` | classifier + regressor |
| `pandas>=2.0` | data manipulation |
| `scikit-learn>=1.3` | IsotonicRegression, LinearRegression, metrics |
| `pymongo>=4.6` | MongoDB data loading |
| `holidays>=0.35` | UAE + India holiday calendars |
| `joblib>=1.3` | model serialization |
| `fastapi>=0.110` | REST API |
| `uvicorn[standard]>=0.27` | ASGI server |

---

## Disabled / Commented-Out Features

These exist in the codebase but are commented out in v2.4:

| Feature | Files | Notes |
|---------|-------|-------|
| Hourly XGBoost model | `train_phase.py`, `infer_phase.py` | `xgb_hourly_full` — training intact, disabled |
| Day-ahead hourly model | `train_phase.py` | `xgb_da_val` — disabled |
| Hourly shape model | `train_phase.py`, `infer_phase.py` | `shape_full` — disabled |
| Hybrid 48h forecast | `infer_phase.py` | `fc_48h` — disabled |
| Hourly decomposition | `infer_phase.py` | `future_hourly_df` — disabled |
| GJ holiday calendar | `features.py`, `train_phase.py` | UAE calendar used instead |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set env vars (optional — defaults point to internal MongoDB)
export MONGO_URI="mongodb://..."
export METER_ID="FUTU0000000004000002"
export ENERGY_MODEL_DIR="./models/energy_forecast"

# Start server
uvicorn energy_forecast.server:app --host 0.0.0.0 --port 8000

# Train models
curl -X POST http://localhost:8000/api/train

# Get 7-day forecast
curl -X POST http://localhost:8000/api/forecast

# Full pipeline (train + forecast in one call)
curl -X POST http://localhost:8000/api/run
```

---

## Research Notebooks (Scripts/)

| Notebook | Purpose |
|----------|---------|
| `energy_forecast_v2.ipynb` | Origin of the production package logic |
| `energy_forecast_v3.ipynb` | Next-generation experiments |
| `daily_validation_diagnostics.ipynb` | Validation analysis |
| `energy_forecasting_calendar.ipynb` | Holiday calendar experiments |
| `weather_energy_forecasting_*.ipynb` | Weather feature experiments |
| `energy_forecasting_nhits.ipynb` | N-HiTS model experiments |
