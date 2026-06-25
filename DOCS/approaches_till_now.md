### 1) Notebook pipeline: XGBoost (short) + N-HiTS (long)

**What it was:** Research-style notebooks (`Scripts/energy_forecasting_nhits.ipynb`, `weather_energy_forecasting_nhits.ipynb`, etc.) load meter data from MongoDB, clean and resample, engineer features, then use **XGBoost for the near term** and **N-HiTS (NeuralForecast)** for a **long hourly horizon** (e.g. about a month of hourly steps). There is also plotting and anomaly-style checks.

**Why it changed / limitations (model behaviour, errors, wrong patterns):**
- Wrong time frequency made the model learn incorrect daily patterns (shifted peaks, wrong totals).
- N-HiTS smoothed sharp industrial behavior and missed shutdowns/spikes.
- XGB + N-HiTS caused visible jumps at the handoff boundary.

### 2) Weather-aware N-HiTS variant

**What it was:** Same N-HiTS + NeuralForecast idea, extended so the neural side can see **weather-related columns** (see `weather_energy_forecasting_nhits.ipynb` and related “fixed” variants).

**Why it changed / limitations (model behaviour, errors, wrong patterns):**
- Misaligned or poor weather data created fake correlations in predictions.
- Model sometimes overreacted to weather and ignored actual usage patterns.
- Still couldn’t handle shutdowns or rare events well.
---

### 3) RMS daily ensemble: Prophet + XGBoost

**What it was:** In `RMS/energy_forecasting/`, forecasts are **daily**: **Prophet** gives level and intervals, **XGBoost** adds another signal, results are **blended** with weights, then clipping, rounding, optional cost/threshold flags.

**Why it changed / limitations (model behaviour, errors, wrong patterns):**
- Prophet oversmoothed data, missing sudden stops or spikes.
- Blending models sometimes produced “middle” predictions that were wrong.
- No shutdown logic → energy spread across wrong days
---

### 4) `energy_forecast` v2.x: “Daily first, then hours” (XGBoost-heavy)

**What it was:** The approach described in `DOCS/approach.md`: clean hourly history → **predict each day’s total energy** (with a **shutdown vs active** style setup and threshold tuning) → **split that total across hours** using profiles and a **day-ahead hourly model** → **7-day** daily + hourly outputs → a **48-hour “hybrid”** that mixes short-term behavior with the daily-guided shape → rich **validation** (daily, hourly, holdouts, etc.).

**Why it changed / limitations (model behaviour, errors, wrong patterns):**
- Hourly prediction remained hard due to compounded daily + shape errors.
- Old patterns persisted during regime changes (shift changes, downtime).
- Hybrid logic sometimes lagged or overshot during transitions.
---

### 5) Operational split: train vs forecast + versioned models

**What it was:** Described in `DOCS/blueprint.md` and `DOCS/retrain_inference_pipeline.md` and reflected in code: **training** (slow, bounded data window) **vs inference** (fast, loads **active** artifacts), **versioned artifact folders**, **`active_model.json`**, safer Mongo reads (RTC windows), APIs like **train / forecast / model-status** alongside a full **run** for debugging.

**Why it changed / limitations (model behaviour, errors, wrong patterns):**
- Training and inference mismatch caused drift and poor predictions.
- Using too much old data made models learn outdated behavior.
- Wrong model version sometimes looked like model failure.
---

### 6) Daily-only mode (current simplification)

**What it was:** Hourly training, shape/hybrid paths, and some UI pieces are **turned off or commented**, with **stub profiles** (e.g. flat 1/24) and docs like `DOCS/commented_hourly_pred.md` explaining what was kept vs commented. The **daily** train/infer path and **daily charts** stay in focus.

**Why it changed / limitations (model behaviour, errors, wrong patterns):**
- Flat hourly fallback (1/24) gives unrealistic usage patterns.
- Removes hourly errors but loses intra-day insights.
- Focus shifts to getting correct daily totals only.
---

### Feature Engineering Versions

| Version | Feature idea in one line |
|--------|---------------------------|
| A | Time grid + target + basic short-horizon history. |
| B | Add weather as extra drivers for long horizon. |
| C | Full industrial hourly table: calendar + cycles + lags + rolls + shutdown/restart. |
| D | Daily totals + day-type / recency so “which day” is right before “which hour.” |
| E | Profiles + learned day-ahead hourly shape + short/long blend for believable curves. |
| F | Rolling windows so features teach **recent** behaviour, not **fossil** behaviour. |


## One-line contrast

| Stage | One line |
|--------|-----------|
| N-HiTS notebooks | Neural long horizon + XGB short term, exploratory. |
| Weather N-HiTS | Same idea, tests weather as extra input. |
| RMS Prophet+XGB | Daily blended classical forecast for many meters. |
| `energy_forecast` v2.4 | Daily XGBoost core + hourly detail + 48h hybrid + validation. |
| Train/infer + registry | Same logic, packaged for ops (versions, APIs, bounded data). |
| Daily-only | Same backbone, **hourly path parked**, daily-first product. |