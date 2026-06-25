# Energy Forecast Model 

This document explains how the energy forecasting model works, what data it learns from, and what each piece of metadata shown in the dashboard means. No machine learning background required.

---

## What Does This Model Do?

Given historical electricity consumption data from a smart meter, the model answers two questions for each of the next **5 days**:

1. **Will the site be operational (consuming meaningful energy) on that day?**
2. **If yes, how many kWh will it consume?**

The output is a daily forecast table showing predicted consumption for the upcoming days.

---

## How Is the Model Trained?

Training happens in a two-stage pipeline. Think of it as two specialists working together.

### Stage 1 - The "Is Anyone Home?" Classifier

This is an **XGBoost Classifier** - a decision-tree ensemble that votes on yes/no questions.

- It looks at historical patterns and learns to predict whether a future day will be an **active day** (≥ 50 kWh consumed) or a **shutdown day** (near-zero consumption - site closed, holiday, maintenance, etc.)
- It outputs a **probability score** (0 to 1). If the score crosses the **tuned threshold**, the day is called active.
- Trained with 300 boosting rounds, shallow trees (depth 4), and a slow learning rate so it generalises well.

### Stage 2 - The "How Much Energy?" Regressor

This is an **XGBoost Regressor** - the same tree-ensemble family, but trained to predict a continuous number (kWh).

- Trained **only on active days** so it never confuses "site was shut" with "site used a little energy."
- Outputs a raw kWh estimate which is then **scaled** and **calibrated** before being shown.
- Trained with 500 boosting rounds and additional regularisation to avoid overfitting to noisy days.

### The Training Sequence

```
Raw meter readings (MongoDB)
        ↓
  Resample → Hourly consumption
        ↓
  Build daily features (lags, rolling averages, calendar flags)
        ↓
  Split last 41 days → Validation set  |  Everything else → Training set
        ↓
  Train Classifier on training set
        ↓
  Tune probability threshold on validation set (minimize billing error + false-day-type rate)
        ↓
  Fit post-calibration corrector on validation active days
        ↓
  Re-train Classifier + Regressor on ALL data (training + validation)
        ↓
  Save model artifacts to disk
```

---

## What Features Does the Model Learn From?

Features are derived signals computed from raw meter data + the calendar. Think of them as the "inputs" the model reads before making a prediction.

### Calendar Features

| Feature | What It Captures |
|---|---|
| Day of week (0 = Mon … 6 = Sun) | Weekly consumption rhythm |
| Day-of-week encoded as sin/cos | Smooth cyclical pattern so Mon and Sun aren't treated as opposites |
| Month (1–12) + sin/cos encoding | Seasonal shifts in demand |
| Day of month | Month-start / month-end patterns |
| Is holiday? | Public holiday calendar flag |
| Is weekend? | Saturday or Sunday |
| Is working day? | Neither weekend nor holiday |
| Is Sunday? | Sundays often behave differently from Saturdays |

### Lag & History Features

These tell the model what the site was doing recently - short-term memory.

| Feature | What It Captures |
|---|---|
| Yesterday's consumption (lag 1) | Inertia - busy yesterday → likely busy today |
| 2 days ago consumption (lag 2) | Medium-term pattern |
| 7 days ago consumption (lag 7) | Same day last week - strong weekly signal |
| 3-day rolling average | Short-trend smoothing |
| 7-day rolling average | Weekly-trend smoothing |
| 14-day rolling average | Fortnight-trend smoothing |

### Shutdown / Zero-Hour Features

These tell the model how "active" the site has been recently.

| Feature | What It Captures |
|---|---|
| % of last 7 days that were shutdown | How often the site went dark recently |
| % of last 14 days that were shutdown | Longer shutdown pattern |
| % of hours yesterday with zero consumption | Was yesterday partially idle? |
| Average hourly consumption last 24 h | Short-term intensity |
| Average hourly consumption last 48 h | Slightly longer view |
| 3-day rolling zero-hour ratio | Rolling idle fraction |
| Consecutive shutdown days before today | How long the site has been dark in a row |

---

## Recency Scaling - Adapting to Drift

A factory's baseline consumption can change over time (new equipment, shift pattern change, production increase). The model accounts for this with a **recency ratio**:

- It computes the average kWh on active days over the **last 30 active days**.
- It compares that to the average over **older active days**.
- The ratio (clipped between 0.75 and 1.35) scales the regressor's raw output up or down.
- Example: ratio = 1.15 means the site is currently consuming 15% more than its historical baseline - every forecast is multiplied by 1.15.

---

## Post-Calibration - Correcting Systematic Bias

Even a well-trained model can be consistently high or low. Post-calibration corrects this.

| Mode | What It Does |
|---|---|
| **none** | No correction applied - raw model output used |
| **affine** | Applies a linear correction: `final = a × prediction + b`. `a` is clipped to [0.88, 1.12], `b` is clipped to ±30% of mean consumption. Corrects both scale and offset bias. |
| **isotonic** | Fits a monotone staircase function on validation predictions vs actuals. More flexible than affine but chosen only if it improves accuracy by > 0.1%. |

---

## Metadata Keys - What Each Field Means

These are the fields shown on the model metadata screen in the dashboard.

| Field | Plain-English Meaning |
|---|---|
| **Pipeline Version** | The version number of the training code used (e.g., `2.4`). Higher number = newer algorithm with more features or better calibration logic. |
| **Meter Id** | The unique identifier of the smart meter this model was trained for. Each meter gets its own personalised model. |
| **Artifact Version** | The timestamp when this specific model was trained and saved (e.g., `2026-06-23T11-58-51Z`). Acts like a version number for the trained model file. |
| **Artifact Relative Dir** | The folder path on disk where this model's files are stored, relative to the model root directory. Format: `{MeterId}/{ArtifactVersion}/`. |
| **Trained At Utc** | The exact date and time (in UTC) when training completed and the model was written to disk. |
| **Train Data Start** | The earliest meter reading included in the training data. All data from this point onwards was used to teach the model. |
| **Train Data End** | The latest meter reading in the training data. The model has no knowledge of events after this date. |
| **Val Days** | The number of most-recent days held back from training to evaluate the model. These days were **not** used to teach the model - they were used to check how well it performs on unseen data. Typically 41 days. |
| **Forecast Days** | How many days into the future the model is asked to predict. Typically 5 days ahead. |
| **Train Raw Lookback Days** | How many days of raw meter data were pulled from the database to train the model. `-1` means "use all available data." |
| **Infer Raw Lookback Days** | How many days of recent meter data are pulled at inference (prediction) time to refresh the lag/rolling features before forecasting. Must be large enough to cover at least 14-day lags (default: 45 days). |
| **Tuned Threshold** | The probability cut-off used by the classifier to decide active vs shutdown. Example: `0.75` means the classifier must be at least 75% confident a day is active before marking it so. Tuned automatically on validation data to minimise billing error. |
| **Recency Ratio Val** | The recency scale factor calculated on the **validation set** only. Shows how much the recent period's consumption differs from the historical average - used during model evaluation. |
| **Recency Ratio Full** | The recency scale factor calculated on the **full dataset** (training + validation). This is the value actually applied when generating live forecasts. |
| **Spw** (Scale Positive Weight) | The ratio of active days to shutdown days in the training data. Example: `13.76` means there were ~14 active days for every 1 shutdown day. The classifier uses this to avoid being biased toward always predicting "active." |
| **Daily Calib Mode** | The post-calibration method applied to regressor outputs. One of `none`, `affine`, or `isotonic` - see the calibration section above. |

---

## Where Are the Model Files Stored?

Each trained model is saved in a timestamped folder:

```
{Model Root}/
└── {Meter Id}/
    └── {Artifact Version}/
        ├── xgb_clf.pkl          ← Classifier model
        ├── xgb_reg.pkl          ← Regressor model
        ├── dow_profiles.csv     ← 24-hour shape by day-of-week
        ├── daytype_profiles.csv ← 24-hour shape by day type (working / weekend / holiday)
        ├── future_daily.csv     ← Latest 5-day forecast output
        └── model_metadata.json  ← Everything in the dashboard + validation stats
```

An `active_model.json` file in the meter folder always points to the latest deployed version.

---

## Quick Glossary

| Term | Meaning |
|---|---|
| **XGBoost** | A machine learning algorithm that builds many small decision trees and combines their votes. Highly accurate on tabular data. |
| **Classifier** | A model that outputs a category (active / shutdown). |
| **Regressor** | A model that outputs a number (kWh consumed). |
| **Validation set** | A portion of data hidden from training, used only to test accuracy. |
| **Lag feature** | The value of a variable from N days ago - gives the model short-term memory. |
| **Rolling average** | The average value over the last N days - smooths out day-to-day noise. |
| **Threshold** | The probability above which the classifier says "yes, this day is active." |
| **Calibration** | A correction step that adjusts predictions to remove systematic over- or under-estimation. |
| **Recency ratio** | A multiplier that adjusts forecasts up or down based on how current consumption compares to the historical average. |
| **Artifact** | A saved, versioned snapshot of a trained model and all associated files. |
