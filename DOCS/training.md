**TRAINING**

The training phase teaches the system two things:

1. “Will this day be active or shutdown?”
2. “If active, how much energy will be consumed?”

That is why the core model is called a **two-stage daily model**:

```text
Raw meter readings
   -> clean hourly data
   -> build daily features
   -> train classifier: active vs shutdown
   -> train regressor: kWh amount for active days
   -> tune threshold
   -> calibrate predictions
   -> retrain final models on full data
```

The main training entry point is [train_phase.py](E:/Projects/IOT-AI-Implementation/energy_forecast/train_phase.py). The API endpoint that triggers it is `/api/train` in [server.py](E:/Projects/IOT-AI-Implementation/energy_forecast/server.py).

**1. API Starts Training**

In [server.py](E:/Projects/IOT-AI-Implementation/energy_forecast/server.py), `/api/train` calls:

```python
tr = run_train_phase(settings)
```

That means the server does not train directly. It delegates training to `run_train_phase`.

If the request says `train_days = "full"`, it trains on all available data. If it is a number, it must be greater than 250 days. This protects the model from being trained on too little history.

Analogy: training with only 30 days of energy data is like judging a factory’s yearly behavior after watching it for one month. You may miss holidays, shutdown patterns, weekends, seasonal behavior, and operating cycles.

**2. Load Raw Meter Data**

Inside [train_phase.py](E:/Projects/IOT-AI-Implementation/energy_forecast/train_phase.py), the code calculates the date range and loads raw MongoDB data:

```python
df_raw = load_raw_dataframe(...)
```

The raw data likely contains minute-level or frequent readings from the energy meter.

Important columns are later renamed in [preprocess.py](E:/Projects/IOT-AI-Implementation/energy_forecast/preprocess.py):

```text
A1 -> total_kwh
RTC -> rtc_timestamp
EnergymeterId -> meter_id
```

`total_kwh` is cumulative meter reading, like an odometer in a car.

Example:

```text
10:00 meter = 5000 kWh
11:00 meter = 5025 kWh
Consumption during that hour = 25 kWh
```

**Impact:** This step gives the model the historical truth it will learn from.

**3. Preprocessing: Clean and Convert to Hourly**

The function `preprocess_hourly` in [preprocess.py](E:/Projects/IOT-AI-Implementation/energy_forecast/preprocess.py) does the cleaning.

It:

1. Renames columns.
2. Sorts by meter and timestamp.
3. Converts meter readings to numbers.
4. Converts timestamps to dates.
5. Drops invalid rows.
6. Treats `total_kwh == 0` as bad/missing data.
7. Resamples data to hourly average readings.
8. Calculates hourly consumption using difference between cumulative readings.

The key line is:

```python
df["consumption_kwh"] = df["total_kwh"].diff().clip(lower=0)
```

If the meter reading increases from 1000 to 1030, consumption is 30 kWh.

If the reading accidentally goes backward, the code clips it to 0 instead of allowing negative consumption.

Analogy: if a car odometer says 10,000 km and then 9,900 km, you know that is probably a data problem. You should not say the car drove -100 km.

**Impact:** This step turns raw meter logs into the actual target value the model needs: hourly energy usage.

**4. Feature Building**

Feature building means giving the model helpful clues.

In [features.py](E:/Projects/IOT-AI-Implementation/energy_forecast/features.py), the active feature builder currently adds:

```text
is_holiday
is_weekend
is_working_day
```

Even though the file contains many commented-out hourly features, the currently active code only adds these calendar flags.

Important detail: in [train_phase.py](E:/Projects/IOT-AI-Implementation/energy_forecast/train_phase.py), the training code uses:

```python
_ae = dubai_holiday_calendar()
df = build_features(df, _ae)
```

So the holiday calendar being used in training is the UAE/Dubai holiday calendar, despite the parameter name `_gj_holidays` inside `build_features`.

Analogy: the model is being told, “This date is a holiday,” “This is a weekend,” or “This is a normal working day.” A factory, office, or plant usually consumes energy differently on those days.

**Impact:** Features help the model find patterns beyond raw numbers. Without features, the model only sees consumption history. With features, it can learn that Fridays, weekends, holidays, or working days behave differently.

**5. Build Daily Aggregation**

The code then converts hourly data into daily data using `build_daily_agg` in [daily.py](E:/Projects/IOT-AI-Implementation/energy_forecast/daily.py).

It groups hourly rows by date and creates:

```text
y = total consumption for that day
daily_zero_hour_ratio = fraction of hours with almost zero consumption
is_holiday
is_weekend
is_working_day
```

Then it adds calendar features:

```text
dow              day of week, Monday=0, Sunday=6
dow_sin/cos      cyclical day-of-week encoding
month
mon_sin/cos      cyclical month encoding
day_of_month
is_sunday
```

The sine/cosine features are important.

Why not just use `dow = 0, 1, 2, ..., 6`?

Because numerically, Sunday `6` looks far away from Monday `0`, but in real life they are next to each other in the weekly cycle.

Analogy: a clock goes from 23:00 back to 00:00. If you use plain numbers, 23 and 0 look far apart. With sine/cosine, the model understands that they are neighbors in a circle.

**Impact:** Daily aggregation changes the problem from “predict every hour directly” to “predict the whole day first.” That is simpler and often more stable for energy forecasting.

**6. Active vs Shutdown Label**

In [constants.py](E:/Projects/IOT-AI-Implementation/energy_forecast/constants.py):

```python
SHUTDOWN_THRESH = 50
```

In [daily.py](E:/Projects/IOT-AI-Implementation/energy_forecast/daily.py), a day is marked active if:

```python
is_active = y >= SHUTDOWN_THRESH
```

So:

```text
daily consumption >= 50 kWh -> active day
daily consumption < 50 kWh  -> shutdown day
```

Analogy: before predicting “how much food a restaurant will sell,” first ask, “Is the restaurant open that day?” If it is closed, sales should be zero.

**Impact:** This avoids predicting normal consumption on shutdown days.

**7. Operational History Features**

The daily aggregation also builds history-based features:

```text
y_lag_1                  yesterday's consumption
y_lag_2                  consumption two days ago
y_lag_7                  same day last week
y_roll_mean_3            average of previous 3 days
y_roll_mean_7            average of previous 7 days
y_roll_mean_14           average of previous 14 days
recent_shutdown_ratio_7  how often recent days were shutdown
recent_shutdown_ratio_14 same idea over 14 days
zero_streak_prev         consecutive shutdown streak before this day
last_24h_mean_cons
last_48h_mean_cons
last_72h_zero_hour_ratio
daily_zero_hour_ratio
is_sunday
```

These are listed in `DAILY_FEATURES_V22` in [constants.py](E:/Projects/IOT-AI-Implementation/energy_forecast/constants.py), and that is the active feature set used during training.

Analogy: if a factory was off yesterday and the day before, there is a higher chance it may still be off today. If last Monday was high usage, this Monday may also be high.

**Impact:** These features teach the model operating rhythm, not just calendar rhythm.

**8. Train/Validation Split**

The model splits historical daily data into:

```text
training data   -> older days
validation data -> most recent N days
```

The validation period is controlled by:

```python
settings.val_days
```

Analogy: the model studies the first part of the textbook, then takes a test on the last chapter it has not seen.

**Impact:** Validation tells you how the model performs on recent unseen data, which is closer to real forecasting.

**9. XGBClassifier: “Will This Day Be Active?”**

The first model is an `XGBClassifier`.

It predicts:

```text
0 -> shutdown day
1 -> active day
```

This is trained in `train_two_stage` in [daily.py](E:/Projects/IOT-AI-Implementation/energy_forecast/daily.py).

It uses:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=spw,
)
```

Simple explanation:

- `n_estimators=300`: build 300 small decision trees.
- `max_depth=4`: each tree is not too deep, so it does not memorize too aggressively.
- `learning_rate=0.05`: learn slowly and carefully.
- `scale_pos_weight=spw`: handle imbalance between active and shutdown days.

XGBoost works like a team of small decision makers. One tree may learn “weekends are lower.” Another may learn “after shutdown streaks, shutdown is more likely.” Another may learn “recent high usage means active is likely.” Together they form a strong prediction.

**Impact:** The classifier protects the forecast from predicting energy usage on days that are probably shutdown.

**10. XGBRegressor: “If Active, How Much kWh?”**

The second model is an `XGBRegressor`.

It predicts the daily kWh amount, but only for active days:

```python
train_active = train_daily[train_daily["is_active"] == 1]
xgb_reg.fit(train_active[feats], train_active["y"])
```

This is important.

Shutdown days are excluded from the regressor because they are a different behavior. If you mix shutdown days and active days in one regression model, the model may learn a weak average that is bad for both.

Analogy: do not train one model to predict both “restaurant closed = 0 sales” and “restaurant open = normal sales.” First decide open/closed, then estimate sales only if open.

**Impact:** The regressor focuses on normal operating consumption and becomes better at estimating active-day load.

**11. Two-Stage Prediction**

Prediction uses both models:

```python
probs = classifier probability of active
reg_pred = regressor predicted kWh
```

Then:

```text
if active probability >= threshold:
    prediction = regressor prediction
else:
    prediction = 0
```

So the model says:

```text
Classifier: "I think this day is active."
Regressor:  "Then I estimate 4,200 kWh."
```

Or:

```text
Classifier: "I think this day is shutdown."
Final:      "Prediction is 0 kWh."
```

**Impact:** This design handles shutdown behavior much better than a single regression model.

**12. Recency Ratio**

After the first prediction, the code applies a `recency_ratio`.

This compares recent active-day consumption with earlier active-day consumption.

Example:

```text
Earlier active average = 4000 kWh/day
Recent active average  = 4400 kWh/day
Recency ratio = 1.10
```

Then predictions are multiplied by `1.10`.

The ratio is clipped between:

```text
0.75 and 1.35
```

Analogy: if a factory recently started running hotter or longer shifts, old history may underpredict. The recency ratio says, “Adjust predictions toward the current operating level.”

**Impact:** Helps adapt to recent changes without letting one strange week distort everything.

**13. Threshold Tuning**

The default classifier threshold is in [constants.py](E:/Projects/IOT-AI-Implementation/energy_forecast/constants.py):

```python
PROB_THRESH = 0.15
```

Normally classifiers use `0.5`, but this code starts lower. That means a day only needs a 15% active probability to be treated as active.

Then `tune_threshold_total_error` searches thresholds from `0.05` to `0.80`.

It chooses the threshold that balances:

```text
total forecast error
false active days
missed active days
```

Analogy: choosing the threshold is like setting a smoke detector sensitivity. Too sensitive and it gives false alarms. Not sensitive enough and it misses real smoke.

Here:

- False active means predicting consumption when the site was shutdown.
- Missed active means predicting zero when the site actually consumed energy.

**Impact:** Threshold tuning can strongly affect total monthly/validation kWh error. A slightly different threshold may change several days from zero to active or active to zero.

**14. LinearRegression Calibration**

After validation predictions are made, the code tries calibration.

Calibration means: “The model is mostly right in shape, but maybe consistently too high or too low. Let’s correct that.”

The affine calibration uses `LinearRegression`:

```text
actual_y ≈ a * predicted_y + b
```

Example:

```text
Model predicts: 4000
Actual tends to be: 4400

Calibration may learn:
actual ≈ 1.08 * prediction + 100
```

So future predictions are nudged upward.

But the code has guardrails:

```text
a is clipped between 0.88 and 1.12
b is clipped to a limited fraction of average y
calibration is disabled if it does not improve MAPE
```

Analogy: it is like adjusting a weighing scale that is always slightly under-reading. But you only allow small adjustments, not wild corrections.

**Impact:** Linear calibration fixes simple bias: consistently too high or too low predictions.

**15. IsotonicRegression Calibration**

The code also tries `IsotonicRegression`.

This is more flexible than a straight line, but still constrained.

It learns a monotonic mapping:

```text
if prediction goes up, calibrated output cannot go down
```

Example:

```text
Raw prediction 3000 -> calibrated 3100
Raw prediction 4000 -> calibrated 4300
Raw prediction 5000 -> calibrated 5700
```

This can fix curved errors.

For example, maybe the model is accurate on low-consumption days but underpredicts high-consumption days. Linear regression may not fix that well, but isotonic regression can.

Analogy: LinearRegression is like using one straight ruler correction. IsotonicRegression is like using a custom lookup chart, but one that always moves upward logically.

The code chooses isotonic only if it beats affine by enough. Otherwise it keeps affine because affine is simpler.

**Impact:** Isotonic calibration can improve high-load or low-load correction, but it is used carefully to avoid overfitting.

**16. Validation Metrics**

The validation step computes:

```text
classifier_accuracy
tp / fp / fn
daily_mae
daily_rmse
daily_mape_pct
validation actual total kWh
validation predicted total kWh
validation total error %
```

Important meanings:

- `MAE`: average absolute daily error.
- `RMSE`: like MAE but punishes large misses more.
- `MAPE`: percentage error on active days.
- `val_total_error_pct`: whether the total validation period is over or under forecast.

Analogy: daily error asks, “Was each day close?” Total error asks, “Was the final electricity bill close?”

**Impact:** The model may have some daily misses but still get total monthly consumption close, or vice versa. Both views matter.

**17. Final Full-Data Training**

After validation is complete, the code trains final models again on all daily data:

```python
xgb_clf_full.fit(daily_agg[DAILY_FEATURES_V22], daily_agg["is_active"])
xgb_reg_full.fit(all_active[DAILY_FEATURES_V22], all_active["y"])
```

This is normal.

Validation is used to tune and measure. Once that is done, the final production model gets all available history.

Analogy: first you hold back some exam questions to check if the student learned properly. Once you are satisfied, the student is allowed to study the whole book before the real exam.

**Impact:** The deployed model has the most data possible.

**18. Profiles**

[profiles.py](E:/Projects/IOT-AI-Implementation/energy_forecast/profiles.py) currently returns flat hourly profiles:

```text
each hour = 1/24 of the day
```

So if daily forecast is 2400 kWh, each hour gets:

```text
2400 / 24 = 100 kWh
```

The older data-driven hourly-shape logic is present in comments, but not active.

**Impact:** The current training phase is mainly daily forecasting. It does not currently learn a sophisticated hour-by-hour shape from historical usage.

**19. Commented-Out Hourly Models**

In [train_phase.py](E:/Projects/IOT-AI-Implementation/energy_forecast/train_phase.py), hourly model training sections are commented out:

```text
day-ahead hourly model
shape model
hourly full model
hourly metrics
hourly feature importance
```

And the returned values are:

```python
xgb_hourly_full=None
xgb_da_val=None
shape_full=None
hourly_short_term_metrics={}
```

**Impact:** Despite constants containing many hourly features, the active training path is daily-only with flat hourly distribution support.

**Final Mental Model**

Think of the system like a factory manager making a weekly energy plan:

1. Clean the meter readings.
2. Convert odometer-style readings into hourly usage.
3. Summarize each day.
4. Add clues: weekend, holiday, Sunday, recent shutdowns, yesterday’s usage.
5. Train one model to answer: “Will the plant run?”
6. Train another model to answer: “If it runs, how much energy?”
7. Tune the active/shutdown decision threshold.
8. Calibrate predictions if validation shows consistent bias.
9. Retrain final models on all data.
10. Save the trained result for forecasting.