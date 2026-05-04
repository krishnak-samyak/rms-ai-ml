"""Thresholds and feature lists (aligned with energy_forecast_v2.ipynb)."""

SHUTDOWN_THRESH = 50
ACTIVE_HOUR_THRESH = 5.0
PROB_THRESH = 0.15
RECENT_WINDOW = 30
FORECAST_DAYS = 7
MIN_DAYS_FOR_DOW = 4
SHORT_TERM_SHUTDOWN_THRESH = 0.15

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

DAILY_FEATURES = [
    "is_holiday",
    "is_weekend",
    "is_working_day",
    "dow",
    "dow_sin",
    "dow_cos",
    "month",
    "mon_sin",
    "mon_cos",
    "day_of_month",
]

# Version 2.1: adds operational-state features to improve shutdown recall.
DAILY_FEATURES_V21 = DAILY_FEATURES + [
    "y_lag_1",
    "y_lag_2",
    "y_lag_7",
    "y_roll_mean_3",
    "y_roll_mean_7",
    "recent_shutdown_ratio_7",
    "zero_streak_prev",
]

# Version 2.2: stronger shutdown/state features and Sunday separation.
DAILY_FEATURES_V22 = DAILY_FEATURES_V21 + [
    "is_sunday",
    "daily_zero_hour_ratio",
    "y_roll_mean_14",
    "recent_shutdown_ratio_14",
    "last_24h_mean_cons",
    "last_48h_mean_cons",
    "last_72h_zero_hour_ratio",
]

HOURLY_FEATURES = [
    "is_holiday",
    "is_weekend",
    "is_working_day",
    "is_working_hour",
    "days_to_next_holiday",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "mon_sin",
    "mon_cos",
    "day_of_month",
    "shift",
    "shift_sin",
    "shift_cos",
    "hours_since_shutdown",
    "just_restarted",
    "cons_lag_1",
    "cons_lag_2",
    "cons_lag_3",
    "cons_lag_6",
    "cons_lag_12",
    "cons_lag_24",
    "cons_lag_48",
    "cons_lag_168",
    "kwh_lag_1",
    "kwh_lag_24",
    "cons_roll_mean_6",
    "cons_roll_mean_12",
    "cons_roll_mean_24",
    "kwh_roll_std_6",
    "kwh_roll_std_12",
    "kwh_roll_std_24",
    "rolling_max_24",
    "rolling_min_24",
    "rolling_range_24",
    "rolling_max_168",
    "rolling_min_168",
    "rolling_range_168",
]

# Day-ahead safe subset: excludes explicit intra-day lags (< 24h).
_INTRADAY_LAGS = {
    "cons_lag_1", "cons_lag_2", "cons_lag_3", "cons_lag_6", "cons_lag_12",
    "kwh_lag_1",
    "cons_roll_mean_6", "cons_roll_mean_12",
    "kwh_roll_std_6", "kwh_roll_std_12",
}
DAYAHEAD_HOURLY_FEATURES = [f for f in HOURLY_FEATURES if f not in _INTRADAY_LAGS]
