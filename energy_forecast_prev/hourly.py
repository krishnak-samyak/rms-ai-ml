"""Short-term hourly XGBoost and recursive 6h forecast."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from energy_forecast.constants import ACTIVE_HOUR_THRESH, HOURLY_FEATURES
from energy_forecast.features import build_features


def train_hourly_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    sample_weight: np.ndarray | None = None,
) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
    )
    eval_set = [(X_val, y_val)] if X_val is not None and len(X_val) else None
    fit_kw: dict = {"verbose": False}
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight
    if eval_set:
        model.fit(X_train, y_train, eval_set=eval_set, **fit_kw)
    else:
        model.fit(X_train, y_train, **fit_kw)
    return model


def hourly_test_metrics(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test).clip(min=0)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    active_mask = y_test > ACTIVE_HOUR_THRESH
    mape = None
    if active_mask.sum() > 0:
        mape = float(
            np.mean(np.abs((y_test[active_mask] - y_pred[active_mask]) / y_test[active_mask])) * 100
        )
    return {"mae": mae, "rmse": rmse, "mape_pct": mape, "n_active_hours_mape": int(active_mask.sum())}


def forecast_recursive_6h(
    model: XGBRegressor,
    df_hist: pd.DataFrame,
    features: list[str],
    _gj_holidays,
    n_steps: int = 6,
) -> pd.DataFrame:
    buf = df_hist.copy()
    preds = []

    for _ in range(n_steps):
        last_ts = buf["rtc_timestamp"].iloc[-1]
        next_ts = last_ts + timedelta(hours=1)

        row = pd.DataFrame(
            {
                "rtc_timestamp": [next_ts],
                "total_kwh": [buf["total_kwh"].iloc[-1]],
                "consumption_kwh": [0.0],
            }
        )
        row = pd.concat([buf, row], ignore_index=True)
        row = build_features(row, _gj_holidays)
        row = row.iloc[[-1]]

        pred_cons = float(model.predict(row[features].fillna(0))[0])
        pred_cons = max(pred_cons, 0.0)

        yesterday_mask = buf["rtc_timestamp"] == next_ts - timedelta(hours=24)
        if yesterday_mask.any():
            yest_cons = buf.loc[yesterday_mask, "consumption_kwh"].iloc[0]
            alpha = 0.6
            pred_cons = alpha * pred_cons + (1 - alpha) * yest_cons

        new_row = pd.DataFrame(
            {
                "rtc_timestamp": [next_ts],
                "total_kwh": [buf["total_kwh"].iloc[-1] + pred_cons],
                "consumption_kwh": [pred_cons],
            }
        )
        buf = pd.concat([buf, new_row], ignore_index=True)
        preds.append({"rtc_timestamp": next_ts, "forecast_kwh": pred_cons})

    return pd.DataFrame(preds)
