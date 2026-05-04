"""48h hybrid: XGBoost hours 1-6 + two-stage daily + DOW decomposition for 7-48h.

Version 2.1 note:
- Daily models may include operational-state features (y_lag_*, rolling means, shutdown streak).
- This module now builds those features recursively for the spanned days so inference
  always matches the trained feature schema.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from energy_forecast.constants import HOURLY_FEATURES, SHORT_TERM_SHUTDOWN_THRESH
from energy_forecast.decompose import decompose_daily_to_hourly
from energy_forecast.hourly import forecast_recursive_6h
from energy_forecast.hourly_shape import predict_future_hourly_recursive


def build_span_daily_for_48h(
    last_ts: pd.Timestamp,
    _gj_holidays,
) -> pd.DataFrame:
    h7_ts = last_ts + timedelta(hours=7)
    h48_ts = last_ts + timedelta(hours=48)
    day_start = h7_ts.normalize()
    day_end = h48_ts.normalize()
    span_dates = pd.date_range(day_start, day_end, freq="D")
    span_daily = pd.DataFrame({"ds": span_dates})
    span_daily["is_holiday"] = span_daily["ds"].dt.date.apply(lambda d: 1 if d in _gj_holidays else 0)
    span_daily["is_weekend"] = (span_daily["ds"].dt.dayofweek >= 5).astype(int)
    span_daily["is_working_day"] = (
        (span_daily["is_weekend"] == 0) & (span_daily["is_holiday"] == 0)
    ).astype(int)
    span_daily["dow"] = span_daily["ds"].dt.dayofweek
    span_daily["dow_sin"] = np.sin(2 * np.pi * span_daily["dow"] / 7)
    span_daily["dow_cos"] = np.cos(2 * np.pi * span_daily["dow"] / 7)
    span_daily["month"] = span_daily["ds"].dt.month
    span_daily["mon_sin"] = np.sin(2 * np.pi * span_daily["month"] / 12)
    span_daily["mon_cos"] = np.cos(2 * np.pi * span_daily["month"] / 12)
    span_daily["day_of_month"] = span_daily["ds"].dt.day
    span_daily["day_type"] = span_daily.apply(
        lambda r: "holiday" if r["is_holiday"] else ("weekend" if r["is_weekend"] else "working"),
        axis=1,
    )
    return span_daily


def _build_state_row(base: dict, y_hist: list[float], active_hist: list[int], zero_ratio_hist: list[float]) -> dict:
    """Create operational-state features compatible with v2.1/v2.2 daily models."""
    y_med = float(np.median(y_hist)) if y_hist else 0.0
    sh_hist = [1 - a for a in active_hist]
    row = dict(base)
    row["y_lag_1"] = y_hist[-1] if len(y_hist) >= 1 else y_med
    row["y_lag_2"] = y_hist[-2] if len(y_hist) >= 2 else y_med
    row["y_lag_7"] = y_hist[-7] if len(y_hist) >= 7 else y_med
    row["y_roll_mean_3"] = float(np.mean(y_hist[-3:])) if len(y_hist) >= 3 else y_med
    row["y_roll_mean_7"] = float(np.mean(y_hist[-7:])) if len(y_hist) >= 7 else y_med
    row["y_roll_mean_14"] = float(np.mean(y_hist[-14:])) if len(y_hist) >= 14 else y_med
    row["recent_shutdown_ratio_7"] = float(np.mean(sh_hist[-7:])) if len(sh_hist) >= 7 else 0.0
    row["recent_shutdown_ratio_14"] = float(np.mean(sh_hist[-14:])) if len(sh_hist) >= 14 else 0.0
    row["daily_zero_hour_ratio"] = zero_ratio_hist[-1] if len(zero_ratio_hist) else 0.0
    row["last_24h_mean_cons"] = row["y_lag_1"] / 24.0
    row["last_48h_mean_cons"] = float(np.mean([row["y_lag_1"], row["y_lag_2"]])) / 24.0
    row["last_72h_zero_hour_ratio"] = float(np.mean(zero_ratio_hist[-3:])) if len(zero_ratio_hist) >= 3 else 0.0
    row["is_sunday"] = 1 if pd.Timestamp(row["ds"]).dayofweek == 6 else 0
    streak = 0
    for a in reversed(active_hist):
        if a == 0:
            streak += 1
        else:
            break
    row["zero_streak_prev"] = streak
    return row


def hybrid_forecast_48h(
    df_hourly: pd.DataFrame,
    xgb_hourly_full,
    xgb_clf_full: XGBClassifier,
    xgb_reg_full: XGBRegressor,
    dow_profiles: pd.DataFrame,
    daytype_profiles: pd.DataFrame,
    recency_full: float,
    _gj_holidays,
    hist_tail_n: int = 200,
    shape_model: XGBRegressor | None = None,
) -> pd.DataFrame:
    last_ts = df_hourly["rtc_timestamp"].max()
    hist_tail = df_hourly.tail(hist_tail_n).copy()
    fc_xgb = forecast_recursive_6h(
        xgb_hourly_full, hist_tail, HOURLY_FEATURES, _gj_holidays, n_steps=6
    )

    span_daily = build_span_daily_for_48h(last_ts, _gj_holidays)

    # Build recent daily history from hourly dataframe for operational-state features.
    hist_daily = (
        df_hourly.set_index("rtc_timestamp")["consumption_kwh"]
        .resample("1D")
        .sum()
        .reset_index()
        .rename(columns={"rtc_timestamp": "ds", "consumption_kwh": "y"})
    )
    hist_daily["is_active"] = (hist_daily["y"] >= 50).astype(int)

    # Use the exact training feature order from models (v2.0/v2.1 compatible).
    clf_feats = list(getattr(xgb_clf_full, "feature_names_in_", []))
    reg_feats = list(getattr(xgb_reg_full, "feature_names_in_", []))
    if not clf_feats:
        raise ValueError("Classifier model does not expose feature_names_in_.")
    if not reg_feats:
        raise ValueError("Regressor model does not expose feature_names_in_.")
    # Predict daily totals sequentially so state features remain coherent.
    y_hist = hist_daily["y"].tolist()
    active_hist = hist_daily["is_active"].tolist()
    zero_ratio_hist = (
        (df_hourly.assign(date=df_hourly["rtc_timestamp"].dt.date, is_zero=(df_hourly["consumption_kwh"] < 0.5).astype(int))
         .groupby("date")["is_zero"].mean().tolist())
        if len(df_hourly)
        else []
    )
    pred_rows = []
    for _, base in span_daily.iterrows():
        row = _build_state_row(base.to_dict(), y_hist, active_hist, zero_ratio_hist)

        # Ensure any missing model features are present.
        for c in clf_feats:
            if c not in row:
                row[c] = 0.0
        for c in reg_feats:
            if c not in row:
                row[c] = 0.0
        row_df = pd.DataFrame([row])
        prob = float(xgb_clf_full.predict_proba(row_df[clf_feats])[:, 1][0])
        reg = float(xgb_reg_full.predict(row_df[reg_feats]).clip(min=0)[0])
        is_active = int(prob >= SHORT_TERM_SHUTDOWN_THRESH)
        pred = reg * recency_full if is_active == 1 else 0.0
        row["clf_prob"] = prob
        row["clf_active"] = is_active
        row["pred"] = float(pred)
        pred_rows.append(row)

        y_hist.append(float(pred))
        active_hist.append(is_active)
        pseudo_zero_ratio = float(np.clip(1.0 - (pred / max(1.0, 100.0)), 0.0, 1.0))
        zero_ratio_hist.append(pseudo_zero_ratio)

    span_daily = pd.DataFrame(pred_rows)

    if shape_model is not None:
        span_hourly = predict_future_hourly_recursive(
            shape_model, df_hourly, span_daily, dow_profiles, daytype_profiles, _gj_holidays
        )
    else:
        span_hourly = decompose_daily_to_hourly(span_daily, dow_profiles, daytype_profiles)
    fc_daily_part = span_hourly[
        (span_hourly["rtc_timestamp"] > last_ts + timedelta(hours=6))
        & (span_hourly["rtc_timestamp"] <= last_ts + timedelta(hours=48))
    ].copy()

    blend_rows = []
    for step in range(len(fc_xgb)):
        ts = fc_xgb.iloc[step]["rtc_timestamp"]
        xgb_val = fc_xgb.iloc[step]["forecast_kwh"]
        daily_match = span_hourly[span_hourly["rtc_timestamp"] == ts]
        daily_val = daily_match["forecast_kwh"].iloc[0] if len(daily_match) > 0 else xgb_val
        hour_idx = step + 1
        if hour_idx <= 3:
            blend_rows.append({"rtc_timestamp": ts, "forecast_kwh": xgb_val})
        else:
            w = (hour_idx - 3) / 3.0
            blended = (1 - w) * xgb_val + w * daily_val
            blend_rows.append({"rtc_timestamp": ts, "forecast_kwh": blended})

    fc_blended = pd.DataFrame(blend_rows)
    fc_48h = pd.concat([fc_blended, fc_daily_part], ignore_index=True)
    return fc_48h.sort_values("rtc_timestamp").reset_index(drop=True)
