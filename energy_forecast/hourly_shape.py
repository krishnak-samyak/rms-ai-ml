"""Day-ahead hourly shape model (v2.4): pure ML hourly decomposition.

Instead of DOW profile fractions, trains an XGBRegressor to predict each hour's
consumption from same-hour historical lags, calendar features, and the profile
baseline.  Predicted hours are rescaled so each day sums to the daily model's
predicted total — giving ML-driven shape while preserving the daily budget.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

SHAPE_FEATURES: list[str] = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "is_weekend",
    "is_working_day",
    "is_working_hour",
    "is_sunday",
    "mon_sin",
    "mon_cos",
    "day_of_month",
    "shift_sin",
    "shift_cos",
    "profile_frac",
    "same_hour_lag1d",
    "same_hour_lag2d",
    "same_hour_lag7d",
    "same_hour_mean_7d",
    "same_hour_same_dow_mean_4w",
    "same_hour_frac_lag1d",
    "same_hour_frac_lag7d",
    "prev_day_total",
    "prev_7d_mean_total",
]

SHUTDOWN_THRESH = 50


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_profile_lookup(
    dow_profiles: pd.DataFrame,
    daytype_profiles: pd.DataFrame,
) -> np.ndarray:
    """7×24 profile-fraction lookup table."""
    lut = np.zeros((7, 24))
    for d in range(7):
        if d in dow_profiles.columns:
            lut[d] = dow_profiles[d].values
        else:
            dt = "weekend" if d >= 5 else "working"
            if dt in daytype_profiles.columns:
                lut[d] = daytype_profiles[dt].values
            else:
                lut[d] = daytype_profiles.iloc[:, 0].values
    return lut


def _shift_code(h: int) -> int:
    if 6 <= h < 14:
        return 1
    if 14 <= h < 22:
        return 2
    return 0


# ---------------------------------------------------------------------------
# feature builder (vectorised, pivot-based for correct gap handling)
# ---------------------------------------------------------------------------

def build_shape_feature_frame(
    df_hourly: pd.DataFrame,
    dow_profiles: pd.DataFrame,
    daytype_profiles: pd.DataFrame,
    _gj: Any,
) -> pd.DataFrame:
    """Compute day-ahead hourly shape features for every row in *df_hourly*.

    Returns a copy with SHAPE_FEATURES + metadata columns appended.
    """
    df = (
        df_hourly[["rtc_timestamp", "consumption_kwh"]]
        .copy()
        .sort_values("rtc_timestamp")
        .reset_index(drop=True)
    )
    ts = df["rtc_timestamp"]
    hour = ts.dt.hour.astype(int)
    dow = ts.dt.dayofweek.astype(int)
    month = ts.dt.month.astype(int)
    df["_date"] = ts.dt.normalize()
    df["_hour"] = hour

    # --- calendar / cyclical features ----------------------------------------
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    is_hol = np.fromiter(
        (1 if d in _gj else 0 for d in ts.dt.date), dtype=np.int32, count=len(df)
    )
    df["is_holiday"] = is_hol
    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_working_day"] = ((df["is_weekend"] == 0) & (df["is_holiday"] == 0)).astype(int)
    df["is_working_hour"] = (
        (hour >= 9) & (hour <= 18) & (df["is_working_day"] == 1)
    ).astype(int)
    df["is_sunday"] = (dow == 6).astype(int)
    df["mon_sin"] = np.sin(2 * np.pi * month / 12)
    df["mon_cos"] = np.cos(2 * np.pi * month / 12)
    df["day_of_month"] = ts.dt.day.astype(float)
    shift_val = hour.map(_shift_code)
    df["shift_sin"] = np.sin(2 * np.pi * shift_val / 3)
    df["shift_cos"] = np.cos(2 * np.pi * shift_val / 3)

    # --- profile fraction ----------------------------------------------------
    lut = _build_profile_lookup(dow_profiles, daytype_profiles)
    df["profile_frac"] = lut[dow.values, hour.values]

    # --- same-hour lags via pivot table (handles missing hours) --------------
    pivot = df.pivot_table(
        index="_date", columns="_hour", values="consumption_kwh", aggfunc="first"
    )
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = np.nan
    pivot = pivot[sorted(pivot.columns)]

    lag1 = pivot.shift(1)
    lag2 = pivot.shift(2)
    lag7 = pivot.shift(7)
    mean7_arr = np.nanmean(
        np.stack([pivot.shift(d).values for d in range(1, 8)], axis=0), axis=0
    )
    mean7 = pd.DataFrame(mean7_arr, index=pivot.index, columns=pivot.columns)
    dow_mean_arr = np.nanmean(
        np.stack([pivot.shift(7 * w).values for w in range(1, 5)], axis=0), axis=0
    )
    dow_mean = pd.DataFrame(dow_mean_arr, index=pivot.index, columns=pivot.columns)

    daily_totals = pivot.sum(axis=1, min_count=1)
    prev_total = daily_totals.shift(1)
    prev_7d_mean = daily_totals.shift(1).rolling(7, min_periods=1).mean()

    def _melt(piv: pd.DataFrame, name: str) -> pd.DataFrame:
        m = piv.reset_index().melt(id_vars=["_date"], var_name="_hour", value_name=name)
        m["_hour"] = m["_hour"].astype(int)
        return m

    for m_df in [
        _melt(lag1, "same_hour_lag1d"),
        _melt(lag2, "same_hour_lag2d"),
        _melt(lag7, "same_hour_lag7d"),
        _melt(mean7, "same_hour_mean_7d"),
        _melt(dow_mean, "same_hour_same_dow_mean_4w"),
    ]:
        df = df.merge(m_df, on=["_date", "_hour"], how="left")

    df["prev_day_total"] = df["_date"].map(prev_total)
    df["prev_7d_mean_total"] = df["_date"].map(prev_7d_mean)

    df["daily_total"] = df["_date"].map(daily_totals)
    df["is_active_day"] = (df["daily_total"].fillna(0) >= SHUTDOWN_THRESH).astype(int)

    df = df.fillna(0.0)

    # Fraction-based lags (normalised by previous day's total — pure shape signal).
    pdt = df["prev_day_total"].clip(lower=1.0)
    df["same_hour_frac_lag1d"] = df["same_hour_lag1d"] / pdt
    p7dt = df["prev_7d_mean_total"].clip(lower=1.0)
    df["same_hour_frac_lag7d"] = df["same_hour_lag7d"] / (p7dt * 24).clip(lower=1.0)

    # Target fraction (only meaningful for active days).
    dt_safe = df["daily_total"].clip(lower=1.0)
    df["target_frac"] = df["consumption_kwh"] / dt_safe

    return df


# ---------------------------------------------------------------------------
# model training
# ---------------------------------------------------------------------------

def train_shape_model(x: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    """Train a day-ahead hourly shape regressor on **fraction** targets."""
    n = len(x)
    if n < 48:
        raise ValueError(f"Shape model needs ≥48 training rows; got {n}.")
    holdout = max(1, int(n * 0.15))
    x_tr, y_tr = x.iloc[:-holdout], y.iloc[:-holdout]
    x_ev, y_ev = x.iloc[-holdout:], y.iloc[-holdout:]
    model = XGBRegressor(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(x_tr, y_tr, eval_set=[(x_ev, y_ev)], verbose=False)
    return model


# ---------------------------------------------------------------------------
# inference: validation (features already computed from actual history)
# ---------------------------------------------------------------------------

def predict_and_rescale(
    model: XGBRegressor,
    feat_frame: pd.DataFrame,
    day_targets: pd.DataFrame | None = None,
    rescale: bool = True,
) -> pd.DataFrame:
    """Predict hourly consumption. If *rescale*, redistribute to daily totals."""
    raw = np.clip(model.predict(feat_frame[SHAPE_FEATURES]), 0.0, None)

    out = feat_frame[["rtc_timestamp"]].copy().reset_index(drop=True)
    out["_date"] = out["rtc_timestamp"].dt.normalize()
    out["raw_kwh"] = raw

    if not rescale or day_targets is None:
        out["forecast_kwh"] = out["raw_kwh"]
        return out[["rtc_timestamp", "forecast_kwh"]]

    tgt = day_targets[["ds", "pred"]].copy()
    tgt["_date"] = pd.to_datetime(tgt["ds"]).dt.normalize()
    out = out.merge(tgt[["_date", "pred"]], on="_date", how="left")
    out["pred"] = out["pred"].fillna(0.0)

    sum_d = out.groupby("_date", sort=False)["raw_kwh"].transform("sum")
    out["forecast_kwh"] = np.where(
        out["pred"] < 1e-9,
        0.0,
        np.where(sum_d > 1e-12, out["raw_kwh"] * (out["pred"] / sum_d), 0.0),
    )
    return out[["rtc_timestamp", "forecast_kwh"]]


# ---------------------------------------------------------------------------
# inference: future / hybrid (recursive — no actual data for target days)
# ---------------------------------------------------------------------------

def predict_future_hourly_recursive(
    model: XGBRegressor,
    df_hourly: pd.DataFrame,
    future_daily_df: pd.DataFrame,
    dow_profiles: pd.DataFrame,
    daytype_profiles: pd.DataFrame,
    _gj: Any,
) -> pd.DataFrame:
    """Predict hourly shape for future days, feeding predictions back as lags."""
    lookup: dict[pd.Timestamp, float] = dict(
        zip(
            df_hourly["rtc_timestamp"].tolist(),
            df_hourly["consumption_kwh"].astype(float).tolist(),
        )
    )
    lut = _build_profile_lookup(dow_profiles, daytype_profiles)
    all_hours: list[dict[str, Any]] = []

    for _, day_row in future_daily_df.sort_values("ds").iterrows():
        ds = pd.Timestamp(day_row["ds"]).normalize()
        day_pred = float(day_row.get("pred", 0.0))

        if day_pred < 1e-9:
            for h in range(24):
                all_hours.append(
                    {"rtc_timestamp": ds + timedelta(hours=h), "forecast_kwh": 0.0}
                )
            continue

        dow_val = ds.dayofweek
        month_val = ds.month
        is_hol = 1 if ds.date() in _gj else 0
        is_we = 1 if dow_val >= 5 else 0
        is_wd = 1 if (is_we == 0 and is_hol == 0) else 0
        is_sun = 1 if dow_val == 6 else 0

        prev_day = ds - timedelta(days=1)
        prev_day_total = sum(
            lookup.get(prev_day + timedelta(hours=hh), 0.0) for hh in range(24)
        )
        prev_7d_totals = [
            sum(
                lookup.get(ds - timedelta(days=d) + timedelta(hours=hh), 0.0)
                for hh in range(24)
            )
            for d in range(1, 8)
        ]
        prev_7d_mean_total = float(np.mean(prev_7d_totals))
        pdt_safe = max(prev_day_total, 1.0)
        p7dt_safe = max(prev_7d_mean_total, 1.0)

        feats: list[dict[str, float]] = []
        for h in range(24):
            t = ds + timedelta(hours=h)
            is_wh = 1 if (9 <= h <= 18 and is_wd == 1) else 0
            sv = _shift_code(h)
            lag1 = lookup.get(t - timedelta(days=1), 0.0)
            lag2 = lookup.get(t - timedelta(days=2), 0.0)
            lag7 = lookup.get(t - timedelta(days=7), 0.0)
            mean7 = float(
                np.mean(
                    [lookup.get(t - timedelta(days=d), 0.0) for d in range(1, 8)]
                )
            )
            dow_mean = float(
                np.mean(
                    [lookup.get(t - timedelta(weeks=w), 0.0) for w in range(1, 5)]
                )
            )
            feats.append(
                {
                    "hour_sin": np.sin(2 * np.pi * h / 24),
                    "hour_cos": np.cos(2 * np.pi * h / 24),
                    "dow_sin": np.sin(2 * np.pi * dow_val / 7),
                    "dow_cos": np.cos(2 * np.pi * dow_val / 7),
                    "is_holiday": float(is_hol),
                    "is_weekend": float(is_we),
                    "is_working_day": float(is_wd),
                    "is_working_hour": float(is_wh),
                    "is_sunday": float(is_sun),
                    "mon_sin": np.sin(2 * np.pi * month_val / 12),
                    "mon_cos": np.cos(2 * np.pi * month_val / 12),
                    "day_of_month": float(ds.day),
                    "shift_sin": np.sin(2 * np.pi * sv / 3),
                    "shift_cos": np.cos(2 * np.pi * sv / 3),
                    "profile_frac": float(lut[dow_val, h]),
                    "same_hour_lag1d": lag1,
                    "same_hour_lag2d": lag2,
                    "same_hour_lag7d": lag7,
                    "same_hour_mean_7d": mean7,
                    "same_hour_same_dow_mean_4w": dow_mean,
                    "same_hour_frac_lag1d": lag1 / pdt_safe,
                    "same_hour_frac_lag7d": lag7 / (p7dt_safe * 24),
                    "prev_day_total": prev_day_total,
                    "prev_7d_mean_total": prev_7d_mean_total,
                }
            )

        feat_df = pd.DataFrame(feats)
        raw_kwh = np.clip(model.predict(feat_df[SHAPE_FEATURES]), 0.0, None)
        raw_sum = float(raw_kwh.sum())
        scaled = (
            raw_kwh * (day_pred / raw_sum) if raw_sum > 1e-12
            else np.full(24, day_pred / 24.0)
        )

        for h in range(24):
            t = ds + timedelta(hours=h)
            fkwh = float(scaled[h])
            all_hours.append({"rtc_timestamp": t, "forecast_kwh": fkwh})
            lookup[t] = fkwh

    return pd.DataFrame(all_hours)
