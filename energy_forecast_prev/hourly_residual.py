"""ML correction for profile-only hourly shape (residual vs DOW baseline).

Trains a conservative XGBRegressor on (actual - baseline) for active days, then
applies predictions and renormalizes so each day still sums to the predicted daily total.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from energy_forecast.decompose import profile_fraction_for_hour

logger = logging.getLogger(__name__)
RESIDUAL_BLEND = 0.65

# Keep features small and aligned at train/inference (no usage lags — avoids leakage at horizon).
RESIDUAL_FEATURES: list[str] = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_holiday",
    "is_weekend",
    "is_sunday",
    "mon_sin",
    "mon_cos",
    "day_of_month",
    "profile_frac",
    "baseline_kwh",
    "day_total",
]


def _calendar_features(rtc: pd.Series, _gj: Any) -> pd.DataFrame:
    t = pd.to_datetime(rtc)
    dt = t.dt
    is_holiday = np.fromiter((1 if d in _gj else 0 for d in dt.date), dtype=np.int32, count=len(t))
    dow = dt.dayofweek.astype(int)
    month = dt.month.astype(int)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
            "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "is_holiday": is_holiday,
            "is_weekend": (dow >= 5).astype(int),
            "is_sunday": (dow == 6).astype(int),
            "mon_sin": np.sin(2 * np.pi * month / 12),
            "mon_cos": np.cos(2 * np.pi * month / 12),
            "day_of_month": dt.day.astype(float),
        },
        index=t.index,
    )


def build_residual_training_frame(
    df_hourly: pd.DataFrame,
    daily_with_type: pd.DataFrame,
    dow_profiles: pd.DataFrame,
    daytype_profiles: pd.DataFrame,
    _gj: Any,
    val_start: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows: active days (optionally before val_start). Target: consumption - baseline."""
    dsub = daily_with_type.copy()
    dsub["ds"] = pd.to_datetime(dsub["ds"]).dt.normalize()
    dsub = dsub.rename(columns={"y": "day_total"})
    df2 = df_hourly.copy()
    df2["_ds"] = pd.to_datetime(df2["rtc_timestamp"]).dt.normalize()
    m = df2.merge(
        dsub[["ds", "day_total", "is_active", "day_type"]],
        left_on="_ds",
        right_on="ds",
        how="inner",
    )
    m = m[m["is_active"] == 1]
    if val_start is not None:
        m = m[m["_ds"] < val_start]
    if len(m) < 48:
        logger.warning("Residual training: only %d rows; model may be weak.", len(m))

    def _row_frac(row: pd.Series) -> float:
        d0 = pd.Timestamp(row["_ds"]).normalize()
        h = int(pd.Timestamp(row["rtc_timestamp"]).hour)
        return profile_fraction_for_hour(
            d0, h, str(row.get("day_type", "working")), dow_profiles, daytype_profiles
        )

    m = m.reset_index(drop=True)
    m["profile_frac"] = m.apply(_row_frac, axis=1)
    m["baseline_kwh"] = m["day_total"].astype(float) * m["profile_frac"]
    y = (m["consumption_kwh"].astype(float) - m["baseline_kwh"]).to_numpy()
    cal = _calendar_features(m["rtc_timestamp"], _gj)
    x = pd.concat(
        [
            cal,
            m[["profile_frac", "baseline_kwh", "day_total"]].astype(float).reset_index(drop=True),
        ],
        axis=1,
    )
    return x[RESIDUAL_FEATURES], y


def train_hourly_residual_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> XGBRegressor:
    """Shallow, regularized model to improve shape without chasing perfect fit."""
    n = len(x_train)
    if n < 10:
        raise ValueError("Not enough rows to train hourly residual model.")
    base = dict(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=6,
        reg_alpha=0.3,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
    )
    if n < 60:
        model = XGBRegressor(**base)
        model.fit(x_train, y_train, verbose=False)
        return model
    last_15 = max(1, int(n * 0.15))
    x_tr = x_train.iloc[:-last_15]
    y_tr = y_train[:-last_15]
    x_ev = x_train.iloc[-last_15:]
    y_ev = y_train[-last_15:]
    model = XGBRegressor(
        n_estimators=400,
        early_stopping_rounds=25,
        **{k: v for k, v in base.items() if k != "n_estimators"},
    )
    model.fit(x_tr, y_tr, eval_set=[(x_ev, y_ev)], verbose=False)
    return model


def apply_hourly_residual(
    model: XGBRegressor,
    decomposed: pd.DataFrame,
    day_pred: pd.DataFrame,
    _gj: Any,
) -> pd.DataFrame:
    """Add residual prediction and renormalize hours to match each day_pred['pred']."""
    out = decomposed.copy()
    if len(out) == 0:
        return out
    out["rtc_timestamp"] = pd.to_datetime(out["rtc_timestamp"])
    day_pred = day_pred.copy()
    day_pred["ds"] = pd.to_datetime(day_pred["ds"]).dt.normalize()
    out["_ds"] = out["rtc_timestamp"].dt.normalize()
    out = out.merge(day_pred[["ds", "pred"]], left_on="_ds", right_on="ds", how="left", suffixes=("", "_day"))
    out["day_total"] = out["pred"].fillna(0.0).astype(float)
    out["baseline_kwh"] = out["forecast_kwh"].astype(float)
    out["profile_frac"] = np.where(
        out["day_total"] > 1e-9,
        out["baseline_kwh"] / out["day_total"],
        0.0,
    )
    cal = _calendar_features(out["rtc_timestamp"], _gj)
    x = pd.concat(
        [
            cal,
            out[["profile_frac", "baseline_kwh", "day_total"]].astype(float).reset_index(drop=True),
        ],
        axis=1,
    )[RESIDUAL_FEATURES]
    res_hat = model.predict(x)
    raw = np.clip(out["baseline_kwh"].to_numpy() + RESIDUAL_BLEND * res_hat, 0.0, None)
    out["raw_kwh"] = raw
    sum_d = out.groupby("_ds", sort=False)["raw_kwh"].transform("sum")
    dtot = out["day_total"]
    new_fc = np.where(
        dtot < 1e-9,
        0.0,
        np.where(
            sum_d > 1e-12,
            raw * (dtot / sum_d),
            0.0 * raw,
        ),
    )
    out["forecast_kwh"] = new_fc
    return out.drop(columns=[c for c in ["_ds", "ds", "pred", "day_total", "profile_frac", "baseline_kwh", "raw_kwh"] if c in out.columns], errors="ignore")
