"""Daily aggregation and two-stage XGBoost (classifier + regressor)."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

from energy_forecast.constants import (
    ACTIVE_HOUR_THRESH,
    DAILY_FEATURES,
    DAILY_FEATURES_V21,
    DAILY_FEATURES_V22,
    PROB_THRESH,
    RECENT_WINDOW,
    SHUTDOWN_THRESH,
)


def build_daily_agg(df: pd.DataFrame) -> pd.DataFrame:
    daily_df = df.copy()
    daily_df["date"] = daily_df["rtc_timestamp"].dt.date
    daily_agg = daily_df.groupby("date").agg(
        y=("consumption_kwh", "sum"),
        is_holiday=("is_holiday", "first"),
        is_weekend=("is_weekend", "first"),
        is_working_day=("is_working_day", "first"),
        daily_zero_hour_ratio=("consumption_kwh", lambda s: float((s < 0.5).mean())),
    ).reset_index()
    daily_agg["ds"] = pd.to_datetime(daily_agg["date"])
    daily_agg["dow"] = daily_agg["ds"].dt.dayofweek
    daily_agg["dow_sin"] = np.sin(2 * np.pi * daily_agg["dow"] / 7)
    daily_agg["dow_cos"] = np.cos(2 * np.pi * daily_agg["dow"] / 7)
    daily_agg["month"] = daily_agg["ds"].dt.month
    daily_agg["mon_sin"] = np.sin(2 * np.pi * daily_agg["month"] / 12)
    daily_agg["mon_cos"] = np.cos(2 * np.pi * daily_agg["month"] / 12)
    daily_agg["day_of_month"] = daily_agg["ds"].dt.day
    daily_agg["is_sunday"] = (daily_agg["dow"] == 6).astype(int)
    daily_agg["is_active"] = (daily_agg["y"] >= SHUTDOWN_THRESH).astype(int)

    # v2.1 operational-state features (from past daily behavior)
    daily_agg["y_lag_1"] = daily_agg["y"].shift(1)
    daily_agg["y_lag_2"] = daily_agg["y"].shift(2)
    daily_agg["y_lag_7"] = daily_agg["y"].shift(7)
    daily_agg["y_roll_mean_3"] = daily_agg["y"].shift(1).rolling(3).mean()
    daily_agg["y_roll_mean_7"] = daily_agg["y"].shift(1).rolling(7).mean()
    daily_agg["y_roll_mean_14"] = daily_agg["y"].shift(1).rolling(14).mean()
    daily_agg["recent_shutdown_ratio_7"] = (1 - daily_agg["is_active"]).shift(1).rolling(7).mean()
    daily_agg["recent_shutdown_ratio_14"] = (1 - daily_agg["is_active"]).shift(1).rolling(14).mean()
    daily_agg["last_24h_mean_cons"] = daily_agg["y_lag_1"] / 24.0
    daily_agg["last_48h_mean_cons"] = daily_agg[["y_lag_1", "y_lag_2"]].mean(axis=1) / 24.0
    daily_agg["last_72h_zero_hour_ratio"] = daily_agg["daily_zero_hour_ratio"].shift(1).rolling(3).mean()

    # Consecutive shutdown streak up to previous day.
    zero_streak_prev = []
    streak = 0
    prev_vals = daily_agg["is_active"].shift(1).fillna(1).astype(int).tolist()
    for v in prev_vals:
        if v == 0:
            streak += 1
        else:
            streak = 0
        zero_streak_prev.append(streak)
    daily_agg["zero_streak_prev"] = zero_streak_prev

    # Safe defaults for early rows.
    y_med = float(daily_agg["y"].median()) if len(daily_agg) else 0.0
    daily_agg["y_lag_1"] = daily_agg["y_lag_1"].fillna(y_med)
    daily_agg["y_lag_2"] = daily_agg["y_lag_2"].fillna(y_med)
    daily_agg["y_lag_7"] = daily_agg["y_lag_7"].fillna(y_med)
    daily_agg["y_roll_mean_3"] = daily_agg["y_roll_mean_3"].fillna(y_med)
    daily_agg["y_roll_mean_7"] = daily_agg["y_roll_mean_7"].fillna(y_med)
    daily_agg["y_roll_mean_14"] = daily_agg["y_roll_mean_14"].fillna(y_med)
    daily_agg["recent_shutdown_ratio_7"] = daily_agg["recent_shutdown_ratio_7"].fillna(0.0)
    daily_agg["recent_shutdown_ratio_14"] = daily_agg["recent_shutdown_ratio_14"].fillna(0.0)
    daily_agg["daily_zero_hour_ratio"] = daily_agg["daily_zero_hour_ratio"].fillna(0.0)
    daily_agg["last_24h_mean_cons"] = daily_agg["last_24h_mean_cons"].fillna(y_med / 24.0)
    daily_agg["last_48h_mean_cons"] = daily_agg["last_48h_mean_cons"].fillna(y_med / 24.0)
    daily_agg["last_72h_zero_hour_ratio"] = daily_agg["last_72h_zero_hour_ratio"].fillna(0.0)
    return daily_agg


def train_two_stage(
    train_daily: pd.DataFrame,
    daily_features: list[str] | None = None,
) -> tuple[XGBClassifier, XGBRegressor, float]:
    feats = daily_features or DAILY_FEATURES
    n_active_train = int(train_daily["is_active"].sum())
    n_shutdown_train = len(train_daily) - n_active_train
    spw = n_active_train / max(n_shutdown_train, 1)

    xgb_clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    xgb_clf.fit(train_daily[feats], train_daily["is_active"], verbose=False)

    train_active = train_daily[train_daily["is_active"] == 1]
    xgb_reg = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
    )
    xgb_reg.fit(train_active[feats], train_active["y"], verbose=False)
    return xgb_clf, xgb_reg, spw


def recency_ratio(train_daily: pd.DataFrame, train_active: pd.DataFrame) -> float:
    recent_active = train_daily[train_daily["is_active"] == 1].tail(RECENT_WINDOW)
    if len(recent_active) == 0 or len(train_active) == 0:
        return 1.0
    ratio = recent_active["y"].mean() / train_active["y"].mean()
    return float(np.clip(ratio, 0.5, 1.5))


def predict_daily_two_stage(
    daily_df: pd.DataFrame,
    xgb_clf: XGBClassifier,
    xgb_reg: XGBRegressor,
    rec_ratio: float,
    threshold: float = PROB_THRESH,
    daily_features: list[str] | None = None,
) -> pd.DataFrame:
    feats = daily_features or DAILY_FEATURES
    out = daily_df.copy()
    probs = xgb_clf.predict_proba(out[feats])[:, 1]
    reg_pred = xgb_reg.predict(out[feats]).clip(min=0)
    out["clf_prob"] = probs
    out["clf_active"] = (probs >= threshold).astype(int)
    out["pred_raw"] = np.where(out["clf_active"] == 1, reg_pred, 0.0)
    out["pred"] = out["pred_raw"] * rec_ratio
    return out


def tune_shutdown_threshold(val_df: pd.DataFrame) -> tuple[float, dict]:
    """
    v2.1 threshold tuning prioritizing shutdown recall while keeping false shutdowns low.
    Uses F2 on shutdown class (beta=2) and returns threshold + diagnostics.
    """
    best_thr = PROB_THRESH
    best_score = -1.0
    best_diag: dict = {}

    y_true_shutdown = (val_df["is_active"] == 0).astype(int)
    probs_active = val_df["clf_prob"].values
    for thr in np.arange(0.05, 0.61, 0.01):
        pred_active = (probs_active >= thr).astype(int)
        pred_shutdown = (pred_active == 0).astype(int)

        tp = int(((pred_shutdown == 1) & (y_true_shutdown == 1)).sum())
        fp = int(((pred_shutdown == 1) & (y_true_shutdown == 0)).sum())
        fn = int(((pred_shutdown == 0) & (y_true_shutdown == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        beta2 = 4.0
        denom = (beta2 * precision + recall)
        f2 = ((1 + beta2) * precision * recall / denom) if denom else 0.0

        if f2 > best_score:
            best_score = f2
            best_thr = float(thr)
            best_diag = {
                "precision_shutdown": float(precision),
                "recall_shutdown": float(recall),
                "f2_shutdown": float(f2),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
    return best_thr, best_diag


def tune_threshold_total_error(
    val_df: pd.DataFrame,
    rec_ratio: float,
    lambda_fn: float = 0.2,
) -> tuple[float, dict]:
    """
    v2.2 threshold tuning objective:
    minimize abs(monthly_total_error_pct) + lambda_fn * FN_rate
    """
    best_thr = PROB_THRESH
    best_obj = float("inf")
    best_diag: dict = {}
    y_true = val_df["y"].values
    probs = val_df["clf_prob"].values
    pred_raw = val_df["pred_raw"].values
    is_active_true = val_df["is_active"].values
    actual_total = float(val_df["y"].sum())

    for thr in np.arange(0.05, 0.81, 0.01):
        pred_active = (probs >= thr).astype(int)
        pred = np.where(pred_active == 1, pred_raw * rec_ratio, 0.0)
        pred_total = float(np.sum(pred))
        total_err_pct = ((pred_total - actual_total) / actual_total * 100) if actual_total else 0.0
        fn = int(((pred_active == 1) & (is_active_true == 0)).sum())
        n_shutdown = int((is_active_true == 0).sum())
        fn_rate = fn / n_shutdown if n_shutdown else 0.0
        obj = abs(total_err_pct) + lambda_fn * (fn_rate * 100.0)

        if obj < best_obj:
            best_obj = obj
            best_thr = float(thr)
            tp = int(((pred_active == 0) & (is_active_true == 0)).sum())
            fp = int(((pred_active == 0) & (is_active_true == 1)).sum())
            best_diag = {
                "objective": float(obj),
                "val_total_error_pct": float(total_err_pct),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "fn_rate": float(fn_rate),
                "lambda_fn": float(lambda_fn),
            }

    return best_thr, best_diag


def add_day_type_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_type"] = out.apply(
        lambda r: "holiday"
        if r["is_holiday"]
        else ("weekend" if r["is_weekend"] else "working"),
        axis=1,
    )
    return out


def build_future_daily_rows(
    last_date: pd.Timestamp,
    forecast_days: int,
    _gj_holidays,
) -> pd.DataFrame:
    future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]
    future_daily = pd.DataFrame({"ds": future_dates})
    future_daily["is_holiday"] = future_daily["ds"].dt.date.apply(lambda d: 1 if d in _gj_holidays else 0)
    future_daily["is_weekend"] = (future_daily["ds"].dt.dayofweek >= 5).astype(int)
    future_daily["is_working_day"] = (
        (future_daily["is_weekend"] == 0) & (future_daily["is_holiday"] == 0)
    ).astype(int)
    future_daily["dow"] = future_daily["ds"].dt.dayofweek
    future_daily["is_sunday"] = (future_daily["dow"] == 6).astype(int)
    future_daily["dow_sin"] = np.sin(2 * np.pi * future_daily["dow"] / 7)
    future_daily["dow_cos"] = np.cos(2 * np.pi * future_daily["dow"] / 7)
    future_daily["month"] = future_daily["ds"].dt.month
    future_daily["mon_sin"] = np.sin(2 * np.pi * future_daily["month"] / 12)
    future_daily["mon_cos"] = np.cos(2 * np.pi * future_daily["month"] / 12)
    future_daily["day_of_month"] = future_daily["ds"].dt.day
    return future_daily


def fill_future_operational_features(
    future_daily: pd.DataFrame,
    history_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    v2.1 recursive daily feature fill:
    uses trailing predicted y and active/shutdown states to populate lag/rolling state features.
    """
    out = future_daily.copy()
    hist = history_daily[["ds", "y", "is_active"]].sort_values("ds").copy()
    y_hist = hist["y"].tolist()
    active_hist = hist["is_active"].tolist()

    y_med = float(np.median(y_hist)) if y_hist else 0.0
    rows = []
    for _, r in out.iterrows():
        y_lag_1 = y_hist[-1] if len(y_hist) >= 1 else y_med
        y_lag_2 = y_hist[-2] if len(y_hist) >= 2 else y_med
        y_lag_7 = y_hist[-7] if len(y_hist) >= 7 else y_med
        y_roll_mean_3 = float(np.mean(y_hist[-3:])) if len(y_hist) >= 3 else y_med
        y_roll_mean_7 = float(np.mean(y_hist[-7:])) if len(y_hist) >= 7 else y_med

        sh_hist = [1 - a for a in active_hist]
        recent_shutdown_ratio_7 = float(np.mean(sh_hist[-7:])) if len(sh_hist) >= 7 else 0.0

        streak = 0
        for a in reversed(active_hist):
            if a == 0:
                streak += 1
            else:
                break

        row = r.to_dict()
        row["y_lag_1"] = y_lag_1
        row["y_lag_2"] = y_lag_2
        row["y_lag_7"] = y_lag_7
        row["y_roll_mean_3"] = y_roll_mean_3
        row["y_roll_mean_7"] = y_roll_mean_7
        row["recent_shutdown_ratio_7"] = recent_shutdown_ratio_7
        row["zero_streak_prev"] = streak
        rows.append(row)

        # placeholders updated by caller after prediction; keep continuity if not overwritten
        y_hist.append(y_lag_1)
        active_hist.append(1)

    return pd.DataFrame(rows)


def validation_metrics(
    val_daily: pd.DataFrame,
    df_hourly: pd.DataFrame,
    val_hourly_forecast: pd.DataFrame,
) -> dict:
    tp = int(((val_daily["clf_active"] == 0) & (val_daily["is_active"] == 0)).sum())
    fp = int(((val_daily["clf_active"] == 0) & (val_daily["is_active"] == 1)).sum())
    fn = int(((val_daily["clf_active"] == 1) & (val_daily["is_active"] == 0)).sum())
    clf_acc = float(accuracy_score(val_daily["is_active"], val_daily["clf_active"]))

    val_active_mask = val_daily["is_active"] == 1
    val_active = val_daily[val_active_mask]
    daily_mae = daily_rmse = daily_mape = None
    if len(val_active) > 0:
        daily_mae = float(mean_absolute_error(val_active["y"], val_active["pred"]))
        daily_rmse = float(np.sqrt(mean_squared_error(val_active["y"], val_active["pred"])))
        daily_mape = float(
            np.mean(np.abs((val_active["y"] - val_active["pred"]) / val_active["y"].clip(lower=1))) * 100
        )

    val_merged = df_hourly.merge(val_hourly_forecast, on="rtc_timestamp", how="inner")
    hourly_mae = float(mean_absolute_error(val_merged["consumption_kwh"], val_merged["forecast_kwh"]))
    hourly_rmse = float(np.sqrt(mean_squared_error(val_merged["consumption_kwh"], val_merged["forecast_kwh"])))
    active_hrs = val_merged["consumption_kwh"] > ACTIVE_HOUR_THRESH
    hourly_mape = None
    if active_hrs.sum() > 0:
        hourly_mape = float(
            np.mean(
                np.abs(
                    (val_merged.loc[active_hrs, "consumption_kwh"] - val_merged.loc[active_hrs, "forecast_kwh"])
                    / val_merged.loc[active_hrs, "consumption_kwh"]
                )
            )
            * 100
        )

    actual_total = float(val_daily["y"].sum())
    pred_total = float(val_daily["pred"].sum())
    total_err_pct = (pred_total - actual_total) / actual_total * 100 if actual_total else 0.0

    return {
        "classifier_accuracy": clf_acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "daily_mae": daily_mae,
        "daily_rmse": daily_rmse,
        "daily_mape_pct": daily_mape,
        "hourly_mae": hourly_mae,
        "hourly_rmse": hourly_rmse,
        "hourly_mape_pct": hourly_mape,
        "active_hours_for_mape": int(active_hrs.sum()),
        "val_actual_total_kwh": actual_total,
        "val_pred_total_kwh": pred_total,
        "val_total_error_pct": float(total_err_pct),
    }
