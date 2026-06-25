"""Daily aggregation and two-stage XGBoost (classifier + regressor)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

from energy_forecast.constants import (
    ACTIVE_HOUR_THRESH,
    DAILY_CALIB_A_MAX,
    DAILY_CALIB_A_MIN,
    DAILY_CALIB_B_MAX_FRAC_OF_Y_MEAN,
    DAILY_CALIB_MIN_ACTIVE_VAL_DAYS,
    DAILY_ISO_MAPE_TIE_EPS,
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


def recency_ratio(train_daily: pd.DataFrame) -> float:
    """Scale active-day predictions toward recent operating level vs prior active level.

    **Bug fix (under-forecast at monthly total):** The old formula used
    ``mean(last N active days) / mean(all active days in train)``. Long-run
    ``all_active`` mean can sit far above or below the *current* regime; a low
    recent slice vs an inflated historical mean drives ratio < 1 and systematically
    shrinks ``pred``, worsening validation totals when the holdout is hotter/busier.

    **New formula:** ``mean(last RECENT_WINDOW active days) /
    mean(earlier active days in the same frame)``. That compares *recent* to
    *immediately prior* active history (same meter, same pipeline). Falls back to
    ``1.0`` if there are too few prior active rows.

    Still clipped to avoid extreme multipliers from small samples.
    """
    act = train_daily.loc[train_daily["is_active"] == 1].sort_values("ds")
    if len(act) < RECENT_WINDOW + 5:
        return 1.0
    recent = act.tail(RECENT_WINDOW)
    prior = act.iloc[: -len(recent)]
    if len(prior) < 5:
        return 1.0
    denom = float(prior["y"].mean())
    if denom <= 0:
        return 1.0
    ratio = float(recent["y"].mean() / denom)
    return float(np.clip(ratio, 0.75, 1.35))


def _active_day_mape_pct(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y) == 0:
        return float("nan")
    return float(np.mean(np.abs((y - pred) / np.clip(y, 1.0, None))) * 100.0)


def fit_daily_affine_calibration(
    val_daily: pd.DataFrame,
    min_active_days: int = DAILY_CALIB_MIN_ACTIVE_VAL_DAYS,
) -> tuple[float, float, dict[str, Any]]:
    """Fit ``y ≈ a * pred + b`` on validation rows where ``is_active`` (truth) is 1.

    **Guardrails**
    - Require at least ``min_active_days`` active truth rows.
    - Clip ``a`` to ``[DAILY_CALIB_A_MIN, DAILY_CALIB_A_MAX]``.
    - Clip ``b`` to ``± DAILY_CALIB_B_MAX_FRAC_OF_Y_MEAN * mean(y)``.
    - If clipped coefficients do **not** improve mean active-day MAPE vs raw ``pred``,
      disable calibration (return ``a=1, b=0``) so we never ship a harmful nudge.

    ``pred`` here must already include classifier + regressor + recency scaling.
    """
    mask = val_daily["is_active"].values == 1
    if int(mask.sum()) < min_active_days:
        return 1.0, 0.0, {
            "enabled": False,
            "reason": "insufficient_active_val_days",
            "n_active": int(mask.sum()),
        }

    va = val_daily.loc[mask]
    y = va["y"].values.astype(float)
    p = va["pred"].values.astype(float)
    mape_before = _active_day_mape_pct(y, p)

    lr = LinearRegression()
    lr.fit(p.reshape(-1, 1), y)
    a_raw = float(lr.coef_[0])
    b_raw = float(lr.intercept_)

    y_bar = float(np.mean(y))
    b_cap = DAILY_CALIB_B_MAX_FRAC_OF_Y_MEAN * max(y_bar, 1.0)
    a = float(np.clip(a_raw, DAILY_CALIB_A_MIN, DAILY_CALIB_A_MAX))
    b = float(np.clip(b_raw, -b_cap, b_cap))

    p_adj = np.clip(a * p + b, 0.0, None)
    mape_after = _active_day_mape_pct(y, p_adj)

    if mape_after >= mape_before * 0.999:
        return 1.0, 0.0, {
            "enabled": False,
            "reason": "no_mape_improvement",
            "mape_before_pct": mape_before,
            "mape_after_clipped_pct": mape_after,
            "a_raw": a_raw,
            "b_raw": b_raw,
        }

    return a, b, {
        "enabled": True,
        "mape_before_pct": mape_before,
        "mape_after_pct": mape_after,
        "a_raw": a_raw,
        "b_raw": b_raw,
        "a_applied": a,
        "b_applied": b,
        "n_active_fit": int(len(y)),
    }


def apply_daily_calibration(
    pred: float | np.ndarray,
    a: float,
    b: float,
    *,
    enabled: bool = True,
) -> float | np.ndarray:
    """Apply stored affine calibration to non-negative daily kWh predictions."""
    if not enabled or (abs(a - 1.0) < 1e-9 and abs(b) < 1e-9):
        return pred
    arr = np.asarray(pred, dtype=float)
    out = np.clip(float(a) * arr + float(b), 0.0, None)
    if np.ndim(pred) == 0:
        return float(out)
    return out


def fit_daily_isotonic_calibration(
    val_daily: pd.DataFrame,
    min_active_days: int = DAILY_CALIB_MIN_ACTIVE_VAL_DAYS,
) -> tuple[tuple[list[float], list[float]] | None, dict[str, Any]]:
    """Fit a **monotone increasing** mapping ``pred → y`` on active validation days.

    Isotonic regression can correct **level vs load** curvature (e.g. under-forecast
    at high kWh) that a single affine line cannot, while forbidding wiggles that
    would invert the ordering of predictions.

    **Guardrails:** same minimum active rows as affine; must improve mean active-day
    MAPE vs raw ``pred``; require at least two isotonic knots.
    """
    mask = val_daily["is_active"].values == 1
    if int(mask.sum()) < min_active_days:
        return None, {
            "enabled": False,
            "reason": "insufficient_active_val_days",
            "n_active": int(mask.sum()),
        }

    va = val_daily.loc[mask]
    y = va["y"].values.astype(float)
    p = va["pred"].values.astype(float)
    mape_before = _active_day_mape_pct(y, p)

    ir = IsotonicRegression(y_min=0.0, increasing=True, out_of_bounds="clip")
    ir.fit(p, y)
    y_hat = np.clip(ir.predict(p), 0.0, None)
    mape_after = _active_day_mape_pct(y, y_hat)

    if mape_after >= mape_before * 0.999:
        return None, {
            "enabled": False,
            "reason": "no_mape_improvement",
            "mape_before_pct": mape_before,
            "mape_after_pct": mape_after,
        }

    xs = ir.X_thresholds_.astype(float).tolist()
    ys = ir.y_thresholds_.astype(float).tolist()
    if len(xs) < 2:
        return None, {
            "enabled": False,
            "reason": "degenerate_isotonic",
            "n_knots": len(xs),
            "mape_after_pct": mape_after,
        }

    return (xs, ys), {
        "enabled": True,
        "mape_before_pct": mape_before,
        "mape_after_pct": mape_after,
        "n_knots": len(xs),
        "n_active_fit": int(len(y)),
    }


def select_daily_calibration(val_daily: pd.DataFrame) -> dict[str, Any]:
    """Fit affine + isotonic on validation; pick one that lowers MAPE with guardrails.

    If both qualify, prefer **isotonic** only when it beats affine by at least
    ``DAILY_ISO_MAPE_TIE_EPS`` (percentage points); otherwise keep **affine** (simpler).
    """
    aff_a, aff_b, aff_diag = fit_daily_affine_calibration(val_daily)
    iso_knots, iso_diag = fit_daily_isotonic_calibration(val_daily)

    iso_ok = bool(iso_diag.get("enabled")) and iso_knots is not None
    aff_ok = bool(aff_diag.get("enabled"))
    iso_x: list[float] = list(iso_knots[0]) if iso_ok else []
    iso_y: list[float] = list(iso_knots[1]) if iso_ok else []

    mode = "none"
    if iso_ok and aff_ok:
        mi = float(iso_diag["mape_after_pct"])
        ma = float(aff_diag["mape_after_pct"])
        if mi + DAILY_ISO_MAPE_TIE_EPS < ma:
            mode = "isotonic"
        else:
            mode = "affine"
    elif iso_ok:
        mode = "isotonic"
    elif aff_ok:
        mode = "affine"

    enabled = mode != "none"
    return {
        "mode": mode,
        "enabled": enabled,
        "affine_a": float(aff_a),
        "affine_b": float(aff_b),
        "iso_x": iso_x,
        "iso_y": iso_y,
        "affine_diag": aff_diag,
        "iso_diag": iso_diag,
    }


def apply_daily_postcalibration(
    pred: float | np.ndarray,
    *,
    mode: str,
    enabled: bool,
    affine_a: float = 1.0,
    affine_b: float = 0.0,
    iso_x: list[float] | None = None,
    iso_y: list[float] | None = None,
) -> float | np.ndarray:
    """Apply stored daily post-calibration (affine **or** isotonic)."""
    if not enabled or mode == "none":
        return pred

    if mode == "affine":
        return apply_daily_calibration(pred, affine_a, affine_b, enabled=True)

    if mode == "isotonic":
        ix = iso_x or []
        iy = iso_y or []
        if len(ix) < 2 or len(iy) < 2:
            return pred
        xk = np.asarray(ix, dtype=float)
        yk = np.asarray(iy, dtype=float)
        arr = np.asarray(pred, dtype=float)
        out = np.interp(arr, xk, yk, left=float(yk[0]), right=float(yk[-1]))
        out = np.clip(out, 0.0, None)
        if np.ndim(pred) == 0:
            return float(out)
        return out

    return pred


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
    lambda_bal: float = 0.2,
) -> tuple[float, dict]:

    best_thr = PROB_THRESH
    best_obj = float("inf")
    best_diag = {}

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

        # Correct definitions
        fp = int(((pred_active == 1) & (is_active_true == 0)).sum())  # false activation
        fn = int(((pred_active == 0) & (is_active_true == 1)).sum())  # missed active

        n_shutdown = int((is_active_true == 0).sum())
        n_active = int((is_active_true == 1).sum())

        fp_rate = fp / n_shutdown if n_shutdown else 0.0
        fn_rate = fn / n_active if n_active else 0.0

        # Balanced objective
        obj = abs(total_err_pct) + lambda_bal * ((fp_rate + fn_rate) * 100.0)

        if obj < best_obj:
            best_obj = obj
            best_thr = float(thr)
            best_diag = {
                "objective": float(obj),
                "val_total_error_pct": float(total_err_pct),
                "fp": fp,
                "fn": fn,
                "fp_rate": float(fp_rate),
                "fn_rate": float(fn_rate),
                "lambda_bal": float(lambda_bal),
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
    df_hourly: pd.DataFrame = None,
    val_hourly_forecast: pd.DataFrame = None,
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

    # val_merged = df_hourly.merge(val_hourly_forecast, on="rtc_timestamp", how="inner")
    # hourly_mae = float(mean_absolute_error(val_merged["consumption_kwh"], val_merged["forecast_kwh"]))
    # hourly_rmse = float(np.sqrt(mean_squared_error(val_merged["consumption_kwh"], val_merged["forecast_kwh"])))
    # active_hrs = val_merged["consumption_kwh"] > ACTIVE_HOUR_THRESH
    # hourly_mape = None
    # if active_hrs.sum() > 0:
    #     hourly_mape = float(
    #         np.mean(
    #             np.abs(
    #                 (val_merged.loc[active_hrs, "consumption_kwh"] - val_merged.loc[active_hrs, "forecast_kwh"])
    #                 / val_merged.loc[active_hrs, "consumption_kwh"]
    #             )
    #         )
    #         * 100
    #     )

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
        # "hourly_mae": hourly_mae,
        # "hourly_rmse": hourly_rmse,
        # "hourly_mape_pct": hourly_mape,
        # "active_hours_for_mape": int(active_hrs.sum()),
        "val_actual_total_kwh": actual_total,
        "val_pred_total_kwh": pred_total,
        "val_total_error_pct": float(total_err_pct),
    }
