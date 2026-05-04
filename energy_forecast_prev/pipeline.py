"""End-to-end pipeline: load, features, profiles, train, validate, forecast."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from energy_forecast.config import Settings
from energy_forecast.constants import (
    DAILY_FEATURES,
    DAYAHEAD_HOURLY_FEATURES,
    HOURLY_FEATURES,
    PROB_THRESH,
    RECENT_WINDOW,
)
from energy_forecast.daily import (
    add_day_type_column,
    build_daily_agg,
    build_future_daily_rows,
    predict_daily_two_stage,
    recency_ratio,
    train_two_stage,
    tune_threshold_total_error,
    validation_metrics,
)
from energy_forecast.data import load_raw_dataframe
from energy_forecast.decompose import decompose_daily_to_hourly
from energy_forecast.features import build_features, gj_holiday_calendar
from energy_forecast.hourly import hourly_test_metrics, train_hourly_model
from energy_forecast.hourly_shape import (
    predict_future_hourly_recursive,
    train_shape_model,
    build_shape_feature_frame,
    SHAPE_FEATURES,
)
from energy_forecast.hybrid import hybrid_forecast_48h as run_hybrid_48h
from energy_forecast.preprocess import preprocess_hourly
from energy_forecast.profiles import build_profiles

logger = logging.getLogger(__name__)

@dataclass
class RunResult:
    """Serializable result for API / dashboard."""

    meter_id: str
    hourly_rows: int
    data_start: str | None
    data_end: str | None
    hourly_short_term_metrics: dict[str, Any]
    hourly_feature_importance_top15: list[dict[str, Any]]
    profiles_meta: dict[str, Any]
    validation: dict[str, Any]
    version: str
    tuned_threshold: float
    threshold_tuning: dict[str, Any]
    recency_ratio_val: float
    recency_ratio_full: float
    future_daily: list[dict[str, Any]]
    future_hourly: list[dict[str, Any]]
    forecast_48h: list[dict[str, Any]]
    val_daily_table: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, indent=2)


def run_pipeline(settings: Settings | None = None, save_artifacts: bool = True) -> RunResult:
    """Run the full forecast workflow.

    v2.4 flow:
    1) Load/clean/build features from Mongo minute data.
    2) Build hourly profiles and daily aggregates.
    3) Train two-stage daily model with operational-state features.
    4) Tune shutdown threshold on validation (total error + FN penalty).
    5a) Train day-ahead hourly model on DAYAHEAD_HOURLY_FEATURES (for validation MAPE).
    5b) Train shape model for future/hybrid hourly decomposition (rescaled to daily total).
    6) Train full models, run 7-day and 48h forecasts, save artifacts.
    """
    settings = settings or Settings.from_env()
    os.makedirs(settings.model_dir, exist_ok=True)
    logger.info("[v2.4] Starting pipeline for meter=%s", settings.meter_id)

    _gj = gj_holiday_calendar()
    logger.info("[1/8] Loading raw data from MongoDB")
    df_raw = load_raw_dataframe(settings)
    logger.info("Raw rows loaded: %s", f"{len(df_raw):,}")
    logger.info("[2/8] Preprocessing to hourly and deriving consumption_kwh")
    df = preprocess_hourly(df_raw)
    df = build_features(df, _gj)
    df = df.dropna()
    logger.info("Hourly feature rows after dropna: %s", f"{len(df):,}")

    logger.info("[3/8] Building DOW/day-type hourly profiles")
    dow_profiles, daytype_profiles, prof_meta = build_profiles(df, _gj)
    daily_agg = build_daily_agg(df)
    logger.info("Daily rows: %d | Active ratio: %.2f%%", len(daily_agg), daily_agg["is_active"].mean() * 100)

    val_start = daily_agg["ds"].max() - timedelta(days=settings.val_days - 1)
    train_daily = daily_agg[daily_agg["ds"] < val_start].copy()
    val_daily = daily_agg[daily_agg["ds"] >= val_start].copy()
    logger.info("[4/8] Train/val split: train=%d val=%d", len(train_daily), len(val_daily))

    # v2.2 uses stronger operational-state features for daily classifier/regressor.
    from energy_forecast.constants import DAILY_FEATURES_V22

    logger.info("[5/8] Training two-stage daily model (v2.2 features)")
    xgb_clf, xgb_reg, spw = train_two_stage(train_daily, daily_features=DAILY_FEATURES_V22)
    train_active = train_daily[train_daily["is_active"] == 1]
    rec_val = recency_ratio(train_daily, train_active)
    # First pass with default threshold, then tune threshold for better shutdown recall.
    val_daily = predict_daily_two_stage(
        val_daily,
        xgb_clf,
        xgb_reg,
        rec_val,
        threshold=PROB_THRESH,
        daily_features=DAILY_FEATURES_V22,
    )
    # v2.2 objective: lower monthly total error while penalizing shutdown misses.
    tuned_thr, thr_diag = tune_threshold_total_error(val_daily, rec_val, lambda_fn=0.25)
    logger.info(
        "Threshold tuning complete: thr=%.3f | objective=%.3f | total_err=%.3f%% | fn=%s",
        tuned_thr,
        thr_diag.get("objective", 0.0),
        thr_diag.get("val_total_error_pct", 0.0),
        thr_diag.get("fn", 0),
    )
    val_daily = predict_daily_two_stage(
        val_daily,
        xgb_clf,
        xgb_reg,
        rec_val,
        threshold=tuned_thr,
        daily_features=DAILY_FEATURES_V22,
    )
    val_daily = add_day_type_column(val_daily)

    # v2.4: day-ahead hourly model — uses pre-computed features from df directly
    # (same-hour lags cons_lag_24/48/168, rolling stats, calendar, operational state).
    logger.info("[5b/8] Training day-ahead hourly model (DAYAHEAD_HOURLY_FEATURES)")
    val_start_ts = pd.Timestamp(val_start)
    da_train = df[df["rtc_timestamp"] < val_start_ts].copy()
    da_val = df[df["rtc_timestamp"] >= val_start_ts].copy()
    mape_weights = 1.0 / np.clip(da_train["consumption_kwh"].values, 1.0, None)
    xgb_da_val = train_hourly_model(
        da_train[DAYAHEAD_HOURLY_FEATURES],
        da_train["consumption_kwh"],
        sample_weight=mape_weights,
    )
    da_val_pred = np.clip(
        xgb_da_val.predict(da_val[DAYAHEAD_HOURLY_FEATURES].fillna(0)), 0.0, None
    )
    val_hourly_fc = pd.DataFrame({
        "rtc_timestamp": da_val["rtc_timestamp"].values,
        "forecast_kwh": da_val_pred,
    })
    logger.info("Day-ahead val model trained on %d rows, predicting %d val hours",
                len(da_train), len(da_val))

    # Full shape model for forecast output (used by hybrid + future hourly).
    shape_full: XGBRegressor | None = None
    try:
        shape_frame = build_shape_feature_frame(df, dow_profiles, daytype_profiles, _gj)
        full_mask = shape_frame["is_active_day"] == 1
        x_shp_full = shape_frame.loc[full_mask, SHAPE_FEATURES]
        y_shp_full = shape_frame.loc[full_mask, "consumption_kwh"]
        shape_full = train_shape_model(x_shp_full, y_shp_full)
        logger.info("Shape model (full) trained on %d active-day rows", len(x_shp_full))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Shape model training failed, falling back to profiles: %s", exc)

    val_metrics = validation_metrics(val_daily, df, val_hourly_fc)
    logger.info(
        "Validation totals: actual=%.2f pred=%.2f err=%.2f%%",
        val_metrics["val_actual_total_kwh"],
        val_metrics["val_pred_total_kwh"],
        val_metrics["val_total_error_pct"],
    )

    logger.info("[6/8] Training full-data models")
    all_active = daily_agg[daily_agg["is_active"] == 1]
    xgb_clf_full = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    xgb_clf_full.fit(daily_agg[DAILY_FEATURES_V22], daily_agg["is_active"], verbose=False)

    xgb_reg_full = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
    )
    xgb_reg_full.fit(all_active[DAILY_FEATURES_V22], all_active["y"], verbose=False)

    recent_full = daily_agg[daily_agg["is_active"] == 1].tail(RECENT_WINDOW)
    rec_full = (
        float(np.clip(recent_full["y"].mean() / all_active["y"].mean(), 0.5, 1.5))
        if len(all_active) and len(recent_full)
        else 1.0
    )
    logger.info("Recency ratios: val=%.3f full=%.3f", rec_val, rec_full)

    logger.info("[7/8] Generating future 7-day forecast")
    last_date = daily_agg["ds"].max()
    future_base_df = build_future_daily_rows(last_date, settings.forecast_days, _gj)
    # v2.1 recursive future feature fill and prediction loop.
    y_hist = daily_agg["y"].tolist()
    active_hist = daily_agg["is_active"].tolist()
    zero_ratio_hist = daily_agg["daily_zero_hour_ratio"].tolist()
    y_med = float(np.median(y_hist)) if y_hist else 0.0
    pred_rows: list[dict[str, Any]] = []
    for _, base_row in future_base_df.iterrows():
        sh_hist = [1 - a for a in active_hist]
        feat_row = base_row.to_dict()
        feat_row["y_lag_1"] = y_hist[-1] if len(y_hist) >= 1 else y_med
        feat_row["y_lag_2"] = y_hist[-2] if len(y_hist) >= 2 else y_med
        feat_row["y_lag_7"] = y_hist[-7] if len(y_hist) >= 7 else y_med
        feat_row["y_roll_mean_3"] = float(np.mean(y_hist[-3:])) if len(y_hist) >= 3 else y_med
        feat_row["y_roll_mean_7"] = float(np.mean(y_hist[-7:])) if len(y_hist) >= 7 else y_med
        feat_row["y_roll_mean_14"] = float(np.mean(y_hist[-14:])) if len(y_hist) >= 14 else y_med
        feat_row["recent_shutdown_ratio_7"] = float(np.mean(sh_hist[-7:])) if len(sh_hist) >= 7 else 0.0
        feat_row["recent_shutdown_ratio_14"] = float(np.mean(sh_hist[-14:])) if len(sh_hist) >= 14 else 0.0
        feat_row["daily_zero_hour_ratio"] = zero_ratio_hist[-1] if len(zero_ratio_hist) else 0.0
        feat_row["last_24h_mean_cons"] = feat_row["y_lag_1"] / 24.0
        feat_row["last_48h_mean_cons"] = (
            float(np.mean([feat_row["y_lag_1"], feat_row["y_lag_2"]])) / 24.0
        )
        feat_row["last_72h_zero_hour_ratio"] = (
            float(np.mean(zero_ratio_hist[-3:])) if len(zero_ratio_hist) >= 3 else 0.0
        )
        streak = 0
        for a in reversed(active_hist):
            if a == 0:
                streak += 1
            else:
                break
        feat_row["zero_streak_prev"] = streak

        row_df = pd.DataFrame([feat_row])
        prob = float(xgb_clf_full.predict_proba(row_df[DAILY_FEATURES_V22])[:, 1][0])
        reg = float(xgb_reg_full.predict(row_df[DAILY_FEATURES_V22]).clip(min=0)[0])
        clf_active = int(prob >= tuned_thr)
        pred = (reg * rec_full) if clf_active == 1 else 0.0

        out = feat_row.copy()
        out["clf_prob"] = prob
        out["clf_active"] = clf_active
        out["pred"] = float(pred)
        pred_rows.append(out)

        # Update history for next future day recursive state.
        y_hist.append(float(pred))
        active_hist.append(int(clf_active))
        pseudo_zero_ratio = float(np.clip(1.0 - (pred / max(1.0, 2 * 50.0)), 0.0, 1.0))
        zero_ratio_hist.append(pseudo_zero_ratio)
    future_daily_df = pd.DataFrame(pred_rows)
    future_daily_df = add_day_type_column(future_daily_df)
    if shape_full is not None:
        future_hourly_df = predict_future_hourly_recursive(
            shape_full, df, future_daily_df, dow_profiles, daytype_profiles, _gj
        )
    else:
        future_hourly_df = decompose_daily_to_hourly(
            future_daily_df, dow_profiles, daytype_profiles
        )
    logger.info(
        "7-day forecast generated: daily_rows=%d hourly_rows=%d total_kwh=%.2f",
        len(future_daily_df),
        len(future_hourly_df),
        float(future_daily_df["pred"].sum()) if len(future_daily_df) else 0.0,
    )

    test_days = 7
    test_start = df["rtc_timestamp"].max() - timedelta(days=test_days)
    train_h = df[df["rtc_timestamp"] < test_start]
    test_h = df[df["rtc_timestamp"] >= test_start]
    xgb_hourly_eval = train_hourly_model(
        train_h[HOURLY_FEATURES],
        train_h["consumption_kwh"],
        test_h[HOURLY_FEATURES],
        test_h["consumption_kwh"],
    )
    h_metrics = hourly_test_metrics(xgb_hourly_eval, test_h[HOURLY_FEATURES], test_h["consumption_kwh"])

    xgb_hourly_full = train_hourly_model(df[HOURLY_FEATURES], df["consumption_kwh"])
    logger.info("[8/8] Generating hybrid 48h forecast")
    fc_48h = run_hybrid_48h(
        df,
        xgb_hourly_full,
        xgb_clf_full,
        xgb_reg_full,
        dow_profiles,
        daytype_profiles,
        rec_full,
        _gj,
        shape_model=shape_full,
    )
    logger.info("Hybrid 48h rows: %d", len(fc_48h))

    imp = (
        pd.Series(xgb_hourly_eval.feature_importances_, index=HOURLY_FEATURES)
        .sort_values(ascending=False)
        .head(15)
    )
    imp_list = [{"feature": k, "importance": float(v)} for k, v in imp.items()]

    if save_artifacts:
        logger.info("Saving artifacts to %s", settings.model_dir)
        joblib.dump(xgb_clf_full, os.path.join(settings.model_dir, "xgb_clf.pkl"))
        joblib.dump(xgb_reg_full, os.path.join(settings.model_dir, "xgb_reg.pkl"))
        joblib.dump(xgb_hourly_full, os.path.join(settings.model_dir, "xgb_hourly.pkl"))
        joblib.dump(xgb_da_val, os.path.join(settings.model_dir, "xgb_dayahead.pkl"))
        if shape_full is not None:
            joblib.dump(shape_full, os.path.join(settings.model_dir, "xgb_shape.pkl"))
        dow_profiles.to_csv(os.path.join(settings.model_dir, "dow_profiles.csv"), index=True)
        daytype_profiles.to_csv(os.path.join(settings.model_dir, "daytype_profiles.csv"), index=True)
        future_daily_df.to_csv(os.path.join(settings.model_dir, "future_daily.csv"), index=False)
        future_hourly_df.to_csv(os.path.join(settings.model_dir, "future_hourly.csv"), index=False)
        fc_48h.to_csv(os.path.join(settings.model_dir, "forecast_48h.csv"), index=False)
        logger.info("Artifacts saved")

    def _rows(frame: pd.DataFrame) -> list[dict]:
        return json.loads(frame.to_json(orient="records", date_format="iso"))

    val_table = val_daily[["ds", "y", "pred", "clf_active", "clf_prob"]].copy()
    val_table["ds"] = val_table["ds"].dt.strftime("%Y-%m-%d")

    return RunResult(
        meter_id=settings.meter_id,
        hourly_rows=len(df),
        data_start=str(df["rtc_timestamp"].min()) if len(df) else None,
        data_end=str(df["rtc_timestamp"].max()) if len(df) else None,
        hourly_short_term_metrics=h_metrics,
        hourly_feature_importance_top15=imp_list,
        profiles_meta=prof_meta,
        validation=val_metrics,
        version="2.4",
        tuned_threshold=float(tuned_thr),
        threshold_tuning=thr_diag,
        recency_ratio_val=float(rec_val),
        recency_ratio_full=float(rec_full),
        future_daily=_rows(future_daily_df),
        future_hourly=_rows(future_hourly_df),
        forecast_48h=_rows(fc_48h),
        val_daily_table=json.loads(val_table.to_json(orient="records")),
    )
