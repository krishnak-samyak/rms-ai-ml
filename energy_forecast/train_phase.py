"""Training phase: load data, features, all model fitting, validation metrics.

Separated from forward forecast (``infer_phase``) for train/infer lifecycle (Step 1).

Legacy ``run_pipeline`` interleaved the 7-day future loop between full daily models
and ``xgb_hourly_full``. The future loop does not use ``xgb_hourly_full``, so this
module trains the full hourly model first; ``infer_phase`` then runs futures + hybrid
with identical math.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from energy_forecast.config import Settings
from energy_forecast.constants import (
    # DAYAHEAD_HOURLY_FEATURES,
    DAILY_FEATURES_V22,
    # HOURLY_FEATURES,
    PROB_THRESH,
    RECENT_WINDOW,
)
from energy_forecast.daily import (
    add_day_type_column,
    apply_daily_postcalibration,
    build_daily_agg,
    predict_daily_two_stage,
    recency_ratio,
    select_daily_calibration,
    train_two_stage,
    tune_threshold_total_error,
    validation_metrics,
)
from energy_forecast.data import load_raw_dataframe, train_raw_rtc_bounds
from energy_forecast.features import build_features, gj_holiday_calendar, dubai_holiday_calendar
# from energy_forecast.hourly import hourly_test_metrics, train_hourly_model
# from energy_forecast.hourly_shape import (
#     SHAPE_FEATURES,
#     build_shape_feature_frame,
#     train_shape_model,
# )
from energy_forecast.preprocess import preprocess_hourly
from energy_forecast.profiles import build_profiles

logger = logging.getLogger(__name__)


@dataclass
class TrainPhaseResult:
    """Artifacts needed for ``infer_phase`` and ``RunResult`` assembly."""

    settings: Settings
    df: pd.DataFrame
    daily_agg: pd.DataFrame
    dow_profiles: pd.DataFrame
    daytype_profiles: pd.DataFrame
    prof_meta: dict[str, Any]
    _ae: Any
    val_start: pd.Timestamp
    val_daily: pd.DataFrame
    tuned_threshold: float
    threshold_tuning: dict[str, Any]
    recency_ratio_val: float
    recency_ratio_full: float
    spw: float
    xgb_clf_full: XGBClassifier
    xgb_reg_full: XGBRegressor
    xgb_hourly_full: Any
    xgb_da_val: Any
    shape_full: Any
    val_metrics: dict[str, Any] = field(default_factory=dict)
    hourly_short_term_metrics: dict[str, Any] = field(default_factory=dict)
    hourly_feature_importance_top15: list[dict[str, Any]] = field(default_factory=list)
    #: Post-calibration on active classifier-positive days: ``affine`` or ``isotonic`` (val holdout).
    daily_calib_mode: str = "none"
    daily_calib_a: float = 1.0
    daily_calib_b: float = 0.0
    daily_calib_iso_x: list[float] = field(default_factory=list)
    daily_calib_iso_y: list[float] = field(default_factory=list)
    daily_calib_enabled: bool = False
    daily_calib: dict[str, Any] = field(default_factory=dict)


def run_train_phase(settings: Settings) -> TrainPhaseResult:
    # _gj = gj_holiday_calendar()
    _ae = dubai_holiday_calendar()
    rtc_gte, rtc_lte = train_raw_rtc_bounds(settings)
    if settings.train_raw_lookback_days < settings.val_days + 30:
        logger.warning(
            "[train] train_raw_lookback_days=%d may be tight vs val_days=%d (+ warmup)",
            settings.train_raw_lookback_days,
            settings.val_days,
        )
    logger.info(
        "[train] Loading raw data from MongoDB (RTC %s .. %s, %d d lookback)",
        rtc_gte,
        rtc_lte,
        settings.train_raw_lookback_days,
    )
    df_raw = load_raw_dataframe(settings, rtc_gte=rtc_gte, rtc_lte=rtc_lte)
    logger.info("[train] Raw rows loaded: %s", f"{len(df_raw):,}")
    df = preprocess_hourly(df_raw)
    df = build_features(df, _ae)
    df = df.dropna()
    logger.info("[train] Hourly feature rows after dropna: %s", f"{len(df):,}")

    dow_profiles, daytype_profiles, prof_meta = build_profiles(df, _ae)
    daily_agg = build_daily_agg(df)
    logger.info(
        "[train] Daily rows: %d | Active ratio: %.2f%%",
        len(daily_agg),
        daily_agg["is_active"].mean() * 100,
    )

    val_start = daily_agg["ds"].max() - timedelta(days=settings.val_days - 1)
    train_daily = daily_agg[daily_agg["ds"] < val_start].copy()
    val_daily = daily_agg[daily_agg["ds"] >= val_start].copy()
    logger.info("[train] Train/val split: train=%d val=%d", len(train_daily), len(val_daily))

    logger.info("[train] Training two-stage daily model (v2.2 features)")
    xgb_clf, xgb_reg, spw = train_two_stage(train_daily, daily_features=DAILY_FEATURES_V22)
    rec_val = recency_ratio(train_daily)
    val_daily = predict_daily_two_stage(
        val_daily,
        xgb_clf,
        xgb_reg,
        rec_val,
        threshold=PROB_THRESH,
        daily_features=DAILY_FEATURES_V22,
    )
    tuned_thr, thr_diag = tune_threshold_total_error(val_daily, rec_val, lambda_bal=0.25)
    logger.info(
        "[train] Threshold tuning: thr=%.3f | objective=%.3f | total_err=%.3f%% | fn=%s",
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

    sel = select_daily_calibration(val_daily)
    cal_on = bool(sel["enabled"])
    cal_mode = str(sel["mode"])
    cal_a = float(sel["affine_a"])
    cal_b = float(sel["affine_b"])
    cal_iso_x = list(sel["iso_x"])
    cal_iso_y = list(sel["iso_y"])
    cal_diag: dict[str, Any] = {
        "chosen_mode": cal_mode,
        "affine": sel["affine_diag"],
        "isotonic": sel["iso_diag"],
    }
    if cal_on:
        mpos = val_daily["clf_active"].astype(int) == 1
        val_daily.loc[mpos, "pred"] = apply_daily_postcalibration(
            val_daily.loc[mpos, "pred"].values,
            mode=cal_mode,
            enabled=True,
            affine_a=cal_a,
            affine_b=cal_b,
            iso_x=cal_iso_x,
            iso_y=cal_iso_y,
        ).astype(np.float32)

    # logger.info("[train] Day-ahead hourly model (DAYAHEAD_HOURLY_FEATURES)")
    # val_start_ts = pd.Timestamp(val_start)
    # da_train = df[df["rtc_timestamp"] < val_start_ts].copy()
    # da_val = df[df["rtc_timestamp"] >= val_start_ts].copy()
    # mape_weights = 1.0 / np.clip(da_train["consumption_kwh"].values, 1.0, None)
    # xgb_da_val = train_hourly_model(
    #     da_train[DAYAHEAD_HOURLY_FEATURES],
    #     da_train["consumption_kwh"],
    #     sample_weight=mape_weights,
    # )
    # da_val_pred = np.clip(
    #     xgb_da_val.predict(da_val[DAYAHEAD_HOURLY_FEATURES].fillna(0)), 0.0, None
    # )
    # val_hourly_fc = pd.DataFrame({
    #     "rtc_timestamp": da_val["rtc_timestamp"].values,
    #     "forecast_kwh": da_val_pred,
    # })
    # logger.info("[train] Day-ahead: train=%d predict=%d hours", len(da_train), len(da_val))

    # shape_full: XGBRegressor | None = None
    # try:
    #     shape_frame = build_shape_feature_frame(df, dow_profiles, daytype_profiles, _gj)
    #     full_mask = shape_frame["is_active_day"] == 1
    #     x_shp_full = shape_frame.loc[full_mask, SHAPE_FEATURES]
    #     y_shp_full = shape_frame.loc[full_mask, "consumption_kwh"]
    #     shape_full = train_shape_model(x_shp_full, y_shp_full)
    #     logger.info("[train] Shape model on %d active-day rows", len(x_shp_full))
    # except Exception as exc:  # noqa: BLE001
    #     logger.warning("[train] Shape model skipped: %s", exc)

    # val_metrics = validation_metrics(val_daily, df, val_hourly_fc)
    val_metrics = validation_metrics(val_daily, df)
    logger.info(
        "[train] Validation totals: actual=%.2f pred=%.2f err=%.2f%% | daily_calib=%s mode=%s",
        val_metrics["val_actual_total_kwh"],
        val_metrics["val_pred_total_kwh"],
        val_metrics["val_total_error_pct"],
        "on" if cal_on else "off",
        cal_mode,
    )

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

    rec_full = recency_ratio(daily_agg)
    logger.info("[train] Recency ratios: val=%.3f full=%.3f", rec_val, rec_full)

    # test_days = 7
    # test_start = df["rtc_timestamp"].max() - timedelta(days=test_days)
    # train_h = df[df["rtc_timestamp"] < test_start]
    # test_h = df[df["rtc_timestamp"] >= test_start]
    # xgb_hourly_eval = train_hourly_model(
    #     train_h[HOURLY_FEATURES],
    #     train_h["consumption_kwh"],
    #     test_h[HOURLY_FEATURES],
    #     test_h["consumption_kwh"],
    # )
    # h_metrics = hourly_test_metrics(xgb_hourly_eval, test_h[HOURLY_FEATURES], test_h["consumption_kwh"])

    # xgb_hourly_full = train_hourly_model(df[HOURLY_FEATURES], df["consumption_kwh"])

    # imp = (
    #     pd.Series(xgb_hourly_eval.feature_importances_, index=HOURLY_FEATURES)
    #     .sort_values(ascending=False)
    #     .head(15)
    # )
    # imp_list = [{"feature": k, "importance": float(v)} for k, v in imp.items()]

    return TrainPhaseResult(
        settings=settings,
        df=df,
        daily_agg=daily_agg,
        dow_profiles=dow_profiles,
        daytype_profiles=daytype_profiles,
        prof_meta=prof_meta,
        _ae=_ae,
        val_start=val_start,
        val_daily=val_daily,
        val_metrics=val_metrics,
        tuned_threshold=float(tuned_thr),
        threshold_tuning=thr_diag,
        recency_ratio_val=float(rec_val),
        recency_ratio_full=float(rec_full),
        spw=spw,
        xgb_clf_full=xgb_clf_full,
        xgb_reg_full=xgb_reg_full,
        xgb_hourly_full=None,
        xgb_da_val=None,
        shape_full=None,
        hourly_short_term_metrics={},
        hourly_feature_importance_top15=[],
        daily_calib_mode=cal_mode,
        daily_calib_a=float(cal_a),
        daily_calib_b=float(cal_b),
        daily_calib_iso_x=cal_iso_x,
        daily_calib_iso_y=cal_iso_y,
        daily_calib_enabled=cal_on,
        daily_calib=cal_diag,
    )
