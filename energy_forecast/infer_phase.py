"""Inference phase: 7-day future daily/hourly + hybrid 48h from trained models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from energy_forecast.config import Settings
from energy_forecast.constants import DAILY_FEATURES_V22
from energy_forecast.daily import add_day_type_column, build_future_daily_rows
# from energy_forecast.decompose import decompose_daily_to_hourly
# from energy_forecast.hybrid import hybrid_forecast_48h as run_hybrid_48h
# from energy_forecast.hourly_shape import predict_future_hourly_recursive
from energy_forecast.train_phase import TrainPhaseResult

logger = logging.getLogger(__name__)


@dataclass
class InferPhaseResult:
    future_daily_df: pd.DataFrame
    # future_hourly_df: pd.DataFrame
    # fc_48h: pd.DataFrame


def run_infer_phase(settings: Settings, tr: TrainPhaseResult) -> InferPhaseResult:
    # df = tr.df
    daily_agg = tr.daily_agg
    _ae = tr._ae
    tuned_thr = tr.tuned_threshold
    rec_full = tr.recency_ratio_full
    xgb_clf_full = tr.xgb_clf_full
    xgb_reg_full = tr.xgb_reg_full
    # xgb_hourly_full = tr.xgb_hourly_full
    # shape_full = tr.shape_full
    # dow_profiles = tr.dow_profiles
    # daytype_profiles = tr.daytype_profiles

    logger.info("[infer] Generating future %d-day forecast", settings.forecast_days)
    last_date = daily_agg["ds"].max()
    future_base_df = build_future_daily_rows(last_date, settings.forecast_days, _ae)

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

        y_hist.append(float(pred))
        active_hist.append(int(clf_active))
        pseudo_zero_ratio = float(np.clip(1.0 - (pred / max(1.0, 2 * 50.0)), 0.0, 1.0))
        zero_ratio_hist.append(pseudo_zero_ratio)

    future_daily_df = pd.DataFrame(pred_rows)
    future_daily_df = add_day_type_column(future_daily_df)
    # if shape_full is not None:
    #     future_hourly_df = predict_future_hourly_recursive(
    #         shape_full, df, future_daily_df, dow_profiles, daytype_profiles, _gj
    #     )
    # else:
    #     future_hourly_df = decompose_daily_to_hourly(
    #         future_daily_df, dow_profiles, daytype_profiles
    #     )
    # logger.info(
    #     "[infer] 7-day: daily_rows=%d hourly_rows=%d total_kwh=%.2f",
    #     len(future_daily_df),
    #     len(future_hourly_df),
    #     float(future_daily_df["pred"].sum()) if len(future_daily_df) else 0.0,
    # )

    # logger.info("[infer] Hybrid 48h forecast")
    # fc_48h = run_hybrid_48h(
    #     df,
    #     xgb_hourly_full,
    #     xgb_clf_full,
    #     xgb_reg_full,
    #     dow_profiles,
    #     daytype_profiles,
    #     rec_full,
    #     _gj,
    #     shape_model=shape_full,
    # )
    # logger.info("[infer] Hybrid 48h rows: %d", len(fc_48h))

    return InferPhaseResult(
        future_daily_df=future_daily_df,
        # future_hourly_df=future_hourly_df,
        # fc_48h=fc_48h,
    )
