"""End-to-end pipeline: orchestrates train phase + infer phase (v2.4)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from energy_forecast.config import Settings
from energy_forecast.infer_phase import run_infer_phase
from energy_forecast.model_registry import persist_training_artifacts
from energy_forecast.train_phase import run_train_phase

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
    future_hourly: list[dict[str, Any]] = field(default_factory=list)
    forecast_48h: list[dict[str, Any]] = field(default_factory=list)
    val_daily_table: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, indent=2)


def run_pipeline(settings: Settings | None = None, save_artifacts: bool = True) -> RunResult:
    """Run the full forecast workflow.

    Step 1 split:
    - ``run_train_phase``: load → features → train all models → validation metrics.
    - ``run_infer_phase``: 7-day future + hybrid 48h.

    ``/api/run`` still calls this end-to-end; behavior matches pre-split v2.4
    (future block does not depend on ``xgb_hourly_full``, so ordering vs. legacy
    is equivalent).
    """
    settings = settings or Settings.from_env()
    os.makedirs(settings.model_dir, exist_ok=True)
    logger.info("[v2.4] Starting pipeline for meter=%s", settings.meter_id)

    tr = run_train_phase(settings)
    inf = run_infer_phase(settings, tr)

    future_daily_df = inf.future_daily_df
    # future_hourly_df = inf.future_hourly_df
    # fc_48h = inf.fc_48h

    if save_artifacts:
        meta_saved = persist_training_artifacts(tr, settings, inf=inf)
        logger.info(
            "Artifacts saved (version %s under store %s)",
            meta_saved.get("artifact_version"),
            settings.model_dir,
        )

    def _rows(frame: pd.DataFrame) -> list[dict]:
        return json.loads(frame.to_json(orient="records", date_format="iso"))

    val_table = tr.val_daily[["ds", "y", "pred", "clf_active", "clf_prob"]].copy()
    val_table["ds"] = val_table["ds"].dt.strftime("%Y-%m-%d")

    df = tr.df
    return RunResult(
        meter_id=settings.meter_id,
        hourly_rows=len(df),
        data_start=str(df["rtc_timestamp"].min()) if len(df) else None,
        data_end=str(df["rtc_timestamp"].max()) if len(df) else None,
        hourly_short_term_metrics=tr.hourly_short_term_metrics,
        hourly_feature_importance_top15=tr.hourly_feature_importance_top15,
        profiles_meta=tr.prof_meta,
        validation=tr.val_metrics,
        version="2.4",
        tuned_threshold=tr.tuned_threshold,
        threshold_tuning=tr.threshold_tuning,
        recency_ratio_val=tr.recency_ratio_val,
        recency_ratio_full=tr.recency_ratio_full,
        future_daily=_rows(future_daily_df),
        future_hourly=[],
        forecast_48h=[],
        val_daily_table=json.loads(val_table.to_json(orient="records")),
    )
