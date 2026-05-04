"""FastAPI app: dashboard + /api/run returning pipeline JSON."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dataclasses import replace
from typing import Union

from energy_forecast.config import Settings
from energy_forecast.infer_phase import run_infer_phase
from energy_forecast.model_registry import (
    active_model_status_extras,
    artifact_paths_ok,
    build_forecast_api_payload,
    load_metadata,
    load_train_phase_for_inference,
    persist_training_artifacts,
    resolve_artifact_dir,
)
from energy_forecast.pipeline import RunResult, run_pipeline
from energy_forecast.train_phase import run_train_phase

_STATIC = Path(__file__).resolve().parent / "static"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Energy Forecast", version="2.4")

if _STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
def index():
    index_path = _STATIC / "index.html"
    if not index_path.is_file():
        return JSONResponse(
            {"error": "static/index.html missing"},
            status_code=500,
        )
    return FileResponse(index_path)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/run")
def api_run():
    """Run full pipeline (Mongo + train + forecasts). Can take a minute."""
    try:
        logger.info("API /api/run invoked")
        result = run_pipeline(Settings.from_env(), save_artifacts=True)
    except Exception as e:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    logger.info("Pipeline completed successfully")
    return JSONResponse(content=_result_to_api_dict(result))

class TrainRequest(BaseModel):
    train_days: Union[int, str, None] = None

@app.post("/api/train")
def api_train(trainRequest: TrainRequest):
    """Train all models and persist artifacts + metadata (no 7-day / 48h forecast)."""
    try:
        settings = Settings.from_env()
        logger.info("API /api/train invoked")
        if trainRequest.train_days is not None:
            # FULL DATA CASE
            if isinstance(trainRequest.train_days, str) and trainRequest.train_days.lower() == "full":
                settings = replace(settings, train_raw_lookback_days=-1)
            # NUMERIC CASE
            elif isinstance(trainRequest.train_days, int):
                if trainRequest.train_days <= 250:
                    raise HTTPException(400, "Train days must be > 250")
                settings = replace(
                    settings,
                    train_raw_lookback_days=trainRequest.train_days
                )
                
        tr = run_train_phase(settings)
        meta = persist_training_artifacts(tr, settings, inf=None)
        artifact_root = resolve_artifact_dir(settings)
    except Exception as e:
        logger.exception("Train failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    logger.info("Train completed successfully")
    return JSONResponse(
        content={
            "status": "trained",
            "trained_at_utc": meta.get("trained_at_utc"),
            "artifact_version": meta.get("artifact_version"),
            "artifact_dir": str(artifact_root) if artifact_root else "",
            "validation": meta.get("validation"),
            "hourly_short_term_metrics": meta.get("hourly_short_term_metrics", {}),
            "model_dir": settings.model_dir,
            "train_days": settings.train_raw_lookback_days
        }
    )


@app.post("/api/forecast")
def api_forecast():
    """Load saved models, pull latest data from Mongo, run 7-day forecasts only."""
    try:
        settings = Settings.from_env()
        logger.info("API /api/forecast invoked")
        tr = load_train_phase_for_inference(settings)
        meta = load_metadata(settings)
        if meta is None:
            raise HTTPException(
                status_code=503,
                detail="No trained model metadata. Run POST /api/train or POST /api/run first.",
            )
        inf = run_infer_phase(settings, tr)
        payload = build_forecast_api_payload(
            tr,
            meta,
            inf.future_daily_df,
            # inf.future_hourly_df,
            # inf.fc_48h,
        )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("Forecast failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    logger.info("Forecast completed successfully")
    return JSONResponse(content=payload)


@app.get("/api/model-status")
def api_model_status():
    """Return metadata + artifact presence for the active model directory."""
    settings = Settings.from_env()
    meta = load_metadata(settings)
    checks = artifact_paths_ok(settings)
    paths = active_model_status_extras(settings)
    return {
        "model_dir": settings.model_dir,
        **paths,
        "has_metadata": meta is not None,
        "metadata": meta,
        "artifacts": checks,
    }


def _result_to_api_dict(result: RunResult) -> dict:
    d = result.__dict__.copy()
    return d


@app.get("/api/config/{train_days}")
def api_config(train_days: Union[int, str] = None):
    s = Settings.from_env()
    return {
        "meter_id": s.meter_id,
        "mongo_db": s.db_name,
        "model_dir": s.model_dir,
        "val_days": s.val_days,
        "forecast_days": s.forecast_days,
        "train_raw_lookback_days": train_days or s.train_raw_lookback_days,
        "infer_raw_lookback_days": s.infer_raw_lookback_days,
        "mongo_uri_set": bool(os.environ.get("MONGO_URI")),
    }
