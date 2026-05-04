"""FastAPI app: dashboard + /api/run returning pipeline JSON."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from energy_forecast.config import Settings
from energy_forecast.pipeline import RunResult, run_pipeline

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


def _result_to_api_dict(result: RunResult) -> dict:
    d = result.__dict__.copy()
    return d


@app.get("/api/config")
def api_config():
    s = Settings.from_env()
    return {
        "meter_id": s.meter_id,
        "mongo_db": s.db_name,
        "mongo_collection": s.collection,
        "model_dir": s.model_dir,
        "val_days": s.val_days,
        "forecast_days": s.forecast_days,
        "mongo_uri_set": bool(os.environ.get("MONGO_URI")),
    }
