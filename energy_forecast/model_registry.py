"""Persist and load model artifacts + JSON metadata (versioned layout, Step 5).

Layout (per blueprint):
  ``{model_dir}/{meter_id}/{artifact_version}/`` — joblibs, CSVs, ``model_metadata.json``
  ``{model_dir}/{meter_id}/active_model.json`` — pointer ``{ "artifact_version": "..." }``

Legacy (pre–step 5): ``{model_dir}/model_metadata.json`` at store root is still resolved
for reads so existing deployments keep working until the next train.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from energy_forecast.config import Settings
from energy_forecast.data import infer_raw_rtc_bounds, load_raw_dataframe
from energy_forecast.daily import build_daily_agg
from energy_forecast.features import build_features, gj_holiday_calendar, dubai_holiday_calendar
from energy_forecast.infer_phase import InferPhaseResult
from energy_forecast.preprocess import preprocess_hourly
from energy_forecast.train_phase import TrainPhaseResult

logger = logging.getLogger(__name__)

METADATA_FILE = "model_metadata.json"
ACTIVE_POINTER = "active_model.json"


def _safe_meter_segment(meter_id: str) -> str:
    s = meter_id.replace("/", "_").replace("\\", "_").strip()
    return s or "unknown_meter"


def meter_models_root(settings: Settings) -> Path:
    """``{ENERGY_MODEL_DIR}/{meter_id}/`` — holds the active pointer and version folders."""
    return Path(settings.model_dir) / _safe_meter_segment(settings.meter_id)


def _read_active_pointer(meter_root: Path) -> dict[str, Any] | None:
    p = meter_root / ACTIVE_POINTER
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Corrupt %s, ignoring", p)
        return None


def _new_artifact_version_dir_name(meter_root: Path) -> str:
    """UTC folder name; filesystem-safe (no ``:``). Uniqueness if same-second retrain."""
    base = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    name = base
    while (meter_root / name).exists():
        name = f"{base}-{secrets.token_hex(2)}"
    return name


def resolve_artifact_dir(settings: Settings) -> Path | None:
    """Directory that holds ``model_metadata.json`` + pickles for inference / status.

    Resolution order:
    1. ``{model_dir}/{meter_id}/active_model.json`` → version subdirectory.
    2. Legacy flat store: ``{model_dir}/model_metadata.json``.
    3. Legacy meter-only: ``{model_dir}/{meter_id}/model_metadata.json`` (no pointer).
    """
    store = Path(settings.model_dir)
    mroot = meter_models_root(settings)
    ptr = _read_active_pointer(mroot)
    if ptr:
        ver = ptr.get("artifact_version")
        if isinstance(ver, str) and ver:
            cand = mroot / ver
            if (cand / METADATA_FILE).is_file():
                return cand
            logger.warning(
                "Active pointer targets missing or incomplete version %r under %s; trying legacy paths",
                ver,
                mroot,
            )
    if (store / METADATA_FILE).is_file():
        return store
    if (mroot / METADATA_FILE).is_file():
        return mroot
    return None


def active_model_status_extras(settings: Settings) -> dict[str, str]:
    """Scalar fields for ``GET /api/model-status`` (paths + active version id)."""
    mroot = meter_models_root(settings)
    ptr = _read_active_pointer(mroot) or {}
    root = resolve_artifact_dir(settings)
    return {
        "meter_models_root": str(mroot),
        "active_artifact_version": str(ptr.get("artifact_version") or ""),
        "active_artifact_dir": str(root) if root is not None else "",
    }


def artifact_dir_required(settings: Settings) -> Path:
    root = resolve_artifact_dir(settings)
    if root is None:
        raise FileNotFoundError(
            f"No trained model under {settings.model_dir!r} for meter {settings.meter_id!r}. "
            f"Run POST /api/train or POST /api/run first."
        )
    return root


def persist_training_artifacts(
    tr: TrainPhaseResult,
    settings: Settings,
    inf: InferPhaseResult | None = None,
) -> dict[str, Any]:
    """Save joblibs, profile CSVs, optional forecast CSVs, and metadata under a new version."""
    meter_root = meter_models_root(settings)
    meter_root.mkdir(parents=True, exist_ok=True)
    version = _new_artifact_version_dir_name(meter_root)
    d = meter_root / version
    d.mkdir(parents=False, exist_ok=False)

    joblib.dump(tr.xgb_clf_full, d / "xgb_clf.pkl")
    joblib.dump(tr.xgb_reg_full, d / "xgb_reg.pkl")
    # joblib.dump(tr.xgb_hourly_full, d / "xgb_hourly.pkl")
    # joblib.dump(tr.xgb_da_val, d / "xgb_dayahead.pkl")
    # if tr.shape_full is not None:
    #     joblib.dump(tr.shape_full, d / "xgb_shape.pkl")
    tr.dow_profiles.to_csv(d / "dow_profiles.csv", index=True)
    tr.daytype_profiles.to_csv(d / "daytype_profiles.csv", index=True)

    if inf is not None:
        inf.future_daily_df.to_csv(d / "future_daily.csv", index=False)
        # inf.future_hourly_df.to_csv(d / "future_hourly.csv", index=False)
        # inf.fc_48h.to_csv(d / "forecast_48h.csv", index=False)

    val_table = tr.val_daily[["ds", "y", "pred", "clf_active", "clf_prob"]].copy()
    val_table["ds"] = val_table["ds"].dt.strftime("%Y-%m-%d")
    val_daily_records = json.loads(val_table.to_json(orient="records"))

    meta: dict[str, Any] = {
        "pipeline_version": "2.4",
        "meter_id": settings.meter_id,
        "artifact_version": version,
        "artifact_relative_dir": f"{_safe_meter_segment(settings.meter_id)}/{version}",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_data_start": str(tr.df["rtc_timestamp"].min()) if len(tr.df) else None,
        "train_data_end": str(tr.df["rtc_timestamp"].max()) if len(tr.df) else None,
        "val_days": settings.val_days,
        "forecast_days": settings.forecast_days,
        "train_raw_lookback_days": settings.train_raw_lookback_days,
        "infer_raw_lookback_days": settings.infer_raw_lookback_days,
        "tuned_threshold": tr.tuned_threshold,
        # "threshold_tuning": tr.threshold_tuning,
        "recency_ratio_val": tr.recency_ratio_val,
        "recency_ratio_full": tr.recency_ratio_full,
        "spw": tr.spw,
        "daily_calib_mode": tr.daily_calib_mode,
        "daily_calib_a": tr.daily_calib_a,
        "daily_calib_b": tr.daily_calib_b,
        "daily_calib_iso_x": tr.daily_calib_iso_x,
        "daily_calib_iso_y": tr.daily_calib_iso_y,
        "daily_calib_enabled": tr.daily_calib_enabled,
        "daily_calibration": tr.daily_calib,
        "validation": tr.val_metrics,
        # "hourly_short_term_metrics": tr.hourly_short_term_metrics,
        # "hourly_feature_importance_top15": tr.hourly_feature_importance_top15,
        "profiles_meta": tr.prof_meta,
        "val_daily_table": val_daily_records,
        # "has_shape": tr.shape_full is not None,
    }
    (d / METADATA_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    pointer = {
        "artifact_version": version,
        "activated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (meter_root / ACTIVE_POINTER).write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    logger.info("Wrote model metadata to %s; active → %s", d / METADATA_FILE, meter_root / ACTIVE_POINTER)
    return meta


def load_metadata(settings: Settings) -> dict[str, Any] | None:
    root = resolve_artifact_dir(settings)
    if root is None:
        return None
    p = root / METADATA_FILE
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def artifact_paths_ok(settings: Settings) -> dict[str, bool]:
    """Quick presence check for core files in the resolved artifact directory."""
    root = resolve_artifact_dir(settings)
    if root is None:
        return {}
    files = [
        "xgb_clf.pkl",
        "xgb_reg.pkl",
        # "xgb_hourly.pkl",
        # "xgb_dayahead.pkl",
        "dow_profiles.csv",
        "daytype_profiles.csv",
        METADATA_FILE,
    ]
    # meta = load_metadata(settings)
    # if meta and meta.get("has_shape"):
    #     files.append("xgb_shape.pkl")
    return {f: (root / f).is_file() for f in files}


def load_train_phase_for_inference(settings: Settings) -> TrainPhaseResult:
    """Reload models + profiles from disk and rebuild hourly frame from Mongo (latest data)."""
    meta = load_metadata(settings)
    if meta is None:
        raise FileNotFoundError(
            f"No {METADATA_FILE} for meter {settings.meter_id!r} under {settings.model_dir!r}. "
            "Run POST /api/train or /api/run first."
        )

    root = artifact_dir_required(settings)
    clf = joblib.load(root / "xgb_clf.pkl")
    reg = joblib.load(root / "xgb_reg.pkl")
    # hourly = joblib.load(root / "xgb_hourly.pkl")
    # dayahead = joblib.load(root / "xgb_dayahead.pkl")
    # shape_path = root / "xgb_shape.pkl"
    # shape_full = joblib.load(shape_path) if meta.get("has_shape") and shape_path.is_file() else None

    dow_profiles = pd.read_csv(root / "dow_profiles.csv", index_col=0)
    daytype_profiles = pd.read_csv(root / "daytype_profiles.csv", index_col=0)

    # _gj = gj_holiday_calendar()
    _ae = dubai_holiday_calendar()
    if settings.infer_raw_lookback_days < 21:
        logger.warning(
            "[infer reload] infer_raw_lookback_days=%d may be tight for ~168h lag features",
            settings.infer_raw_lookback_days,
        )
    train_data_end = meta.get("train_data_end")
    ig, il = infer_raw_rtc_bounds(
        settings,
        end=pd.Timestamp(train_data_end) if train_data_end else None,
    )
    logger.info("[infer reload] Raw Mongo RTC window %s .. %s (anchored to train_data_end=%s)", ig, il, train_data_end)
    df_raw = load_raw_dataframe(settings, rtc_gte=ig, rtc_lte=il)
    df = preprocess_hourly(df_raw)
    df = build_features(df, _ae).dropna()
    daily_agg = build_daily_agg(df)

    vdays = int(meta.get("val_days", settings.val_days))
    val_start = daily_agg["ds"].max() - timedelta(days=vdays - 1)
    val_daily = pd.DataFrame(columns=["ds", "y", "pred", "clf_active", "clf_prob"])

    return TrainPhaseResult(
        settings=settings,
        df=df,
        daily_agg=daily_agg,
        dow_profiles=dow_profiles,
        daytype_profiles=daytype_profiles,
        prof_meta=meta.get("profiles_meta", {}),
        _ae=_ae,
        val_start=val_start,
        val_daily=val_daily,
        val_metrics=meta.get("validation", {}),
        tuned_threshold=float(meta["tuned_threshold"]),
        threshold_tuning=meta.get("threshold_tuning", {}),
        recency_ratio_val=float(meta.get("recency_ratio_val", 1.0)),
        recency_ratio_full=float(meta["recency_ratio_full"]),
        spw=float(meta.get("spw", 1.0)),
        xgb_clf_full=clf,
        xgb_reg_full=reg,
        xgb_hourly_full=None,
        xgb_da_val=None,
        shape_full=None,
        hourly_short_term_metrics=meta.get("hourly_short_term_metrics", {}),
        hourly_feature_importance_top15=meta.get("hourly_feature_importance_top15", []),
        daily_calib_mode=str(
            meta.get("daily_calib_mode")
            or ("affine" if meta.get("daily_calib_enabled") else "none")
        ),
        daily_calib_a=float(meta.get("daily_calib_a", 1.0)),
        daily_calib_b=float(meta.get("daily_calib_b", 0.0)),
        daily_calib_iso_x=list(meta.get("daily_calib_iso_x") or []),
        daily_calib_iso_y=list(meta.get("daily_calib_iso_y") or []),
        daily_calib_enabled=bool(meta.get("daily_calib_enabled", False)),
        daily_calib=dict(meta.get("daily_calibration") or {}),
    )


def build_forecast_api_payload(
    tr: TrainPhaseResult,
    meta: dict[str, Any],
    future_daily_df: pd.DataFrame,
    # future_hourly_df: pd.DataFrame,
    # fc_48h: pd.DataFrame,
) -> dict[str, Any]:
    """Shape JSON like ``RunResult`` for the dashboard + inference extras."""
    df = tr.df

    def _rows(frame: pd.DataFrame) -> list[dict]:
        return json.loads(frame.to_json(orient="records", date_format="iso"))

    return {
        "meter_id": tr.settings.meter_id,
        "hourly_rows": len(df),
        "data_start": str(df["rtc_timestamp"].min()) if len(df) else None,
        "data_end": str(df["rtc_timestamp"].max()) if len(df) else None,
        "hourly_short_term_metrics": meta.get("hourly_short_term_metrics", tr.hourly_short_term_metrics),
        "hourly_feature_importance_top15": meta.get(
            "hourly_feature_importance_top15", tr.hourly_feature_importance_top15
        ),
        "profiles_meta": meta.get("profiles_meta", tr.prof_meta),
        "validation": meta.get("validation", tr.val_metrics),
        "version": meta.get("pipeline_version", "2.4"),
        "tuned_threshold": float(meta.get("tuned_threshold", tr.tuned_threshold)),
        "threshold_tuning": meta.get("threshold_tuning", tr.threshold_tuning),
        "recency_ratio_val": float(meta.get("recency_ratio_val", tr.recency_ratio_val)),
        "recency_ratio_full": float(meta.get("recency_ratio_full", tr.recency_ratio_full)),
        "future_daily": _rows(future_daily_df),
        "future_hourly": [],
        "forecast_48h": [],
        "val_daily_table": meta.get("val_daily_table", []),
        "model_metadata": {
            "trained_at_utc": meta.get("trained_at_utc"),
            "train_data_end": meta.get("train_data_end"),
            "inference_data_end": str(df["rtc_timestamp"].max()) if len(df) else None,
            "artifact_version": meta.get("artifact_version"),
            "daily_calib_enabled": meta.get("daily_calib_enabled"),
            "daily_calib_mode": meta.get("daily_calib_mode"),
            "daily_calib_a": meta.get("daily_calib_a"),
            "daily_calib_b": meta.get("daily_calib_b"),
            "daily_calib_iso_knots": len(meta.get("daily_calib_iso_x") or []),
        },
    }
