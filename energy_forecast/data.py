"""Load raw meter rows from MongoDB."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from pymongo import ASCENDING, DESCENDING, MongoClient

from energy_forecast.config import Settings


def train_raw_rtc_bounds(settings: Settings) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive RTC window for training loads: ``[start, end]`` in UTC."""
    end = pd.Timestamp.now(tz="UTC")
    if settings.train_raw_lookback_days == -1:
        return None, end
    start = end - pd.Timedelta(days=settings.train_raw_lookback_days)
    return start, end


def infer_raw_rtc_bounds(
    settings: Settings,
    end: pd.Timestamp | datetime | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive RTC window for inference-time Mongo refresh (UTC).

    ``end`` defaults to now() but should be set to the model's ``train_data_end``
    so that a stale model (trained on May data, run in July) still fetches the
    correct lookback window rather than querying for non-existent future rows.
    """
    if end is None:
        end = pd.Timestamp.now(tz="UTC")
    else:
        end = pd.Timestamp(end)
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
    start = end - pd.Timedelta(days=settings.infer_raw_lookback_days)
    return start, end


def load_raw_dataframe(
    settings: Settings,
    *,
    rtc_gte: pd.Timestamp | datetime | None = None,
    rtc_lte: pd.Timestamp | datetime | None = None,
) -> pd.DataFrame:
    """
    Load raw rows for EnergymeterId.

    Modes:
    1. FULL DATA (rtc_gte=None, rtc_lte=None)
       → all rows, sorted by RTC ASC (better for time-series)

    2. BOUNDED WINDOW
       → RTC filtered between gte/lte
    """

    client = MongoClient(settings.mongo_uri)
    collection = client[settings.db_name][settings.collection]

    query: dict[str, Any] = {
        "EnergymeterId": settings.meter_id
    }

    # FULL DATA MODE
    if rtc_gte is None and rtc_lte is None:
        cursor = collection.find(query).sort("RTC", ASCENDING)

    # BOUNDED MODE
    else:
        rng: dict[str, Any] = {}
        if rtc_gte is not None:
            rng["$gte"] = pd.Timestamp(rtc_gte).to_pydatetime()
        if rtc_lte is not None:
            rng["$lte"] = pd.Timestamp(rtc_lte).to_pydatetime()
        query["RTC"] = rng
        cursor = collection.find(query).sort("RTC", ASCENDING)

    raw = list(cursor)
    if not raw:
        raise ValueError(
            f"No raw documents found for meter_id={settings.meter_id!r} "
            f"in window [{rtc_gte}, {rtc_lte}]. "
            "Check that RTC is stored as ISODate (not string) and the lookback window covers your data."
        )

    df = pd.DataFrame(raw)

    # Ensure sorted (extra safety)
    if "RTC" in df.columns:
        df = df.sort_values("RTC").reset_index(drop=True)

    return df
