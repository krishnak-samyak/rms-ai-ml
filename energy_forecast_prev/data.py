"""Load raw meter rows from MongoDB."""

from __future__ import annotations

import pandas as pd
from pymongo import DESCENDING, MongoClient

from energy_forecast.config import Settings


def load_raw_dataframe(settings: Settings) -> pd.DataFrame:
    client = MongoClient(settings.mongo_uri)
    collection = client[settings.db_name][settings.collection]
    query = {"EnergymeterId": settings.meter_id}
    cursor = collection.find(query).sort("_id", DESCENDING)
    raw = list(cursor)
    return pd.DataFrame(raw)
