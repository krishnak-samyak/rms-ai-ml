"""Settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mongo_uri: str
    db_name: str
    collection: str
    meter_id: str
    model_dir: str
    val_days: int = 31
    forecast_days: int = 7

    @classmethod
    def from_env(cls) -> "Settings":
        default_dir = os.path.join(os.getcwd(), "models", "energy_forecast")
        return cls(
            mongo_uri=os.environ.get("MONGO_URI", "mongodb://rmsv1:TX9wHA5X9g@10.100.111.5:27019/IOTDeviceMonitor?authSource=admin"),
            db_name=os.environ.get("MONGO_DB", "IOTDeviceMonitor"),
            collection=os.environ.get("MONGO_COLLECTION", "FUTU00_DataMonitor"),
            meter_id=os.environ.get("METER_ID", "FUTU0000000004000002"),
            model_dir=os.environ.get("ENERGY_MODEL_DIR", default_dir),
        )
