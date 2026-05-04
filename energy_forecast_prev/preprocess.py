"""Clean minute-level data and resample to hourly consumption."""

from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess_hourly(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df = df.rename(
        columns={
            "A1": "total_kwh",
            "RTC": "rtc_timestamp",
            "EnergymeterId": "meter_id",
        }
    )
    df = df.sort_values(["meter_id", "rtc_timestamp"])
    df["total_kwh"] = pd.to_numeric(df["total_kwh"], errors="coerce")
    df["rtc_timestamp"] = pd.to_datetime(df["rtc_timestamp"], errors="coerce")
    df = df.dropna(subset=["rtc_timestamp", "total_kwh"])
    df.loc[df["total_kwh"] == 0, "total_kwh"] = np.nan
    df = df.dropna()
    df = (
        df.set_index("rtc_timestamp")
        .resample("1h")["total_kwh"]
        .mean()
        .reset_index()
    )
    df["consumption_kwh"] = df["total_kwh"].diff().clip(lower=0)
    df.loc[df["consumption_kwh"].isna(), "consumption_kwh"] = 0
    return df
