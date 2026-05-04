"""Split predicted daily totals into hourly kWh using DOW profiles."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def decompose_daily_to_hourly(
    daily_pred_df: pd.DataFrame,
    dow_profiles: pd.DataFrame,
    daytype_profiles: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in daily_pred_df.iterrows():
        d = row["ds"]
        pred_total = row["pred"]
        dow_val = d.dayofweek

        if dow_val in dow_profiles.columns:
            profile = dow_profiles[dow_val].values
        else:
            dt = row.get("day_type", "working")
            if dt in daytype_profiles.columns:
                profile = daytype_profiles[dt].values
            else:
                profile = daytype_profiles.iloc[:, 0].values

        for h in range(24):
            rows.append(
                {
                    "rtc_timestamp": d + timedelta(hours=h),
                    "forecast_kwh": float(pred_total) * float(profile[h]),
                }
            )
    return pd.DataFrame(rows)


def profile_fraction_for_hour(
    d: pd.Timestamp,
    hour: int,
    day_type: str,
    dow_profiles: pd.DataFrame,
    daytype_profiles: pd.DataFrame,
) -> float:
    """Profile weight for one clock hour (identical to decompose_daily_to_hourly)."""
    d = pd.Timestamp(d)
    h = int(hour)
    if h < 0 or h > 23:
        return 0.0
    dow_val = d.dayofweek
    if dow_val in dow_profiles.columns:
        profile = dow_profiles[dow_val].values
    else:
        dt = day_type or "working"
        if dt in daytype_profiles.columns:
            profile = daytype_profiles[dt].values
        else:
            profile = daytype_profiles.iloc[:, 0].values
    p = float(profile[h])
    if np.isnan(p) or p < 0:
        return 0.0
    return p
