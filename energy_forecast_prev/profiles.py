"""DOW (7) and day-type (3) hourly fraction profiles from active days only."""

from __future__ import annotations

import holidays
import pandas as pd

from energy_forecast.constants import DOW_NAMES, MIN_DAYS_FOR_DOW, SHUTDOWN_THRESH


def build_profiles(
    df_hourly: pd.DataFrame,
    _gj_holidays: holidays.HolidayBase,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    hourly_full = df_hourly.copy()
    hourly_full["date"] = hourly_full["rtc_timestamp"].dt.date
    hourly_full["hour"] = hourly_full["rtc_timestamp"].dt.hour

    daily_totals = hourly_full.groupby("date")["consumption_kwh"].sum().reset_index()
    daily_totals.columns = ["date", "daily_total"]
    active_dates = set(daily_totals.loc[daily_totals["daily_total"] >= SHUTDOWN_THRESH, "date"])

    hourly_active = hourly_full[hourly_full["date"].isin(active_dates)].copy()
    hourly_active = hourly_active.merge(daily_totals, on="date")
    hourly_active["frac"] = hourly_active["consumption_kwh"] / hourly_active["daily_total"].clip(lower=1)

    hourly_active["day_type"] = hourly_active["rtc_timestamp"].apply(
        lambda t: "holiday"
        if t.date() in _gj_holidays
        else ("weekend" if t.dayofweek >= 5 else "working")
    )
    daytype_profiles = hourly_active.groupby(["day_type", "hour"])["frac"].mean().unstack(level=0).fillna(0)
    for col in daytype_profiles.columns:
        s = daytype_profiles[col].sum()
        if s > 0:
            daytype_profiles[col] /= s

    hourly_active["dow_val"] = hourly_active["rtc_timestamp"].dt.dayofweek
    dow_profiles = pd.DataFrame(index=range(24))
    dow_day_counts = hourly_active.groupby("dow_val")["date"].nunique()

    for d in range(7):
        n_days = dow_day_counts.get(d, 0)
        if n_days >= MIN_DAYS_FOR_DOW:
            profile = hourly_active[hourly_active["dow_val"] == d].groupby("hour")["frac"].mean()
            profile = profile.reindex(range(24), fill_value=0)
        else:
            if d >= 5:
                profile = daytype_profiles.get("weekend", daytype_profiles.iloc[:, 0])
            else:
                profile = daytype_profiles.get("working", daytype_profiles.iloc[:, 0])
        s = float(profile.sum())
        if s > 0:
            profile = profile / s
        dow_profiles[d] = profile.values

    meta = {
        "dow_day_counts": {DOW_NAMES[i]: int(dow_day_counts.get(i, 0)) for i in range(7)},
    }
    return dow_profiles, daytype_profiles, meta
