"""Calendar, cyclical, shift, shutdown, lag and rolling features."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def gj_holiday_calendar(years: range | tuple[int, ...] | None = None) -> holidays.HolidayBase:
    yr = years if years is not None else range(2023, 2028)
    return holidays.India(state="GJ", years=yr)


def dubai_holiday_calendar(years: range | tuple[int, ...] | None = None) -> holidays.HolidayBase:
    yr = years if years is not None else range(2023, 2028)
    return holidays.country_holidays("AE", years=yr)

def build_features(frame: pd.DataFrame, _gj_holidays: holidays.HolidayBase) -> pd.DataFrame:
    """Add features to hourly DataFrame (returns new frame with same index order)."""
    frame = frame.copy()
    ts = frame["rtc_timestamp"]

    frame["is_holiday"] = ts.dt.date.apply(lambda d: 1 if d in _gj_holidays else 0)
    frame["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    frame["is_working_day"] = ((frame["is_weekend"] == 0) & (frame["is_holiday"] == 0)).astype(int)
    # frame["is_working_hour"] = (
    #     (ts.dt.hour >= 9) & (ts.dt.hour <= 18) & (frame["is_working_day"] == 1)
    # ).astype(int)

    # frame["days_to_next_holiday"] = ts.apply(
    #     lambda t: min((h - t.date()).days for h in _gj_holidays if (h - t.date()).days >= 0)
    #     if any((h - t.date()).days >= 0 for h in _gj_holidays)
    #     else 30
    # ).clip(upper=30)

    # frame["hour_of_day"] = ts.dt.hour
    # frame["dow"] = ts.dt.dayofweek
    # frame["month"] = ts.dt.month
    # frame["day_of_month"] = ts.dt.day

    # frame["hour_sin"] = np.sin(2 * np.pi * frame["hour_of_day"] / 24)
    # frame["hour_cos"] = np.cos(2 * np.pi * frame["hour_of_day"] / 24)
    # frame["dow_sin"] = np.sin(2 * np.pi * frame["dow"] / 7)
    # frame["dow_cos"] = np.cos(2 * np.pi * frame["dow"] / 7)
    # frame["mon_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    # frame["mon_cos"] = np.cos(2 * np.pi * frame["month"] / 12)

    # def _shift(h: int) -> int:
    #     if 6 <= h < 14:
    #         return 1
    #     if 14 <= h < 22:
    #         return 2
    #     return 0

    # frame["shift"] = ts.dt.hour.map(_shift)
    # frame["shift_sin"] = np.sin(2 * np.pi * frame["shift"] / 3)
    # frame["shift_cos"] = np.cos(2 * np.pi * frame["shift"] / 3)

    # frame["is_zero"] = (frame["consumption_kwh"] < 0.5).astype(int)
    # _grp = (frame["is_zero"] != frame["is_zero"].shift()).cumsum()
    # frame["hours_since_shutdown"] = frame.groupby(_grp).cumcount()
    # frame.loc[frame["is_zero"] == 1, "hours_since_shutdown"] = 0
    # frame["hours_since_shutdown"] = frame["hours_since_shutdown"].clip(upper=168)
    # frame["just_restarted"] = (
    #     (frame["is_zero"].shift(1) == 1) & (frame["is_zero"] == 0)
    # ).astype(int)

    # for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
    #     frame[f"cons_lag_{lag}"] = frame["consumption_kwh"].shift(lag)
    #     frame[f"kwh_lag_{lag}"] = frame["total_kwh"].shift(lag)

    # for win in [6, 12, 24]:
    #     frame[f"cons_roll_mean_{win}"] = frame["consumption_kwh"].rolling(win).mean()
    #     frame[f"kwh_roll_std_{win}"] = frame["total_kwh"].rolling(win).std()
    # for win in [24, 168]:
    #     frame[f"rolling_max_{win}"] = frame["consumption_kwh"].rolling(win).max()
    #     frame[f"rolling_min_{win}"] = frame["consumption_kwh"].rolling(win).min()
    #     frame[f"rolling_range_{win}"] = frame[f"rolling_max_{win}"] - frame[f"rolling_min_{win}"]

    return frame
