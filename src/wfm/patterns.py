# src/wfm/patterns.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import IO, List, Literal, Optional, TypeAlias, Union, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


PatternType: TypeAlias = Literal[
    "uniform",
    "ramp_up_peak_ramp_down",
    "gaussian_peak",
    "custom_weights",
    "profile_csv",
]


@dataclass(frozen=True)
class HoursOfOperation:
    """
    Defines open intervals for a day.
    start_time and end_time are "HH:MM" strings.
    end_time is end-exclusive.

    Supports overnight windows:
      start_time="22:00", end_time="06:00"
      => open from 22:00..24:00 and 00:00..06:00.
    """
    start_time: str
    end_time: str
    interval_minutes: int


PROFILE_REQUIRED_COLUMNS = ["time", "weight"]  # time as HH:MM, weight numeric


# -----------------------------
# Helpers
# -----------------------------
def _time_to_minutes(t: str) -> int:
    hh, mm = t.split(":")
    return int(hh) * 60 + int(mm)


def _ensure_datetime_series(df: pd.DataFrame, col: str) -> pd.Series[pd.Timestamp]:
    """
    Returns df[col] coerced to datetime64[ns] as a *typed* Series[pd.Timestamp].

    This is the key change: returning a typed series makes Pylance recognize
    s.dt.hour / s.dt.minute / s.dt.strftime.
    """
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

    s = pd.to_datetime(df[col], errors="coerce")
    if s.isna().any():
        bad_rows = df.index[s.isna()].tolist()[:10]
        raise ValueError(f"Column '{col}' has invalid datetime values. Example bad rows: {bad_rows}")

    # runtime check + type narrowing hook
    assert is_datetime64_any_dtype(s), f"Column '{col}' is not datetime64 after conversion"

    # Pylance-friendly cast
    return cast(pd.Series[pd.Timestamp], s)


def _normalize_day_df(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy with interval_start normalized to datetime64[ns]."""
    out = df.copy()
    out["interval_start"] = _ensure_datetime_series(out, "interval_start")
    return out


# -----------------------------
# Day grid + Hours of Operation
# -----------------------------
def build_day_intervals(date_ts: pd.Timestamp, interval_minutes: int) -> pd.DataFrame:
    """
    Returns a full-day interval grid:
      00:00 inclusive to 24:00 exclusive.
    """
    if interval_minutes <= 0 or (1440 % interval_minutes) != 0:
        raise ValueError("interval_minutes must be > 0 and divide 1440 evenly (e.g., 5, 10, 15, 30, 60).")

    start = pd.Timestamp(date_ts.date())
    times = pd.date_range(start=start, periods=int(1440 / interval_minutes), freq=f"{interval_minutes}min")
    return pd.DataFrame({"interval_start": times, "interval_minutes": int(interval_minutes)})


def apply_hours_of_operation(df: pd.DataFrame, hoo: HoursOfOperation) -> pd.DataFrame:
    """
    Adds/overwrites is_open based on hours-of-operation.
    Handles overnight windows (e.g., 22:00 -> 06:00).
    """
    out = _normalize_day_df(df)

    start_m = _time_to_minutes(hoo.start_time)
    end_m = _time_to_minutes(hoo.end_time)

    s = cast(pd.Series[pd.Timestamp], out["interval_start"])
    mins = s.dt.hour * 60 + s.dt.minute

    if start_m < end_m:
        out["is_open"] = (mins >= start_m) & (mins < end_m)
    else:
        out["is_open"] = (mins >= start_m) | (mins < end_m)

    return out


# -----------------------------
# Profile CSV tooling
# -----------------------------
def intraday_time_grid(interval_minutes: int) -> pd.DataFrame:
    """
    Returns a full-day HH:MM grid for the chosen interval length.
    Example: 15-min => 96 rows from 00:00..23:45
    """
    if interval_minutes <= 0 or (1440 % interval_minutes) != 0:
        raise ValueError("interval_minutes must be > 0 and divide 1440 evenly (e.g., 5, 10, 15, 30, 60).")

    times = pd.date_range("2000-01-01 00:00:00", periods=int(1440 / interval_minutes), freq=f"{interval_minutes}min")
    return pd.DataFrame({"time": times.strftime("%H:%M")})


def build_profile_template(interval_minutes: int, default_weight: float = 1.0) -> pd.DataFrame:
    """Builds a template profile DataFrame: time, weight."""
    if default_weight < 0:
        raise ValueError("default_weight must be >= 0")

    grid = intraday_time_grid(interval_minutes)
    grid["weight"] = float(default_weight)
    return grid


def read_intraday_profile_csv(file: Union[str, os.PathLike, IO[bytes], IO[str]]) -> pd.DataFrame:
    """
    Reads an intraday profile CSV with columns:
      - time: "HH:MM"
      - weight: numeric (nonnegative)
    """
    df = pd.read_csv(file)

    missing = [c for c in PROFILE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Profile CSV missing columns: {missing}. Required: {PROFILE_REQUIRED_COLUMNS}")

    out = df.copy()
    out["time"] = out["time"].astype(str).str.strip()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")

    if out["weight"].isna().any():
        bad = out[out["weight"].isna()]
        raise ValueError(f"Profile CSV has non-numeric weights in rows: {bad.index.tolist()}")

    if (out["weight"] < 0).any():
        raise ValueError("Profile CSV weights must be nonnegative")

    if (~out["time"].str.match(r"^\d{2}:\d{2}$")).any():
        raise ValueError("Profile CSV 'time' values must be in HH:MM format (e.g., 13:30)")

    # Validate times are within 00:00..23:59
    for t in out["time"].tolist():
        hh, mm = t.split(":")
        ih, im = int(hh), int(mm)
        if not (0 <= ih <= 23 and 0 <= im <= 59):
            raise ValueError(f"Profile CSV has invalid time value: {t}")

    if out["time"].duplicated().any():
        dupes = out.loc[out["time"].duplicated(), "time"].tolist()
        raise ValueError(f"Profile CSV has duplicate time rows: {dupes}")

    return out.sort_values("time").reset_index(drop=True)


def weights_from_intraday_profile(df_day: pd.DataFrame, profile_df: pd.DataFrame) -> np.ndarray:
    """
    Maps profile weights onto df_day intervals by HH:MM.
    Missing times default to 0.
    Closed intervals remain 0 if df_day includes is_open.
    """
    df_day = _normalize_day_df(df_day)
    s = cast(pd.Series[pd.Timestamp], df_day["interval_start"])

    out_w = np.zeros(len(df_day), dtype=float)
    prof_map = dict(zip(profile_df["time"].tolist(), profile_df["weight"].tolist()))
    times = s.dt.strftime("%H:%M").tolist()

    for i, t in enumerate(times):
        out_w[i] = float(prof_map.get(t, 0.0))

    if "is_open" in df_day.columns:
        open_mask = df_day["is_open"].astype(bool).to_numpy()
        out_w[~open_mask] = 0.0

    return np.maximum(out_w, 0.0)


def validate_profile_alignment(df_day: pd.DataFrame, profile_df: pd.DataFrame) -> None:
    """
    Ensures profile aligns to the chosen interval grid.
    - Must have at least one matching time.
    - Enforced: profile must include all open interval times (prevents silent 0 allocations).
    """
    df_day = _normalize_day_df(df_day)
    s = cast(pd.Series[pd.Timestamp], df_day["interval_start"])

    day_times = set(s.dt.strftime("%H:%M").tolist())
    prof_times = set(profile_df["time"].tolist())

    if len(day_times.intersection(prof_times)) == 0:
        raise ValueError(
            "No matching times found between the selected interval grid and the profile CSV. "
            "Ensure the profile 'time' values (HH:MM) match your chosen interval length."
        )

    if "is_open" in df_day.columns:
        open_mask = df_day["is_open"].astype(bool)
        open_times = set(s.loc[open_mask].dt.strftime("%H:%M").tolist())
        missing_open = sorted(list(open_times.difference(prof_times)))
        if missing_open:
            raise ValueError(
                "Profile CSV is missing times for open intervals. "
                f"Example missing times: {missing_open[:10]} (showing up to 10). "
                "Build/download the correct template for your interval length to ensure full coverage."
            )


# -----------------------------
# Patterns
# -----------------------------
def pattern_weights(
    df_day: pd.DataFrame,
    pattern: PatternType,
    *,
    peak_time: Optional[str] = None,
    ramp_fraction: float = 0.25,
    sigma_fraction: float = 0.18,
    custom_weights: Optional[List[float]] = None,
    profile_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Returns weights aligned to df_day rows (one day of intervals).
    Weights are nonnegative and are set to 0 for closed intervals if is_open exists.
    """
    df_day = _normalize_day_df(df_day)
    s = cast(pd.Series[pd.Timestamp], df_day["interval_start"])

    if "is_open" in df_day.columns:
        open_mask = df_day["is_open"].astype(bool).to_numpy()
    else:
        open_mask = np.ones(len(df_day), dtype=bool)

    n = len(df_day)
    w = np.zeros(n, dtype=float)

    idx_open = np.where(open_mask)[0]
    if len(idx_open) == 0:
        return w

    if pattern == "uniform":
        w[idx_open] = 1.0

    elif pattern == "ramp_up_peak_ramp_down":
        m = len(idx_open)
        r = max(1, int(math.floor(m * float(ramp_fraction))))
        plateau = max(1, m - 2 * r)

        up = np.linspace(0.2, 1.0, r, endpoint=False)
        mid = np.ones(plateau, dtype=float)
        down = np.linspace(1.0, 0.2, r, endpoint=True)
        shape = np.concatenate([up, mid, down])

        if len(shape) < m:
            shape = np.pad(shape, (0, m - len(shape)), constant_values=0.2)
        elif len(shape) > m:
            shape = shape[:m]

        w[idx_open] = shape

    elif pattern == "gaussian_peak":
        mins = s.dt.hour * 60 + s.dt.minute

        if peak_time is None:
            peak_idx = idx_open[len(idx_open) // 2]
        else:
            peak_m = _time_to_minutes(peak_time)
            dist = mins.to_numpy(dtype=float) - float(peak_m)
            dist[~open_mask] = 1e9
            peak_idx = int(np.argmin(np.abs(dist)))

        m = len(idx_open)
        sigma = max(1.0, m * float(sigma_fraction))

        open_positions = np.arange(m, dtype=float)
        peak_pos = float(np.where(idx_open == peak_idx)[0][0])
        shape = np.exp(-0.5 * ((open_positions - peak_pos) / sigma) ** 2)

        w[idx_open] = shape

    elif pattern == "custom_weights":
        if custom_weights is None:
            raise ValueError("custom_weights must be provided for custom_weights pattern")
        if len(custom_weights) != len(idx_open):
            raise ValueError(f"custom_weights length must equal number of open intervals ({len(idx_open)})")
        w[idx_open] = np.array(custom_weights, dtype=float)

    elif pattern == "profile_csv":
        if profile_df is None:
            raise ValueError("profile_df must be provided for profile_csv pattern")
        validate_profile_alignment(df_day, profile_df)
        w = weights_from_intraday_profile(df_day, profile_df)

    else:
        raise ValueError(f"Unsupported pattern: {pattern}")

    return np.maximum(w, 0.0)


def allocate_volume_to_intervals(
    df_day: pd.DataFrame,
    daily_volume: float,
    weights: np.ndarray,
    *,
    volume_rounding: Literal["none", "round"] = "round",
) -> pd.DataFrame:
    """
    Allocates daily_volume into interval volumes using weights.
    If weights sum to 0, volumes are 0.

    volume_rounding="round" uses largest remainder method to preserve totals.
    """
    if daily_volume < 0:
        raise ValueError("daily_volume must be >= 0")
    if len(weights) != len(df_day):
        raise ValueError("weights must align to df_day length")

    out = df_day.copy()
    wsum = float(np.sum(weights))

    if wsum == 0.0 or daily_volume == 0.0:
        out["volume"] = 0.0
        return out

    vols = float(daily_volume) * (weights / wsum)

    if volume_rounding == "round":
        if abs(float(daily_volume) - round(float(daily_volume))) > 1e-9:
            raise ValueError(
                "daily_volume must be an integer when volume_rounding='round'. "
                "Use volume_rounding='none' to keep float allocations."
            )

        flo = np.floor(vols)
        rem = vols - flo
        remaining = int(round(float(daily_volume) - float(flo.sum())))
        order = np.argsort(-rem)

        if remaining > 0:
            flo[order[:remaining]] += 1
        elif remaining < 0:
            order2 = np.argsort(rem)
            for j in order2[: abs(remaining)]:
                if flo[j] > 0:
                    flo[j] -= 1

        vols = flo

    out["volume"] = vols.astype(float)
    return out


__all__ = [
    "PatternType",
    "HoursOfOperation",
    "build_day_intervals",
    "apply_hours_of_operation",
    "intraday_time_grid",
    "build_profile_template",
    "read_intraday_profile_csv",
    "weights_from_intraday_profile",
    "validate_profile_alignment",
    "pattern_weights",
    "allocate_volume_to_intervals",
]