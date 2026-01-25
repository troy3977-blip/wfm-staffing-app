from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, List

import numpy as np
import pandas as pd


PatternType = Literal["uniform", "ramp_up_peak_ramp_down", "gaussian_peak", "custom_weights"]


@dataclass(frozen=True)
class HoursOfOperation:
    """
    Defines open intervals for a day.
    Example: open from 09:00 to 17:00 in 15-min intervals.
    """
    start_time: str  # "HH:MM"
    end_time: str    # "HH:MM" (end exclusive)
    interval_minutes: int


PROFILE_REQUIRED_COLUMNS = ["time", "weight"]  # time as HH:MM, weight as numeric


def _time_to_minutes(t: str) -> int:
    hh, mm = t.split(":")
    return int(hh) * 60 + int(mm)


def build_day_intervals(date: pd.Timestamp, interval_minutes: int) -> pd.DataFrame:
    """
    Returns all intervals in the day (00:00 inclusive to 24:00 exclusive).
    """
    if interval_minutes <= 0 or (1440 % interval_minutes) != 0:
        raise ValueError("interval_minutes must be > 0 and divide 1440 evenly (e.g., 5, 10, 15, 30, 60).")

    start = pd.Timestamp(date.date())
    times = pd.date_range(start=start, periods=int(1440 / interval_minutes), freq=f"{interval_minutes}min")
    return pd.DataFrame({"interval_start": times, "interval_minutes": interval_minutes})


def apply_hours_of_operation(df: pd.DataFrame, hoo: HoursOfOperation) -> pd.DataFrame:
    """
    Adds/overwrites is_open based on HOO window for the day.
    """
    out = df.copy()
    start_m = _time_to_minutes(hoo.start_time)
    end_m = _time_to_minutes(hoo.end_time)

    mins = out["interval_start"].dt.hour * 60 + out["interval_start"].dt.minute
    out["is_open"] = (mins >= start_m) & (mins < end_m)
    return out


def pattern_weights(
    df: pd.DataFrame,
    pattern: PatternType,
    peak_time: Optional[str] = None,            # e.g. "13:30"
    ramp_fraction: float = 0.25,                # for ramp_up_peak_ramp_down
    sigma_fraction: float = 0.18,               # for gaussian_peak
    custom_weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Returns weights aligned to df rows (one day of intervals).
    Weights are nonnegative and will be zero for closed intervals if is_open present.
    """
    if "is_open" in df.columns:
        open_mask = df["is_open"].astype(bool).to_numpy()
    else:
        open_mask = np.ones(len(df), dtype=bool)

    n = len(df)
    w = np.zeros(n, dtype=float)

    idx_open = np.where(open_mask)[0]
    if len(idx_open) == 0:
        return w

    if pattern == "uniform":
        w[idx_open] = 1.0

    elif pattern == "ramp_up_peak_ramp_down":
        # Piecewise linear: ramp up -> peak plateau -> ramp down
        m = len(idx_open)
        r = max(1, int(math.floor(m * ramp_fraction)))
        plateau = max(1, m - 2 * r)

        up = np.linspace(0.2, 1.0, r, endpoint=False)
        mid = np.ones(plateau) * 1.0
        down = np.linspace(1.0, 0.2, r, endpoint=True)
        shape = np.concatenate([up, mid, down])

        # Fix length if rounding created mismatch
        if len(shape) < m:
            shape = np.pad(shape, (0, m - len(shape)), constant_values=0.2)
        elif len(shape) > m:
            shape = shape[:m]

        w[idx_open] = shape

    elif pattern == "gaussian_peak":
        if peak_time is None:
            peak_idx = idx_open[len(idx_open) // 2]
        else:
            peak_m = _time_to_minutes(peak_time)
            mins = df["interval_start"].dt.hour * 60 + df["interval_start"].dt.minute
            dist = (mins.to_numpy() - peak_m).astype(float)
            dist[~open_mask] = 1e9
            peak_idx = int(np.argmin(np.abs(dist)))

        m = len(idx_open)
        sigma = max(1.0, m * sigma_fraction)

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

    else:
        raise ValueError(f"Unsupported pattern: {pattern}")

    return np.maximum(w, 0.0)


def allocate_volume_to_intervals(
    df_day: pd.DataFrame,
    daily_volume: float,
    weights: np.ndarray,
    volume_rounding: Literal["none", "round"] = "round",
) -> pd.DataFrame:
    """
    Allocates daily_volume into interval-level volumes using weights.
    If weights sum to 0, volumes are 0.

    volume_rounding="round" uses largest remainder method to preserve the total.
    """
    if daily_volume < 0:
        raise ValueError("daily_volume must be >= 0")
    if len(weights) != len(df_day):
        raise ValueError("weights must align to df_day length")

    out = df_day.copy()
    wsum = float(weights.sum())
    if wsum == 0.0 or daily_volume == 0.0:
        out["volume"] = 0.0
        return out

    vols = daily_volume * (weights / wsum)

    if volume_rounding == "round":
        flo = np.floor(vols)
        rem = vols - flo
        remaining = int(round(daily_volume - flo.sum()))
        order = np.argsort(-rem)  # descending remainder
        if remaining > 0:
            flo[order[:remaining]] += 1
        vols = flo

    out["volume"] = vols.astype(float)
    return out


def read_intraday_profile_csv(file) -> pd.DataFrame:
    """
    Reads an intraday arrival profile CSV with columns:
      - time: "HH:MM"
      - weight: numeric (nonnegative)

    Example:
      time,weight
      09:00,1.0
      09:15,1.1
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
    Maps profile weights onto df_day intervals based on time-of-day HH:MM.
    Missing times default to 0.
    Closed intervals remain 0 (if df_day has is_open).
    """
    if "interval_start" not in df_day.columns:
        raise ValueError("df_day must include 'interval_start'")

    out_w = np.zeros(len(df_day), dtype=float)

    prof_map = dict(zip(profile_df["time"].tolist(), profile_df["weight"].tolist()))
    times = df_day["interval_start"].dt.strftime("%H:%M").tolist()

    for i, t in enumerate(times):
        out_w[i] = float(prof_map.get(t, 0.0))

    if "is_open" in df_day.columns:
        open_mask = df_day["is_open"].astype(bool).to_numpy()
        out_w[~open_mask] = 0.0

    return np.maximum(out_w, 0.0)


def validate_profile_alignment(df_day: pd.DataFrame, profile_df: pd.DataFrame) -> None:
    """
    Ensures the profile aligns to the chosen interval grid.
    - Must have at least one matching time.
    - Enforced: profile must include all open interval times (prevents silent 0 allocations).
    """
    day_times = df_day["interval_start"].dt.strftime("%H:%M").tolist()
    day_set = set(day_times)
    prof_set = set(profile_df["time"].tolist())

    matches = day_set.intersection(prof_set)
    if len(matches) == 0:
        raise ValueError(
            "No matching times found between the selected interval grid and the profile CSV. "
            "Ensure the profile 'time' values (HH:MM) match your chosen interval length."
        )

    if "is_open" in df_day.columns:
        open_times = set(
            df_day.loc[df_day["is_open"].astype(bool), "interval_start"].dt.strftime("%H:%M").tolist()
        )
        missing_open = sorted(list(open_times.difference(prof_set)))
        if len(missing_open) > 0:
            raise ValueError(
                "Profile CSV is missing times for open intervals. "
                f"Example missing times: {missing_open[:10]} (showing up to 10). "
                "Download the template for this interval length to ensure full coverage."
            )


def intraday_time_grid(interval_minutes: int) -> pd.DataFrame:
    """
    Returns a full-day HH:MM time grid for the given interval length.
    Example: 15-min -> 96 rows: 00:00 ... 23:45
    """
    if interval_minutes <= 0 or (1440 % interval_minutes) != 0:
        raise ValueError("interval_minutes must be > 0 and divide 1440 evenly (e.g., 5, 10, 15, 30, 60).")

    times = pd.date_range(
        "2000-01-01 00:00:00",
        periods=int(1440 / interval_minutes),
        freq=f"{interval_minutes}min",
    )
    return pd.DataFrame({"time": times.strftime("%H:%M")})


def build_profile_template(interval_minutes: int, default_weight: float = 1.0) -> pd.DataFrame:
    """
    Builds a template profile DataFrame with all times for the chosen interval length.
    Users can edit weights and re-upload.
    """
    if default_weight < 0:
        raise ValueError("default_weight must be >= 0")

    grid = intraday_time_grid(interval_minutes)
    grid["weight"] = float(default_weight)
    return grid