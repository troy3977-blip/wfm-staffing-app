from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ["interval_start", "interval_minutes", "volume", "aht_seconds", "is_open"]


def read_interval_csv(file) -> pd.DataFrame:
    """
    Reads the interval-level staffing input CSV.
    Expected columns:
      interval_start (datetime parsable)
      interval_minutes (int)
      volume (float)
      aht_seconds (float)
      is_open (0/1 or true/false)

    Returns a normalized DataFrame.
    """
    df = pd.read_csv(file)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Required: {REQUIRED_COLUMNS}")

    df = df.copy()
    df["interval_start"] = pd.to_datetime(df["interval_start"])
    df["interval_minutes"] = df["interval_minutes"].astype(int)
    df["volume"] = df["volume"].astype(float)
    df["aht_seconds"] = df["aht_seconds"].astype(float)

    # Normalize is_open to bool
    if df["is_open"].dtype == object:
        df["is_open"] = df["is_open"].astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
    else:
        df["is_open"] = df["is_open"].astype(int).astype(bool)

    return df.sort_values("interval_start").reset_index(drop=True)