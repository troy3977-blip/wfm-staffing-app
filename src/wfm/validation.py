from __future__ import annotations

import pandas as pd


REQUIRED_INTERVAL_COLUMNS = {"interval_start", "interval_minutes", "volume", "aht_seconds", "is_open"}


def validate_interval_df(df: pd.DataFrame) -> None:
    missing = REQUIRED_INTERVAL_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Interval dataframe missing required columns: {sorted(missing)}. "
            f"Expected: {sorted(REQUIRED_INTERVAL_COLUMNS)}"
        )

    if df.empty:
        
        raise ValueError("Interval dataframe is empty")

    # basic type sanity
    if (df["interval_minutes"].astype(int) <= 0).any():
        raise ValueError("interval_minutes must be > 0 for all rows")

    if (pd.to_numeric(df["volume"], errors="coerce").isna()).any():
        raise ValueError("volume must be numeric")

    if (pd.to_numeric(df["aht_seconds"], errors="coerce").isna()).any():
        raise ValueError("aht_seconds must be numeric")

    # ensure interval_start is parseable
    parsed = pd.to_datetime(df["interval_start"], errors="coerce")
    if parsed.isna().any():
        bad = df.index[parsed.isna()].tolist()[:10]
        raise ValueError(f"interval_start has invalid timestamps. Example bad rows: {bad}")