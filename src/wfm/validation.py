from __future__ import annotations

import pandas as pd


def validate_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds basic validation flags to an interval input DataFrame.
    """
    out = df.copy()
    out["flag_volume_negative"] = out["volume"] < 0
    out["flag_aht_nonpositive"] = out["aht_seconds"] <= 0
    out["flag_interval_nonpositive"] = out["interval_minutes"] <= 0
    out["flag_open_with_zero_volume"] = (out["is_open"] == True) & (out["volume"] == 0)
    return out