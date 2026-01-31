from __future__ import annotations

import pandas as pd


def read_interval_csv(path_or_file) -> pd.DataFrame:
    """
    Reads an interval CSV expected to include:
      interval_start, interval_minutes, volume, aht_seconds, is_open
    """
    df = pd.read_csv(path_or_file)
    return df