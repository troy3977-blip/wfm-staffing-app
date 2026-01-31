import pandas as pd
import pytest

from wfm.validation import validate_interval_df


def test_validate_interval_df_missing_columns():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError):
        validate_interval_df(df)


def test_validate_interval_df_ok():
    df = pd.DataFrame(
        {
            "interval_start": ["2024-01-01 00:00:00"],
            "interval_minutes": [15],
            "volume": [10],
            "aht_seconds": [300],
            "is_open": [True],
        }
    )
    validate_interval_df(df)