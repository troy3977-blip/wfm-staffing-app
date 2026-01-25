import pandas as pd
from wfm.validation import validate_intervals


def test_validate_intervals_flags_expected_columns():
    df = pd.DataFrame(
        {
            "interval_start": ["2026-01-26 09:00:00", "2026-01-26 09:15:00"],
            "interval_minutes": [15, 15],
            "volume": [10.0, -1.0],
            "aht_seconds": [300.0, 0.0],
            "is_open": [True, True],
        }
    )

    out = validate_intervals(df)

    assert len(out) == 2

    # row 0: ok volume, ok aht, open with volume
    assert not out.loc[0, "flag_volume_negative"]
    assert not out.loc[0, "flag_aht_nonpositive"]
    assert not out.loc[0, "flag_interval_nonpositive"]
    assert not out.loc[0, "flag_open_with_zero_volume"]

    # row 1: negative volume + nonpositive AHT should flag
    assert out.loc[1, "flag_volume_negative"]
    assert out.loc[1, "flag_aht_nonpositive"]