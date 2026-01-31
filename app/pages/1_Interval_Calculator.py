from __future__ import annotations

from datetime import date
from typing import Any, Optional, cast

import pandas as pd
import streamlit as st

from wfm.patterns import (
    PatternType,
    HoursOfOperation,
    build_day_intervals,
    apply_hours_of_operation,
    build_profile_template,
    read_intraday_profile_csv,
    validate_profile_alignment,
    pattern_weights,
    allocate_volume_to_intervals,
)


def _normalize_date_input(x: Any, fallback: date) -> date:
    """
    Streamlit date_input can behave like:
      - date
      - (date, date) for range
      - () in some edge cases
      - None (depending on value/defaults)
    This returns a guaranteed single `date`.
    """
    if x is None:
        return fallback

    # Range mode: tuple/list
    if isinstance(x, (tuple, list)):
        if len(x) >= 1 and x[0] is not None:
            return cast(date, x[0])
        return fallback

    # Single date
    if isinstance(x, date):
        return x

    return fallback


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Interval Calculator", layout="wide")
st.title("Interval Calculator")
st.caption("Build an interval input table (open/closed + volume allocation + AHT) for staffing & Monte Carlo pages.")


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Day + Grid")

    raw_day = st.date_input("Date", value=date.today())
    day = _normalize_date_input(raw_day, fallback=date.today())

    interval_minutes = st.selectbox("Interval minutes", [5, 10, 15, 30, 60], index=2)

    st.divider()
    st.header("Hours of Operation")
    start_time = st.text_input("Start (HH:MM)", value="09:00")
    end_time = st.text_input("End (HH:MM)", value="17:00")

    st.divider()
    st.header("Daily Inputs")
    daily_volume = st.number_input("Daily Volume", min_value=0, value=1000, step=50)
    aht_seconds = st.number_input("AHT (seconds)", min_value=1, value=300, step=10)

    st.divider()
    st.header("Intraday Pattern")
    pattern_str = st.selectbox(
        "Pattern",
        ["uniform", "ramp_up_peak_ramp_down", "gaussian_peak", "custom_weights", "profile_csv"],
        index=0,
    )
    pattern = cast(PatternType, pattern_str)

    peak_time: Optional[str] = None
    if pattern == "gaussian_peak":
        peak_time = st.text_input("Peak time (HH:MM)", value="13:00")

    profile_df: Optional[pd.DataFrame] = None
    if pattern == "profile_csv":
        st.caption("Upload a CSV with columns: time, weight (time formatted HH:MM).")
        uploaded = st.file_uploader("Profile CSV", type=["csv"])

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Build template"):
                st.session_state["profile_template_df"] = build_profile_template(int(interval_minutes))
        with c2:
            if st.button("Clear template"):
                st.session_state.pop("profile_template_df", None)

        tmpl = st.session_state.get("profile_template_df")
        if isinstance(tmpl, pd.DataFrame):
            st.dataframe(tmpl, height=220, use_container_width=True)

        if uploaded is not None:
            try:
                profile_df = read_intraday_profile_csv(uploaded)
            except Exception as e:
                st.error(f"Profile CSV error: {e}")
                profile_df = None


# -----------------------------
# Build intervals
# -----------------------------
st.subheader("Generated Interval Table")

day_ts = pd.Timestamp(day)

hoo = HoursOfOperation(
    start_time=str(start_time).strip(),
    end_time=str(end_time).strip(),
    interval_minutes=int(interval_minutes),
)

try:
    df = build_day_intervals(day_ts, int(interval_minutes))
    df = apply_hours_of_operation(df, hoo)
    df["aht_seconds"] = float(aht_seconds)

    # Build weights
    if pattern == "profile_csv":
        if profile_df is None:
            st.warning("No profile uploaded. Using uniform weights.")
            weights = pattern_weights(df, pattern="uniform")
        else:
            validate_profile_alignment(df, profile_df)
            weights = pattern_weights(df, pattern="profile_csv", profile_df=profile_df)
    else:
        weights = pattern_weights(df, pattern=pattern, peak_time=peak_time)

    # Allocate daily volume
    df = allocate_volume_to_intervals(df, float(daily_volume), weights, volume_rounding="round")

except Exception as e:
    st.error(f"Could not build intervals: {e}")
    st.stop()

st.dataframe(df, use_container_width=True)

st.session_state["intervals_df"] = df
st.success("Saved to session_state as `intervals_df` (used by Interval Staffing Table + Monte Carlo pages).")

st.download_button(
    "Download interval inputs (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="interval_inputs.csv",
    mime="text/csv",
)
