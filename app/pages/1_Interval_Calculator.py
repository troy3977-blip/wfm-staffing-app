# app/pages/1_Interval_Calculator.py
from __future__ import annotations

from datetime import date
from typing import cast

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

st.set_page_config(page_title="Interval Calculator", layout="wide")
st.title("Interval Calculator")
st.caption(
    "Build a day of intervals, apply Hours of Operation, apply an intraday pattern, "
    "and allocate daily volume into interval-level volume."
)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Day + Grid")

    # IMPORTANT: This returns either date or tuple[date,...] depending on widget usage.
    # We force it to a single date for simplicity and to satisfy type-checkers.
    day_input = st.date_input("Date", value=date.today())
    if isinstance(day_input, (tuple, list)):
        day0 = day_input[0]
    else:
        day0 = day_input

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

    peak_time: str | None = None
    if pattern == "gaussian_peak":
        peak_time = st.text_input("Peak time (HH:MM)", value="13:00")

    profile_df: pd.DataFrame | None = None
    if pattern == "profile_csv":
        st.caption("Upload a CSV with columns: time, weight (HH:MM, numeric).")
        uploaded_profile = st.file_uploader("Upload profile CSV", type=["csv"])

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Build template"):
                st.session_state["profile_template_df"] = build_profile_template(int(interval_minutes))

        with col_b:
            if "profile_template_df" in st.session_state:
                st.download_button(
                    "Download template CSV",
                    data=st.session_state["profile_template_df"].to_csv(index=False).encode("utf-8"),
                    file_name=f"profile_template_{int(interval_minutes)}min.csv",
                    mime="text/csv",
                )

        if "profile_template_df" in st.session_state:
            st.dataframe(st.session_state["profile_template_df"], height=220)

        if uploaded_profile is not None:
            profile_df = read_intraday_profile_csv(uploaded_profile)

# -----------------------------
# Build intervals
# -----------------------------
st.subheader("Interval Table")

# Convert selected day to Timestamp safely
day_ts = pd.Timestamp(day0)

hoo = HoursOfOperation(
    start_time=str(start_time).strip(),
    end_time=str(end_time).strip(),
    interval_minutes=int(interval_minutes),
)

df = build_day_intervals(date=day_ts, interval_minutes=int(interval_minutes))
df = apply_hours_of_operation(df, hoo)
df["aht_seconds"] = float(aht_seconds)

# -----------------------------
# Weights + allocation
# -----------------------------
try:
    if pattern == "profile_csv":
        if profile_df is None:
            st.warning("Upload a profile CSV to use the profile_csv pattern. Falling back to uniform.")
            weights = pattern_weights(df, pattern=cast(PatternType, "uniform"))
        else:
            validate_profile_alignment(df, profile_df)
            weights = pattern_weights(df, pattern=cast(PatternType, "profile_csv"), profile_df=profile_df)
    else:
        weights = pattern_weights(df, pattern=pattern, peak_time=peak_time)

    df = allocate_volume_to_intervals(
        df_day=df,
        daily_volume=float(daily_volume),
        weights=weights,
        volume_rounding="round",
    )

except Exception as e:
    st.error(f"Error building intervals: {e}")
    st.stop()

st.dataframe(df, use_container_width=True)

# Save for other pages
st.session_state["intervals_df"] = df
st.success("Saved to session_state as `intervals_df`")

st.download_button(
    "Download interval inputs (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="interval_inputs.csv",
    mime="text/csv",
)