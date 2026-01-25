import streamlit as st

st.set_page_config(page_title="WFM Interval Staffing (Erlang-C)", layout="wide")

st.title("WFM Interval Staffing (Erlang-C) — MVP")
st.write(
    """
This app computes **required agents per interval** using an Erlang-C staffing engine.

Included:
- Single Interval Calculator
- Interval Staffing Table (CSV upload or generate from daily total + pattern)
- Shrinkage conversion (on-phone → scheduled)
- Optional occupancy cap
- Hours-of-operation masking + arrival patterns
- Custom intraday profile CSV + profile builder
"""
)

st.info("Use the left sidebar to navigate to the calculator or the interval table.")