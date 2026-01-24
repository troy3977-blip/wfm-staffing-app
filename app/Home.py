import streamlit as st

st.set_page_config(page_title="WFM Interval Staffing (Erlang-C)", layout="wide")

st.title("WFM Interval Staffing (Erlang-C) — MVP")
st.write(
    """
This app computes **required agents per interval** using an Erlang-C staffing engine.

**MVP scope included:**
- Single Interval Calculator
- Interval Staffing Table (CSV upload → results → download)
- Shrinkage and optional occupancy constraint
- Hours-of-operation (is_open) masking
"""
)

st.info("Use the left sidebar to navigate to the calculator or the interval table.")