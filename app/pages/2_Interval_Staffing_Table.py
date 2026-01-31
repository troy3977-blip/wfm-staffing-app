import pandas as pd
import streamlit as st
from typing import cast

from wfm.staffing import StaffingInputs, TargetType, compute_required_agents
from wfm.validation import validate_interval_df

st.set_page_config(page_title="Interval Staffing Table", layout="wide")
st.title("Interval Staffing Table (Erlang-C)")

st.sidebar.header("Target")
target_type = cast(TargetType, st.sidebar.selectbox("Target type", ["service_level", "asa"], index=0))

service_level_target = 0.80
service_level_time_seconds = 60.0
asa_target_seconds = 30.0

if target_type == "service_level":
    service_level_target = st.sidebar.slider("Service Level target", 0.50, 0.99, 0.80, 0.01)
    service_level_time_seconds = st.sidebar.number_input("Service Level seconds", min_value=0.0, value=60.0, step=1.0)
else:
    asa_target_seconds = st.sidebar.number_input("ASA target (seconds)", min_value=1.0, value=30.0, step=1.0)

st.sidebar.divider()
st.sidebar.header("Constraints")
shrinkage = st.sidebar.slider("Shrinkage", 0.00, 0.70, 0.30, 0.01)
use_occ = st.sidebar.checkbox("Apply occupancy cap", value=True)
occupancy_target = st.sidebar.slider("Occupancy cap", 0.50, 0.98, 0.85, 0.01) if use_occ else None

st.subheader("Interval Inputs")
df = None
if "intervals_df" in st.session_state and isinstance(st.session_state["intervals_df"], pd.DataFrame):
    df = st.session_state["intervals_df"].copy()
    st.success("Loaded intervals from session_state: `intervals_df`")
else:
    uploaded = st.file_uploader("Upload interval CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

if df is None:
    st.info("Run the Interval Calculator first or upload a CSV.")
    st.stop()

try:
    validate_interval_df(df)
except Exception as e:
    st.error(str(e))
    st.stop()

run = st.button("Compute staffing", type="primary")

if run:
    results = []
    for _, row in df.iterrows():
        inputs = StaffingInputs(
            volume=float(row["volume"]),
            aht_seconds=float(row["aht_seconds"]),
            interval_minutes=int(row["interval_minutes"]),
            is_open=bool(row["is_open"]),
            target_type=target_type,
            service_level_target=float(service_level_target),
            service_level_time_seconds=float(service_level_time_seconds),
            asa_target_seconds=float(asa_target_seconds),
            occupancy_target=float(occupancy_target) if occupancy_target is not None else None,
            shrinkage=float(shrinkage),
        )
        res = compute_required_agents(inputs)
        results.append(
            dict(
                erlangs=res.offered_load_erlangs,
                required_on_phone=res.required_on_phone,
                required_scheduled=res.required_scheduled,
                achieved_service_level=res.achieved_service_level,
                achieved_asa_seconds=res.achieved_asa_seconds,
                achieved_occupancy=res.achieved_occupancy,
            )
        )

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    st.session_state["staffing_df"] = out

    st.subheader("Staffing Results")
    st.dataframe(out, use_container_width=True)

    st.download_button(
        "Download staffing table (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="interval_staffing_table.csv",
        mime="text/csv",
    )