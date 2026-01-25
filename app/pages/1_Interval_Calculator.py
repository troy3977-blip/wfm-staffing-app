import streamlit as st

from staffing import StaffingInputs, compute_required_agents

st.set_page_config(page_title="Interval Calculator", layout="wide")
st.title("Single Interval Calculator")

with st.sidebar:
    st.header("Interval Inputs")
    volume = st.number_input("Volume (contacts offered)", min_value=0.0, value=100.0, step=1.0)
    aht = st.number_input("AHT (seconds)", min_value=1.0, value=420.0, step=1.0)
    interval_minutes = st.selectbox("Interval length (minutes)", options=[15, 30, 60], index=0)
    is_open = st.checkbox("Interval is open", value=True)

    st.divider()
    st.header("Target")
    target_type = st.selectbox("Target type", options=["service_level", "asa"], index=0)

    if target_type == "service_level":
        sl_target = st.slider("Service Level target", min_value=0.50, max_value=0.99, value=0.80, step=0.01)
        sl_time = st.number_input("Service Level time threshold (seconds)", min_value=0.0, value=20.0, step=1.0)
        asa_target = 30.0
    else:
        asa_target = st.number_input("ASA target (seconds)", min_value=1.0, value=30.0, step=1.0)
        sl_target = 0.80
        sl_time = 20.0

    st.divider()
    st.header("Adjustments / Constraints")
    shrinkage = st.slider("Shrinkage", min_value=0.0, max_value=0.60, value=0.30, step=0.01)

    use_occ = st.checkbox("Apply occupancy cap", value=True)
    occupancy_target = st.slider("Occupancy cap", min_value=0.50, max_value=0.95, value=0.85, step=0.01) if use_occ else None

inputs = StaffingInputs(
    volume=volume,
    aht_seconds=aht,
    interval_minutes=int(interval_minutes),
    is_open=is_open,
    target_type=target_type,  # type: ignore
    service_level_target=float(sl_target),
    service_level_time_seconds=float(sl_time),
    asa_target_seconds=float(asa_target),
    occupancy_target=float(occupancy_target) if occupancy_target is not None else None,
    shrinkage=float(shrinkage),
)

try:
    result = compute_required_agents(inputs)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Offered Load (Erlangs)", f"{result.offered_load_erlangs:.2f}")
col2.metric("Required On-Phone Agents", f"{result.required_on_phone}")
col3.metric("Required Scheduled Agents", f"{result.required_scheduled}")

st.subheader("Achieved performance")
c1, c2, c3 = st.columns(3)
c1.metric("Service Level", f"{result.achieved_service_level:.3f}")
c2.metric("ASA (seconds)", f"{result.achieved_asa_seconds:.1f}")
c3.metric("Occupancy", f"{result.achieved_occupancy:.3f}")

st.caption("MVP uses Erlang-C. Abandonment is not modeled (add Erlang-A later if required).")