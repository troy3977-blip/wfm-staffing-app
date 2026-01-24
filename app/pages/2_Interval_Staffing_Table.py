import io
import pandas as pd
import streamlit as st

from src.wfm.io import read_interval_csv
from src.wfm.validation import validate_intervals
from src.wfm.staffing import StaffingInputs, compute_required_agents

st.set_page_config(page_title="Interval Staffing Table", layout="wide")
st.title("Interval Staffing Table (CSV Upload)")

st.write(
    """
Upload a CSV with required columns:

- `interval_start` (datetime)
- `interval_minutes` (int)
- `volume` (float)
- `aht_seconds` (float)
- `is_open` (0/1 or true/false)

Then apply global targets and assumptions to compute required agents per interval.
"""
)

with st.sidebar:
    st.header("Global Target / Assumptions")
    target_type = st.selectbox("Target type", options=["service_level", "asa"], index=0)

    if target_type == "service_level":
        sl_target = st.slider("Service Level target", 0.50, 0.99, 0.80, 0.01)
        sl_time = st.number_input("Service Level time threshold (seconds)", min_value=0.0, value=20.0, step=1.0)
        asa_target = 30.0
    else:
        asa_target = st.number_input("ASA target (seconds)", min_value=1.0, value=30.0, step=1.0)
        sl_target = 0.80
        sl_time = 20.0

    shrinkage = st.slider("Shrinkage", 0.0, 0.60, 0.30, 0.01)
    use_occ = st.checkbox("Apply occupancy cap", value=True)
    occupancy_target = st.slider("Occupancy cap", 0.50, 0.95, 0.85, 0.01) if use_occ else None

uploaded = st.file_uploader("Upload interval CSV", type=["csv"])

if uploaded is None:
    st.info("No file uploaded yet. Download a sample CSV below.")
    sample_path = "data/sample_inputs.csv"
    with open(sample_path, "rb") as f:
        st.download_button("Download sample_inputs.csv", data=f, file_name="sample_inputs.csv", mime="text/csv")
    st.stop()

try:
    df = read_interval_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

df_validated = validate_intervals(df)

st.subheader("Input preview (first 20 rows)")
st.dataframe(df_validated.head(20), use_container_width=True)

# Compute staffing
results = []
errors = 0

for _, row in df.iterrows():
    inputs = StaffingInputs(
        volume=float(row["volume"]),
        aht_seconds=float(row["aht_seconds"]) if float(row["volume"]) > 0 else 1.0,
        interval_minutes=int(row["interval_minutes"]),
        is_open=bool(row["is_open"]),
        target_type=target_type,
        service_level_target=float(sl_target),
        service_level_time_seconds=float(sl_time),
        asa_target_seconds=float(asa_target),
        occupancy_target=float(occupancy_target) if occupancy_target is not None else None,
        shrinkage=float(shrinkage),
    )
    try:
        res = compute_required_agents(inputs)
        results.append(
            {
                "interval_start": row["interval_start"],
                "interval_minutes": row["interval_minutes"],
                "volume": row["volume"],
                "aht_seconds": row["aht_seconds"],
                "is_open": row["is_open"],
                "erlangs": res.offered_load_erlangs,
                "required_on_phone": res.required_on_phone,
                "required_scheduled": res.required_scheduled,
                "service_level": res.achieved_service_level,
                "asa_seconds": res.achieved_asa_seconds,
                "occupancy": res.achieved_occupancy,
            }
        )
    except Exception:
        errors += 1
        results.append(
            {
                "interval_start": row["interval_start"],
                "interval_minutes": row["interval_minutes"],
                "volume": row["volume"],
                "aht_seconds": row["aht_seconds"],
                "is_open": row["is_open"],
                "erlangs": None,
                "required_on_phone": None,
                "required_scheduled": None,
                "service_level": None,
                "asa_seconds": None,
                "occupancy": None,
            }
        )

out = pd.DataFrame(results).sort_values("interval_start").reset_index(drop=True)

st.subheader("Results")
st.dataframe(out, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Intervals", f"{len(out)}")
c2.metric("Intervals with errors", f"{errors}")
c3.metric("Total scheduled (sum)", f"{out['required_scheduled'].fillna(0).sum():.0f}")

# Download results
buf = io.StringIO()
out.to_csv(buf, index=False)
st.download_button(
    label="Download results CSV",
    data=buf.getvalue(),
    file_name="interval_staffing_results.csv",
    mime="text/csv",
)