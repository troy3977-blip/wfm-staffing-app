import streamlit as st

st.set_page_config(page_title="Methodology", layout="wide")
st.title("Methodology (MVP) — Erlang-C Interval Staffing")

st.markdown(
    """
### Key definitions

- **Offered load (Erlangs)**:  
  `erlangs = (volume * aht_seconds) / interval_seconds`

- **Erlang-C staffing**:
  Finds the minimum number of agents `N` meeting:
  - Service Level target **or** ASA target
  - Optional **occupancy cap**: `erlangs / N <= occupancy_target`
  - Interval closed → staffing is 0

- **Shrinkage conversion**:
  `scheduled = ceil(on_phone / (1 - shrinkage))`

- **FTE-hours per interval (output)**:
  `fte_hours_interval = scheduled * (interval_minutes / 60)`

### Arrival patterns (generated mode)
You may allocate a daily total volume into interval volumes using:
- Uniform / Ramp / Gaussian peak
- Custom intraday profile CSV (time,weight)
- Profile builder (template + bootstrap)
"""
)