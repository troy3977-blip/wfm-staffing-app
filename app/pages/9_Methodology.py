import streamlit as st

st.set_page_config(page_title="Methodology", layout="wide")
st.title("Methodology (MVP) — Erlang-C Interval Staffing")

st.markdown(
    """
### Key definitions

- **Offered load (Erlangs)**:  
  \n`erlangs = (volume * aht_seconds) / interval_seconds`

- **Erlang-C staffing**:
  Finds the minimum number of agents `N` such that:
  - Service Level target is met (**SL(T)**) *or* ASA target is met
  - Optional **occupancy cap** is satisfied: `erlangs / N <= occupancy_target`
  - Interval closed → staffing is 0

- **Shrinkage conversion**:
  \n`scheduled = ceil(on_phone / (1 - shrinkage))`

### Notes
- Abandonment is not modeled in Erlang-C MVP. Add Erlang-A later if you need abandonment modeling.
"""
)