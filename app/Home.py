import streamlit as st

st.set_page_config(page_title="WFM Staffing App", layout="wide")

st.title("WFM Staffing App")
st.markdown(
    """
Use the pages on the left:

1. **Interval Calculator** – create/build interval inputs  
2. **Interval Staffing Table** – compute Erlang-C staffing per interval  
3. **Monte Carlo Mode** – simulate risk percentiles under uncertainty  
9. **Methodology** – formulas and assumptions
"""
)