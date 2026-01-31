import streamlit as st

st.set_page_config(page_title="Methodology", layout="wide")
st.title("Methodology")

st.markdown(
    r"""
### Erlang C

- Offered load (Erlangs):  
  \[
  a = \frac{\text{volume} \cdot \text{AHT}}{\text{interval\_seconds}}
  \]

- Probability of wait (Erlang C):
  \[
  P(W>0) = \frac{\frac{a^n}{n!}\cdot\frac{n}{n-a}}{\sum_{k=0}^{n-1}\frac{a^k}{k!} + \frac{a^n}{n!}\cdot\frac{n}{n-a}}
  \]

- ASA:
  \[
  ASA = P(W>0)\cdot \frac{AHT}{n-a}
  \]

- Service Level at threshold \(T\):
  \[
  SL(T)=1 - P(W>0)\cdot e^{-(n-a)\cdot(T/AHT)}
  \]

### Staffing logic

- Find the minimum **on-phone** \(n\) such that the chosen target is met.
- If occupancy cap is enabled, require:
  \[
  \text{occupancy} = \frac{a}{n} \le \text{occupancy\_target}
  \]
- Convert to **scheduled** using shrinkage:
  \[
  \text{scheduled} = \left\lceil \frac{n}{1-\text{shrinkage}} \right\rceil
  \]

### Monte Carlo mode

- Draw random volume and/or AHT per interval.
- Compute staffing per draw.
- Report mean + percentiles (P50/P90/P95) per interval.
"""
)