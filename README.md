# WFM Interval Staffing App (Erlang-C)

A Streamlit web application that computes required contact-center agents per interval using an Erlang-C staffing engine.

## Features

- Single Interval Calculator
- Interval Staffing Table
  - Upload interval volumes (CSV)
  - Or generate volumes from a daily total using Hours of Operation + Arrival Pattern
- Shrinkage conversion (on-phone → scheduled)
- Optional occupancy cap (as a solve constraint)
- FTE-hours per interval output
- Custom intraday profile CSV + Profile Builder (template + bootstrap)

## Methodology

Offered load (Erlangs): $erlangs = \frac{volume \times aht\_seconds}{interval\_seconds}$

Erlang-C search finds minimum agents $N$ meeting:

- Service level target OR ASA target
- Optional occupancy cap: $\frac{erlangs}{N} \leq occupancy\_target$

Scheduled agents: $scheduled = \lceil \frac{on\_phone}{1 - shrinkage} \rceil$

FTE-hours per interval: $fte\_hours = scheduled \times \frac{interval\_minutes}{60}$

## Run Locally

```bash
python -m venv .venv
# Windows:
#   .venv\Scripts\activate
# Mac/Linux:
#   source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/Home.py
