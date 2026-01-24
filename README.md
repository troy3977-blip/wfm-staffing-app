# WFM Interval Staffing App (Erlang-C) — MVP

A Streamlit web application that computes required contact-center agents per interval using an Erlang-C staffing engine.

## Features (MVP)
- Single Interval Calculator
- Interval Staffing Table (CSV upload → results → download)
- Shrinkage conversion (on-phone → scheduled)
- Optional occupancy cap
- Hours-of-operation masking (`is_open`)

## Methodology
- Offered load (Erlangs): `erlangs = (volume * aht_seconds) / interval_seconds`
- Erlang-C search finds minimum agents `N` meeting:
  - Service level target **or** ASA target
  - optional occupancy cap: `erlangs / N <= occupancy_target`
- Scheduled = `ceil(on_phone / (1 - shrinkage))`

## Running locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/Home.py