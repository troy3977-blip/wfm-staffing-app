# app/pages/3_Monte_Carlo_Mode.py
from __future__ import annotations

from typing import cast

import pandas as pd
import streamlit as st

from wfm.monte_carlo import (
    MonteCarloConfig,
    run_interval_monte_carlo,
    MAX_SIMS_DEFAULT,
    MAX_TOTAL_SIMS,
    VolumeDist,
    AHTDist,
)
from wfm.staffing import TargetType


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Monte Carlo Mode", layout="wide")
st.title("Monte Carlo Staffing (Risk-Aware)")
st.caption(
    "Simulate uncertainty in interval volume and/or AHT to estimate staffing percentiles (P50/P90/P95) "
    "and risk of missing service targets."
)

st.markdown(
    """
This page estimates **staffing risk** under uncertainty by simulating **N scenarios per interval**.

Use it to answer questions like:
- “How many agents do I need at P90 volume?”
- “What’s the risk of missing 80/60 with today’s staffing assumptions?”
- “How sensitive is staffing to AHT variability?”
"""
)


# -----------------------------
# Sidebar controls (inputs ONLY)
# -----------------------------
with st.sidebar:
    st.header("Simulation Controls")

    n_sims_ui = st.slider(
        "Simulations per interval",
        min_value=1,
        max_value=int(MAX_SIMS_DEFAULT),
        value=2000,
        step=100,
        help=f"Hard-capped at {MAX_SIMS_DEFAULT} for production safety.",
    )
    seed = st.number_input("Random seed", min_value=0, value=10, step=1)

    st.divider()
    st.subheader("Uncertainty Model")

    simulate_volume = st.checkbox("Simulate Volume", value=True)
    volume_dist = st.selectbox("Volume distribution", ["poisson", "normal", "lognormal"], index=0)
    volume_cv = st.slider("Volume CV (normal/lognormal only)", 0.05, 0.60, 0.15, 0.01)

    simulate_aht = st.checkbox("Simulate AHT", value=False)
    aht_dist = st.selectbox("AHT distribution", ["lognormal", "normal"], index=0)
    aht_cv = st.slider("AHT CV", 0.02, 0.60, 0.10, 0.01)

    st.divider()
    st.header("Target + Constraints")

    target_type = cast(TargetType, st.selectbox("Target type", ["service_level", "asa"], index=0))

    # Defaults
    service_level_target = 0.80
    service_level_time_seconds = 60.0
    asa_target_seconds = 30.0

    if target_type == "service_level":
        service_level_target = st.slider("Service Level target", 0.50, 0.99, 0.80, 0.01)
        service_level_time_seconds = st.number_input(
            "Service Level seconds (e.g., 60)",
            min_value=0.0,
            value=60.0,
            step=1.0,
        )
        asa_target_seconds = 30.0
    else:
        asa_target_seconds = st.number_input("ASA target (seconds)", min_value=1.0, value=30.0, step=1.0)
        service_level_target = 0.80
        service_level_time_seconds = 60.0

    shrinkage = st.slider("Shrinkage", 0.00, 0.70, 0.30, 0.01)

    use_occ = st.checkbox("Apply occupancy cap", value=True)
    occupancy_target = None
    if use_occ:
        occupancy_target = st.slider("Occupancy cap", 0.50, 0.98, 0.85, 0.01)


# -----------------------------
# Load interval data
# -----------------------------
st.subheader("Interval Inputs")
st.caption("Preferred: computed from your Interval Staffing Table page via session_state. Fallback: upload CSV.")

df: pd.DataFrame | None = None

if "intervals_df" in st.session_state and isinstance(st.session_state["intervals_df"], pd.DataFrame):
    df = st.session_state["intervals_df"].copy()
    st.success("Loaded intervals from session_state: `intervals_df`")
else:
    uploaded = st.file_uploader("Upload interval CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

if df is None:
    st.info("Run the Interval Staffing Table page first (so `intervals_df` exists), or upload a CSV.")
    st.stop()


# Normalize expected columns (defensive)
try:
    required_cols = {"interval_start", "interval_minutes", "volume", "aht_seconds", "is_open"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["interval_start"] = pd.to_datetime(df["interval_start"], errors="coerce")
    if df["interval_start"].isna().any():
        raise ValueError("interval_start has invalid datetime values (could not parse).")

    df["interval_minutes"] = pd.to_numeric(df["interval_minutes"], errors="coerce").astype("Int64")
    if df["interval_minutes"].isna().any():
        raise ValueError("interval_minutes has invalid numeric values.")
    df["interval_minutes"] = df["interval_minutes"].astype(int)

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype(float)
    if df["volume"].isna().any():
        raise ValueError("volume has invalid numeric values.")

    df["aht_seconds"] = pd.to_numeric(df["aht_seconds"], errors="coerce").astype(float)
    if df["aht_seconds"].isna().any():
        raise ValueError("aht_seconds has invalid numeric values.")

    df["is_open"] = df["is_open"].astype(bool)

except Exception as e:
    st.error(str(e))
    st.stop()

st.dataframe(df, use_container_width=True)


# -----------------------------
# Run Monte Carlo
# -----------------------------
run = st.button("Run Monte Carlo", type="primary")
MAX_SIMS_DEFAULT = 150_000  # ensure consistent with wfm.monte_carlo
n_intervals = len(df)

if n_sims > MAX_N_SIMS:
    st.error(f"n_sims too high. Max allowed is {MAX_N_SIMS}.")
    st.stop()

if n_intervals * n_sims > MAX_TOTAL_SIMS:
    st.error(
        f"Request too large: intervals({n_intervals}) * sims({n_sims}) = {n_intervals*n_sims:,} "
        f"exceeds max {MAX_TOTAL_SIMS:,}. Increase interval size or reduce sims."
    )
    st.stop()

if run:
    # Apply UI cap (and engine also enforces MAX_SIMS_DEFAULT)
    n_sims = min(int(n_sims_ui), int(MAX_SIMS_DEFAULT))

    cfg = MonteCarloConfig(
        n_sims=n_sims,
        seed=int(seed),
        volume_dist=cast(VolumeDist, str(volume_dist)),
        volume_cv=float(volume_cv),
        aht_dist=cast(AHTDist, str(aht_dist)),
        aht_cv=float(aht_cv),
    )

    with st.spinner("Simulating..."):
        out = run_interval_monte_carlo(
            interval_df=df,
            cfg=cfg,
            simulate_volume=bool(simulate_volume),
            simulate_aht=bool(simulate_aht),
            target_type=target_type,
            service_level_target=float(service_level_target),
            service_level_time_seconds=float(service_level_time_seconds),
            asa_target_seconds=float(asa_target_seconds),
            shrinkage=float(shrinkage),
            occupancy_target=float(occupancy_target) if occupancy_target is not None else None,
        )

    st.session_state["mc_results_df"] = out

    if "interval_start" in out.columns and "scheduled_p90" in out.columns:
        st.session_state["mc_p90_recommendations"] = out[["interval_start", "scheduled_p90"]].rename(
            columns={"scheduled_p90": "recommended_scheduled_p90"}
        )

    st.subheader("Monte Carlo Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intervals", f"{len(out)}")
    c2.metric("Sims / Interval", f"{n_sims}")
    c3.metric("Scheduled P90 Total", f"{out['scheduled_p90'].sum():.1f}" if "scheduled_p90" in out.columns else "—")
    c4.metric("Scheduled P95 Total", f"{out['scheduled_p95'].sum():.1f}" if "scheduled_p95" in out.columns else "—")

    st.caption("Percentiles are per-interval. Totals are simple sums across intervals (quick comparison metric).")

    show_cols = [
        "interval_start",
        "interval_minutes",
        "volume",
        "aht_seconds",
        "scheduled_mean",
        "scheduled_p50",
        "scheduled_p90",
        "scheduled_p95",
        "on_phone_mean",
        "on_phone_p50",
        "on_phone_p90",
        "on_phone_p95",
        "sla_breach_rate",
        "asa_mean",
        "occ_mean",
    ]
    show_cols = [c for c in show_cols if c in out.columns]
    st.dataframe(out[show_cols], use_container_width=True)

    st.download_button(
        "Download Monte Carlo Results (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="monte_carlo_staffing.csv",
        mime="text/csv",
    )

    with st.expander("Interpretation (how to use these outputs)"):
        st.markdown(
            """
- **scheduled_p90 / scheduled_p95**: recommended staffing if you want high-percentile coverage.
- **sla_breach_rate**: estimated probability of missing the service-level target (only when `target_type=service_level`).
- **asa_mean** and **occ_mean**: sanity checks to avoid unrealistic utilization.
"""
        )

st.caption(
    "MVP note: This is an uncertainty wrapper around Erlang-C staffing. If you later add Erlang-A, "
    "breach rates become more realistic under abandonment."
)