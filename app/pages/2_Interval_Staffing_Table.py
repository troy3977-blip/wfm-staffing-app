import io
import pandas as pd
import streamlit as st

from wfm.io import read_interval_csv
from wfm.validation import validate_intervals
from wfm.staffing import StaffingInputs, compute_required_agents
from wfm.patterns import (
    HoursOfOperation,
    build_day_intervals,
    apply_hours_of_operation,
    pattern_weights,
    allocate_volume_to_intervals,
    read_intraday_profile_csv,
    weights_from_intraday_profile,
    validate_profile_alignment,
    build_profile_template,
    intraday_time_grid,
)

st.set_page_config(page_title="Interval Staffing Table", layout="wide")
st.title("Interval Staffing Table (CSV Upload or Pattern Generation)")

st.write(
    """
Two modes:

1) **Upload interval volumes** (CSV contains `volume` per interval)
2) **Generate interval volumes** from a daily total using **Hours of Operation + Arrival Pattern**
   - Built-ins: uniform / ramp / gaussian
   - Custom intraday profile CSV upload
   - Profile builder: download template + bootstrap a profile
"""
)

with st.sidebar:
    st.header("Mode")
    mode = st.radio("Input mode", ["Upload interval volumes", "Generate from daily total + pattern"], index=1)

    st.divider()
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

    st.divider()
    st.header("Outputs")
    show_fte_hours = st.checkbox("Show FTE-hours per interval", value=True)


# --------------------------
# Mode 1: Upload interval CSV
# --------------------------
if mode == "Upload interval volumes":
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


# ---------------------------------------------------
# Mode 2: Generate from daily total + HOO + pattern
# ---------------------------------------------------
else:
    st.subheader("Generate interval volumes from Hours of Operation + Arrival Pattern")

    colA, colB, colC = st.columns(3)
    with colA:
        date = st.date_input("Date (single day)", value=pd.Timestamp.today().date())
    with colB:
        interval_minutes = st.selectbox("Interval length (minutes)", options=[15, 30, 60], index=0)
    with colC:
        daily_volume = st.number_input("Daily total volume (offered)", min_value=0.0, value=1000.0, step=10.0)

    col1, col2 = st.columns(2)
    with col1:
        hoo_start = st.text_input("Hours of Operation start (HH:MM)", value="09:00")
    with col2:
        hoo_end = st.text_input("Hours of Operation end (HH:MM)", value="17:00")

    aht_seconds_global = st.number_input("AHT (seconds) for generated volumes", min_value=1.0, value=420.0, step=1.0)

    # Profile builder expander
    with st.expander("Profile builder (template + bootstrap)", expanded=False):
        st.write(
            "Download a complete intraday profile template for the selected interval length, "
            "or bootstrap a profile from a built-in curve, then edit weights in Excel and re-upload."
        )

        default_weight = st.number_input("Default weight for template", min_value=0.0, value=1.0, step=0.1)
        tpl = build_profile_template(int(interval_minutes), default_weight=float(default_weight))
        st.download_button(
            "Download intraday profile template CSV",
            data=tpl.to_csv(index=False),
            file_name=f"intraday_profile_template_{interval_minutes}min.csv",
            mime="text/csv",
        )

        st.divider()
        st.write("Bootstrap a profile from a built-in pattern and download it for editing.")
        boot_pattern = st.selectbox(
            "Bootstrap pattern",
            options=["uniform", "ramp_up_peak_ramp_down", "gaussian_peak"],
            index=1,
            key="boot_pattern",
        )
        boot_peak_time = None
        if boot_pattern == "gaussian_peak":
            boot_peak_time = st.text_input("Bootstrap peak time (HH:MM)", value="13:00", key="boot_peak")

        try:
            df_boot = build_day_intervals(pd.Timestamp(date), int(interval_minutes))
            hoo_boot = HoursOfOperation(start_time=hoo_start, end_time=hoo_end, interval_minutes=int(interval_minutes))
            df_boot = apply_hours_of_operation(df_boot, hoo_boot)
            w_boot = pattern_weights(df_boot, pattern=boot_pattern, peak_time=boot_peak_time)
        except Exception as e:
            st.error(f"Could not bootstrap profile: {e}")
            w_boot = None

        if w_boot is not None:
            boot_profile = intraday_time_grid(int(interval_minutes))
            boot_profile["weight"] = 0.0
            boot_times = df_boot["interval_start"].dt.strftime("%H:%M").tolist()
            boot_map = dict(zip(boot_times, w_boot.tolist()))
            boot_profile["weight"] = boot_profile["time"].map(lambda t: float(boot_map.get(t, 0.0)))

            st.download_button(
                "Download bootstrapped profile CSV",
                data=boot_profile.to_csv(index=False),
                file_name=f"intraday_profile_bootstrap_{boot_pattern}_{interval_minutes}min.csv",
                mime="text/csv",
            )

    # Arrival pattern selection
    pattern = st.selectbox(
        "Arrival pattern",
        options=["uniform", "ramp_up_peak_ramp_down", "gaussian_peak", "custom_profile_csv"],
        index=1,
    )

    peak_time = None
    profile_df = None

    if pattern == "gaussian_peak":
        peak_time = st.text_input("Peak time (HH:MM)", value="13:00")

    if pattern == "custom_profile_csv":
        st.markdown("#### Upload intraday profile CSV")
        st.caption("CSV columns required: time (HH:MM), weight (nonnegative). Must include ALL open interval times.")
        profile_file = st.file_uploader("Intraday profile CSV", type=["csv"], key="profile_csv")
        if profile_file is None:
            st.stop()

        try:
            profile_df = read_intraday_profile_csv(profile_file)
        except Exception as e:
            st.error(f"Could not read profile CSV: {e}")
            st.stop()

    # Build day intervals and apply HOO
    df_day = build_day_intervals(pd.Timestamp(date), int(interval_minutes))
    hoo = HoursOfOperation(start_time=hoo_start, end_time=hoo_end, interval_minutes=int(interval_minutes))
    df_day = apply_hours_of_operation(df_day, hoo)
    df_day["aht_seconds"] = float(aht_seconds_global)

    # Determine weights
    if pattern == "custom_profile_csv":
        try:
            validate_profile_alignment(df_day, profile_df)
        except Exception as e:
            st.error(str(e))
            st.stop()
        w = weights_from_intraday_profile(df_day, profile_df)
    else:
        w = pattern_weights(df_day, pattern=pattern, peak_time=peak_time)

    # Allocate volumes
    df = allocate_volume_to_intervals(df_day, daily_volume=float(daily_volume), weights=w, volume_rounding="round")

    with st.expander("Preview weights and allocated volumes (first 60 rows)", expanded=False):
        prev = df_day[["interval_start", "interval_minutes", "is_open"]].copy()
        prev["weight_raw"] = w
        prev["weight_norm"] = (w / w.sum()) if w.sum() > 0 else 0.0
        prev["allocated_volume"] = df["volume"]
        st.dataframe(prev.head(60), use_container_width=True)

df_validated = validate_intervals(df)

st.subheader("Input preview (first 30 rows)")
st.dataframe(df_validated.head(30), use_container_width=True)

# --------------------------
# Compute staffing per interval
# --------------------------
results = []
errors = 0

for _, row in df.iterrows():
    inputs = StaffingInputs(
        volume=float(row["volume"]),
        aht_seconds=float(row["aht_seconds"]) if float(row["volume"]) > 0 else 1.0,
        interval_minutes=int(row["interval_minutes"]),
        is_open=bool(row.get("is_open", True)),
        target_type=target_type,
        service_level_target=float(sl_target),
        service_level_time_seconds=float(sl_time),
        asa_target_seconds=float(asa_target),
        occupancy_target=float(occupancy_target) if occupancy_target is not None else None,
        shrinkage=float(shrinkage),
    )

    try:
        res = compute_required_agents(inputs)
        interval_hours = float(row["interval_minutes"]) / 60.0
        fte_hours = res.required_scheduled * interval_hours

        results.append(
            {
                "interval_start": row["interval_start"],
                "interval_minutes": row["interval_minutes"],
                "is_open": bool(row.get("is_open", True)),
                "volume": row["volume"],
                "aht_seconds": row["aht_seconds"],
                "erlangs": res.offered_load_erlangs,
                "required_on_phone": res.required_on_phone,
                "required_scheduled": res.required_scheduled,
                "fte_hours_interval": fte_hours,
                "service_level": res.achieved_service_level,
                "asa_seconds": res.achieved_asa_seconds,
                "occupancy": res.achieved_occupancy,
            }
        )
    except Exception:
        errors += 1
        results.append(
            {
                "interval_start": row.get("interval_start", None),
                "interval_minutes": row.get("interval_minutes", None),
                "is_open": bool(row.get("is_open", True)) if "is_open" in row else True,
                "volume": row.get("volume", None),
                "aht_seconds": row.get("aht_seconds", None),
                "erlangs": None,
                "required_on_phone": None,
                "required_scheduled": None,
                "fte_hours_interval": None,
                "service_level": None,
                "asa_seconds": None,
                "occupancy": None,
            }
        )

out = pd.DataFrame(results).sort_values("interval_start").reset_index(drop=True)

if not show_fte_hours and "fte_hours_interval" in out.columns:
    out = out.drop(columns=["fte_hours_interval"])

st.subheader("Results")
st.dataframe(out, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Intervals", f"{len(out)}")
c2.metric("Intervals with errors", f"{errors}")
c3.metric("Total scheduled (sum)", f"{out['required_scheduled'].fillna(0).sum():.0f}")
if "fte_hours_interval" in out.columns:
    c4.metric("Total FTE-hours (sum)", f"{out['fte_hours_interval'].fillna(0).sum():.2f}")

buf = io.StringIO()
out.to_csv(buf, index=False)
st.download_button(
    label="Download results CSV",
    data=buf.getvalue(),
    file_name="interval_staffing_results.csv",
    mime="text/csv",
)