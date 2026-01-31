# src/wfm/monte_carlo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, TypeAlias, cast

import numpy as np
import pandas as pd

from .staffing import StaffingInputs, TargetType, compute_required_agents


# -----------------------------
# Public types + safety caps
# -----------------------------
VolumeDist: TypeAlias = Literal["poisson", "normal", "lognormal"]
AHTDist: TypeAlias = Literal["normal", "lognormal"]

# UI / safety caps (used by app; safe to import in tests too)
MAX_SIMS_DEFAULT: int = 2000
MAX_TOTAL_SIMS: int = 150_000


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class MonteCarloConfig:
    n_sims: int = 1000
    seed: int = 42
    volume_dist: VolumeDist = "poisson"
    volume_cv: float = 0.15
    aht_dist: AHTDist = "lognormal"
    aht_cv: float = 0.10


# -----------------------------
# Validation
# -----------------------------
_REQUIRED_INTERVAL_COLS = {"interval_start", "interval_minutes", "volume", "aht_seconds", "is_open"}


def validate_interval_df(df: pd.DataFrame) -> None:
    missing = _REQUIRED_INTERVAL_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Interval dataframe missing required columns: {sorted(missing)}. "
            f"Required: {sorted(_REQUIRED_INTERVAL_COLS)}"
        )

    if pd.to_datetime(df["interval_start"], errors="coerce").isna().any():
        raise ValueError("interval_start contains non-datetime values (or unparsable strings).")
    if pd.to_numeric(df["interval_minutes"], errors="coerce").isna().any():
        raise ValueError("interval_minutes contains non-numeric values.")
    if pd.to_numeric(df["volume"], errors="coerce").isna().any():
        raise ValueError("volume contains non-numeric values.")
    if pd.to_numeric(df["aht_seconds"], errors="coerce").isna().any():
        raise ValueError("aht_seconds contains non-numeric values.")


# -----------------------------
# Random draws
# -----------------------------
def _draw_volume(
    base_volume: float,
    dist: VolumeDist,
    n: int,
    rng: np.random.Generator,
    cv: float,
) -> np.ndarray:
    v = max(float(base_volume), 0.0)

    if dist == "poisson":
        draws = rng.poisson(lam=v, size=n).astype(float)
    elif dist == "normal":
        sigma = max(float(cv), 1e-9) * max(v, 1e-9)
        draws = rng.normal(loc=v, scale=sigma, size=n)
    elif dist == "lognormal":
        cv2 = max(float(cv), 1e-9)
        mu = np.log(max(v, 1e-9)) - 0.5 * np.log(1.0 + cv2**2)
        sigma = np.sqrt(np.log(1.0 + cv2**2))
        draws = rng.lognormal(mean=mu, sigma=sigma, size=n)
    else:
        raise ValueError(f"Unsupported volume distribution: {dist}")

    return np.clip(draws, 0.0, None)


def _draw_aht(
    base_aht_seconds: float,
    dist: AHTDist,
    n: int,
    rng: np.random.Generator,
    cv: float,
) -> np.ndarray:
    a = max(float(base_aht_seconds), 1.0)

    if dist == "lognormal":
        cv2 = max(float(cv), 1e-9)
        mu = np.log(a) - 0.5 * np.log(1.0 + cv2**2)
        sigma = np.sqrt(np.log(1.0 + cv2**2))
        draws = rng.lognormal(mean=mu, sigma=sigma, size=n)
    elif dist == "normal":
        sigma = max(float(cv), 1e-9) * a
        draws = rng.normal(loc=a, scale=sigma, size=n)
    else:
        raise ValueError(f"Unsupported AHT distribution: {dist}")

    return np.clip(draws, 1.0, None)


# -----------------------------
# Core simulation for one interval
# -----------------------------
def _simulate_interval(
    *,
    base_volume: float,
    base_aht_seconds: float,
    interval_minutes: int,
    is_open: bool,
    cfg: MonteCarloConfig,
    simulate_volume: bool,
    simulate_aht: bool,
    target_type: TargetType,
    service_level_target: float,
    service_level_time_seconds: float,
    asa_target_seconds: float,
    shrinkage: float,
    occupancy_target: Optional[float],
) -> Dict[str, Any]:
    if not is_open or float(base_volume) <= 0.0:
        return {
            "scheduled_mean": 0.0,
            "scheduled_p50": 0.0,
            "scheduled_p90": 0.0,
            "scheduled_p95": 0.0,
            "on_phone_mean": 0.0,
            "on_phone_p50": 0.0,
            "on_phone_p90": 0.0,
            "on_phone_p95": 0.0,
            "sla_breach_rate": 0.0 if target_type == "service_level" else np.nan,
            "asa_mean": 0.0,
            "occ_mean": 0.0,
        }

    n_sims = int(cfg.n_sims)
    if n_sims <= 0:
        raise ValueError("cfg.n_sims must be > 0")

    # Deterministic per interval but not identical everywhere
    mix = hash((cfg.seed, float(base_volume), float(base_aht_seconds), int(interval_minutes))) & 0xFFFFFFFF
    rng = np.random.default_rng(int(mix))

    vol_draws = (
        _draw_volume(base_volume, cfg.volume_dist, n_sims, rng, cfg.volume_cv)
        if simulate_volume
        else np.full(n_sims, float(base_volume))
    )
    aht_draws = (
        _draw_aht(base_aht_seconds, cfg.aht_dist, n_sims, rng, cfg.aht_cv)
        if simulate_aht
        else np.full(n_sims, float(base_aht_seconds))
    )

    scheduled = np.empty(n_sims, dtype=float)
    on_phone = np.empty(n_sims, dtype=float)
    asa_vals = np.empty(n_sims, dtype=float)
    occ_vals = np.empty(n_sims, dtype=float)
    breach = np.empty(n_sims, dtype=float)

    for i in range(n_sims):
        inputs = StaffingInputs(
            volume=float(vol_draws[i]),
            aht_seconds=float(aht_draws[i]),
            interval_minutes=int(interval_minutes),
            is_open=True,
            target_type=target_type,
            service_level_target=float(service_level_target),
            service_level_time_seconds=float(service_level_time_seconds),
            asa_target_seconds=float(asa_target_seconds),
            occupancy_target=float(occupancy_target) if occupancy_target is not None else None,
            shrinkage=float(shrinkage),
        )

        res = compute_required_agents(inputs)

        scheduled[i] = float(res.required_scheduled)
        on_phone[i] = float(res.required_on_phone)
        asa_vals[i] = float(res.achieved_asa_seconds)
        occ_vals[i] = float(res.achieved_occupancy)

        if target_type == "service_level":
            breach[i] = 1.0 if float(res.achieved_service_level) < float(service_level_target) else 0.0
        else:
            breach[i] = np.nan

    out: Dict[str, Any] = {
        "scheduled_mean": float(np.mean(scheduled)),
        "scheduled_p50": float(np.quantile(scheduled, 0.50)),
        "scheduled_p90": float(np.quantile(scheduled, 0.90)),
        "scheduled_p95": float(np.quantile(scheduled, 0.95)),
        "on_phone_mean": float(np.mean(on_phone)),
        "on_phone_p50": float(np.quantile(on_phone, 0.50)),
        "on_phone_p90": float(np.quantile(on_phone, 0.90)),
        "on_phone_p95": float(np.quantile(on_phone, 0.95)),
        "asa_mean": float(np.mean(asa_vals)),
        "occ_mean": float(np.mean(occ_vals)),
        "sla_breach_rate": float(np.nanmean(breach)) if not np.all(np.isnan(breach)) else np.nan,
    }
    return out


# -----------------------------
# Public API
# -----------------------------
def run_interval_monte_carlo(
    *,
    interval_df: pd.DataFrame,
    cfg: MonteCarloConfig,
    simulate_volume: bool = True,
    simulate_aht: bool = False,
    target_type: TargetType = "service_level",
    service_level_target: float = 0.80,
    service_level_time_seconds: float = 60.0,
    asa_target_seconds: float = 30.0,
    shrinkage: float = 0.30,
    occupancy_target: Optional[float] = 0.85,
) -> pd.DataFrame:
    validate_interval_df(interval_df)

    df = interval_df.copy()
    df["interval_start"] = pd.to_datetime(df["interval_start"], errors="coerce")
    df["interval_minutes"] = pd.to_numeric(df["interval_minutes"], errors="coerce").astype(int)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype(float)
    df["aht_seconds"] = pd.to_numeric(df["aht_seconds"], errors="coerce").astype(float)
    df["is_open"] = df["is_open"].astype(bool)

    tt = cast(TargetType, target_type)

    rows: list[Dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append(
            _simulate_interval(
                base_volume=float(r["volume"]),
                base_aht_seconds=float(r["aht_seconds"]),
                interval_minutes=int(r["interval_minutes"]),
                is_open=bool(r["is_open"]),
                cfg=cfg,
                simulate_volume=bool(simulate_volume),
                simulate_aht=bool(simulate_aht),
                target_type=tt,
                service_level_target=float(service_level_target),
                service_level_time_seconds=float(service_level_time_seconds),
                asa_target_seconds=float(asa_target_seconds),
                shrinkage=float(shrinkage),
                occupancy_target=float(occupancy_target) if occupancy_target is not None else None,
            )
        )

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


__all__ = [
    "VolumeDist",
    "AHTDist",
    "MAX_SIMS_DEFAULT",
    "MAX_TOTAL_SIMS",
    "MonteCarloConfig",
    "run_interval_monte_carlo",
    "validate_interval_df",
]