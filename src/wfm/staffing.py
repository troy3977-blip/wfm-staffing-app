# src/wfm/staffing.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, TypeAlias

from .erlangc import asa_erlang_c, offered_load_erlangs, service_level_erlang_c

TargetType: TypeAlias = Literal["service_level", "asa"]


# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class StaffingInputs:
    volume: float
    aht_seconds: float
    interval_minutes: int
    is_open: bool

    # Target definition
    target_type: TargetType
    service_level_target: float
    service_level_time_seconds: float
    asa_target_seconds: float

    # Constraints / adjustments
    occupancy_target: Optional[float]
    shrinkage: float


@dataclass(frozen=True)
class StaffingResult:
    offered_load_erlangs: float
    required_on_phone: int
    required_scheduled: int
    achieved_service_level: float
    achieved_asa_seconds: float
    achieved_occupancy: float


# -----------------------------
# Internal helpers
# -----------------------------
def _validate_inputs(inputs: StaffingInputs) -> None:
    if inputs.interval_minutes <= 0:
        raise ValueError("interval_minutes must be > 0")

    if inputs.volume < 0:
        raise ValueError("volume must be >= 0")

    if inputs.volume > 0 and inputs.aht_seconds <= 0:
        raise ValueError("aht_seconds must be > 0 when volume > 0")

    if not (0.0 <= inputs.shrinkage < 1.0):
        raise ValueError("shrinkage must be in [0, 1)")

    if inputs.occupancy_target is not None:
        if not (0.0 < float(inputs.occupancy_target) <= 1.0):
            raise ValueError("occupancy_target must be in (0, 1] when provided")

    if inputs.target_type == "service_level":
        if not (0.0 < inputs.service_level_target < 1.0):
            raise ValueError("service_level_target must be between 0 and 1 (exclusive)")
        if inputs.service_level_time_seconds < 0:
            raise ValueError("service_level_time_seconds must be >= 0")
        # asa_target_seconds can exist but is ignored for SL target
    elif inputs.target_type == "asa":
        if inputs.asa_target_seconds <= 0:
            raise ValueError("asa_target_seconds must be > 0")
        # service_level fields can exist but are ignored for ASA target
    else:
        raise ValueError(f"Unsupported target_type: {inputs.target_type}")


def _meets_target(
    target_type: TargetType,
    sl_achieved: float,
    asa_achieved: float,
    sl_target: float,
    asa_target: float,
) -> bool:
    if target_type == "service_level":
        return sl_achieved >= sl_target
    if target_type == "asa":
        return asa_achieved <= asa_target
    # should be unreachable due to validation
    raise ValueError(f"Unsupported target_type: {target_type}")


def _compute_metrics(
    *,
    a: float,
    n: int,
    aht_seconds: float,
    sl_time_seconds: float,
) -> tuple[float, float, float]:
    """Returns (service_level, asa_seconds, occupancy)."""
    # These functions assume n > a for stability; we will bracket accordingly.
    sl = service_level_erlang_c(a, n, aht_seconds, sl_time_seconds)
    asa = asa_erlang_c(a, n, aht_seconds)
    occ = a / n if n > 0 else 0.0
    return float(sl), float(asa), float(occ)


def _is_feasible(
    inputs: StaffingInputs,
    *,
    a: float,
    n: int,
) -> bool:
    sl, asa, occ = _compute_metrics(
        a=a,
        n=n,
        aht_seconds=inputs.aht_seconds,
        sl_time_seconds=inputs.service_level_time_seconds,
    )

    meets = _meets_target(
        inputs.target_type,
        sl_achieved=sl,
        asa_achieved=asa,
        sl_target=inputs.service_level_target,
        asa_target=inputs.asa_target_seconds,
    )

    if not meets:
        return False

    if inputs.occupancy_target is not None:
        # occupancy cap: do not exceed
        if occ > float(inputs.occupancy_target):
            return False

    return True


def _scheduled_from_on_phone(on_phone: int, shrinkage: float) -> int:
    # shrinkage in [0,1)
    return int(math.ceil(on_phone / (1.0 - shrinkage)))


# -----------------------------
# Public API
# -----------------------------
def compute_required_agents(inputs: StaffingInputs, max_agents: int = 5000) -> StaffingResult:
    """
    Find minimum N (on-phone) such that:
      - the target (service level or ASA) is met, AND
      - optional occupancy cap is met (occupancy <= occupancy_target)

    Returns both on-phone and scheduled agents (shrinkage-adjusted).

    Performance:
      Uses exponential bracketing + binary search (fast for Monte Carlo).
    """
    _validate_inputs(inputs)

    # Closed or no volume => zero staffing outputs (stable behavior)
    if (not inputs.is_open) or inputs.volume == 0:
        return StaffingResult(
            offered_load_erlangs=0.0,
            required_on_phone=0,
            required_scheduled=0,
            achieved_service_level=1.0,
            achieved_asa_seconds=0.0,
            achieved_occupancy=0.0,
        )

    interval_seconds = float(inputs.interval_minutes) * 60.0
    a = float(offered_load_erlangs(inputs.volume, inputs.aht_seconds, interval_seconds))

    # Very tiny load shortcut (prevents corner instability)
    if a <= 1e-9:
        n = 1
        scheduled = _scheduled_from_on_phone(n, inputs.shrinkage)
        sl, asa, occ = _compute_metrics(
            a=a,
            n=n,
            aht_seconds=inputs.aht_seconds,
            sl_time_seconds=inputs.service_level_time_seconds,
        )
        return StaffingResult(
            offered_load_erlangs=a,
            required_on_phone=n,
            required_scheduled=scheduled,
            achieved_service_level=sl,
            achieved_asa_seconds=asa,
            achieved_occupancy=occ,
        )

    # Lower bound: must be strictly above load for Erlang-C stability.
    # Also respect occupancy cap lower bound: a/n <= occ_cap => n >= a/occ_cap
    min_n_occ = 1
    if inputs.occupancy_target is not None:
        occ_cap = float(inputs.occupancy_target)
        min_n_occ = int(math.ceil(a / occ_cap))

    low = max(1, int(math.floor(a)) + 1, min_n_occ)
    if low > max_agents:
        raise RuntimeError(f"Occupancy/loads imply > max_agents (low={low}, max_agents={max_agents})")

    # 1) Exponential bracketing to find a feasible high
    high = low
    while high <= max_agents and not _is_feasible(inputs, a=a, n=high):
        # grow quickly; avoids O(N) scans
        high = min(high * 2, max_agents + 1)

    if high > max_agents:
        raise RuntimeError(f"Could not find staffing solution up to max_agents={max_agents}")

    # 2) Binary search in [low, high] for minimal feasible n
    lo, hi = low, high
    while lo < hi:
        mid = (lo + hi) // 2
        if _is_feasible(inputs, a=a, n=mid):
            hi = mid
        else:
            lo = mid + 1

    n = int(lo)
    sl, asa, occ = _compute_metrics(
        a=a,
        n=n,
        aht_seconds=inputs.aht_seconds,
        sl_time_seconds=inputs.service_level_time_seconds,
    )
    scheduled = _scheduled_from_on_phone(n, inputs.shrinkage)

    return StaffingResult(
        offered_load_erlangs=float(a),
        required_on_phone=n,
        required_scheduled=scheduled,
        achieved_service_level=float(sl),
        achieved_asa_seconds=float(asa),
        achieved_occupancy=float(occ),
    )


def result_to_dict(result: StaffingResult) -> Dict[str, Any]:
    return {
        "erlangs": result.offered_load_erlangs,
        "required_on_phone": result.required_on_phone,
        "required_scheduled": result.required_scheduled,
        "service_level": result.achieved_service_level,
        "asa_seconds": result.achieved_asa_seconds,
        "occupancy": result.achieved_occupancy,
    }


__all__ = [
    "TargetType",
    "StaffingInputs",
    "StaffingResult",
    "compute_required_agents",
    "result_to_dict",
]