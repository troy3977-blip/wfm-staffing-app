from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any

from .erlangc import (
    offered_load_erlangs,
    service_level_erlang_c,
    asa_erlang_c,
)


TargetType = Literal["service_level", "asa"]


@dataclass(frozen=True)
class StaffingInputs:
    volume: float
    aht_seconds: float
    interval_minutes: int
    is_open: bool = True

    # Target definition
    target_type: TargetType = "service_level"
    service_level_target: float = 0.80
    service_level_time_seconds: float = 20.0
    asa_target_seconds: float = 30.0

    # Constraints / adjustments
    occupancy_target: Optional[float] = 0.85  # if None, ignore
    shrinkage: float = 0.30  # 0.30 => 30%


@dataclass(frozen=True)
class StaffingResult:
    offered_load_erlangs: float
    required_on_phone: int
    required_scheduled: int
    achieved_service_level: float
    achieved_asa_seconds: float
    achieved_occupancy: float


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
    raise ValueError(f"Unsupported target_type: {target_type}")


def compute_required_agents(inputs: StaffingInputs, max_agents: int = 5000) -> StaffingResult:
    """
    Find minimum N such that target met and optional occupancy constraint met.
    Returns both on-phone and scheduled agents (shrinkage-adjusted).

    Notes:
    - Hours of operation: if is_open is False => zero staffing outputs.
    - Occupancy constraint is treated as a cap: occupancy <= occupancy_target.
    """
    if inputs.interval_minutes <= 0:
        raise ValueError("interval_minutes must be > 0")
    if inputs.volume < 0:
        raise ValueError("volume must be >= 0")
    if inputs.aht_seconds <= 0 and inputs.volume > 0:
        raise ValueError("aht_seconds must be > 0 when volume > 0")
    if not (0 <= inputs.shrinkage < 1):
        raise ValueError("shrinkage must be in [0, 1)")
    if inputs.target_type == "service_level":
        if not (0 < inputs.service_level_target < 1.0):
            raise ValueError("service_level_target must be between 0 and 1")
        if inputs.service_level_time_seconds < 0:
            raise ValueError("service_level_time_seconds must be >= 0")
    if inputs.target_type == "asa":
        if inputs.asa_target_seconds <= 0:
            raise ValueError("asa_target_seconds must be > 0")

    if not inputs.is_open or inputs.volume == 0:
        return StaffingResult(
            offered_load_erlangs=0.0,
            required_on_phone=0,
            required_scheduled=0,
            achieved_service_level=1.0,
            achieved_asa_seconds=0.0,
            achieved_occupancy=0.0,
        )

    interval_seconds = inputs.interval_minutes * 60.0
    a = offered_load_erlangs(inputs.volume, inputs.aht_seconds, interval_seconds)

    # Start just above load for stability.
    n = max(1, math.ceil(a) + 1)

    while n <= max_agents:
        sl = service_level_erlang_c(a, n, inputs.aht_seconds, inputs.service_level_time_seconds)
        asa = asa_erlang_c(a, n, inputs.aht_seconds)
        occ = float(a / n) if n > 0 else 0.0

        meets = _meets_target(
            inputs.target_type,
            sl_achieved=sl,
            asa_achieved=asa,
            sl_target=inputs.service_level_target,
            asa_target=inputs.asa_target_seconds,
        )

        occ_ok = True
        if inputs.occupancy_target is not None:
            # occupancy_target as a cap: do not exceed.
            occ_ok = occ <= inputs.occupancy_target

        if meets and occ_ok:
            scheduled = int(math.ceil(n / (1.0 - inputs.shrinkage)))
            return StaffingResult(
                offered_load_erlangs=float(a),
                required_on_phone=int(n),
                required_scheduled=scheduled,
                achieved_service_level=float(sl),
                achieved_asa_seconds=float(asa),
                achieved_occupancy=float(occ),
            )

        n += 1

    raise RuntimeError(f"Could not find staffing solution up to max_agents={max_agents}")


def result_to_dict(result: StaffingResult) -> Dict[str, Any]:
    return {
        "erlangs": result.offered_load_erlangs,
        "required_on_phone": result.required_on_phone,
        "required_scheduled": result.required_scheduled,
        "service_level": result.achieved_service_level,
        "asa_seconds": result.achieved_asa_seconds,
        "occupancy": result.achieved_occupancy,
    }