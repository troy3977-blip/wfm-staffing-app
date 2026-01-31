# src/wfm/__init__.py

# -----------------------------
# Staffing / Erlang-C core
# -----------------------------
from .staffing import (
    TargetType,
    StaffingInputs,
    StaffingResult,
    compute_required_agents,
    result_to_dict,
)

from .erlangc import (
    offered_load_erlangs,
    service_level_erlang_c,
    asa_erlang_c,
)

# -----------------------------
# Monte Carlo
# -----------------------------
from .monte_carlo import (
    MonteCarloConfig,
    run_interval_monte_carlo,
    validate_interval_df,
    VolumeDist,
    AHTDist,
    MAX_SIMS_DEFAULT,
)

# -----------------------------
# Public API
# -----------------------------
__all__ = [
    # Staffing
    "TargetType",
    "StaffingInputs",
    "StaffingResult",
    "compute_required_agents",
    "result_to_dict",

    # Erlang-C
    "offered_load_erlangs",
    "service_level_erlang_c",
    "asa_erlang_c",

    # Monte Carlo
    "MonteCarloConfig",
    "run_interval_monte_carlo",
    "validate_interval_df",
    "VolumeDist",
    "AHTDist",
    "MAX_SIMS_DEFAULT",
]