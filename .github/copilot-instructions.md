# WFM Staffing App - Copilot Instructions

## Project Architecture

**Purpose**: Streamlit web app computing required contact-center agents per interval using Erlang-C queueing theory.

**Core Modules** (in `src/wfm/`):
- `erlangc.py`: Pure math - Erlang-C probability, service level, ASA calculations
- `staffing.py`: Business logic - `StaffingInputs` (dataclass) → `StaffingResult` via `compute_required_agents()`
- `io.py`: CSV parsing with validation (normalize `is_open` to bool, parse datetimes)
- `patterns.py`, `validation.py`: Arrival pattern generators and input validation

**Streamlit UI** (in `app/`):
- `Home.py`: Landing page
- `pages/1_Interval_Calculator.py`: Single interval staffing solver
- `pages/2_Interval_Staffing_Table.py`: Bulk CSV processing or volume generation from daily total + intraday pattern
- `pages/9_Methodology.py`: Formulas reference

## Critical Workflows

**Setup**: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

**Run**: `streamlit run app/Home.py`

**Test**: `pytest tests/` (pytest configured in `pyproject.toml`)

**Key Entry Points**:
- `compute_required_agents(inputs: StaffingInputs) → StaffingResult`: Does binary search for minimum agents N meeting either service-level or ASA target. Treats occupancy as a constraint cap (occupancy ≤ target).
- `read_interval_csv(file) → DataFrame`: Validates required columns, normalizes data types.

## Project-Specific Patterns

**Inputs are Frozen Dataclasses**: `StaffingInputs` and `StaffingResult` use `@dataclass(frozen=True)` for immutability and clarity. Always use these when computing staffing.

**Dual Targets**: Inputs support either `target_type="service_level"` (default) or `"asa"`. The `_meets_target()` logic handles both—respect this design when adding solver variants.

**Shrinkage & Hours Logic**:
- `is_open=False` → zero staffing (masked intervals)
- `required_on_phone` = agents needed for service level/ASA
- `required_scheduled = ceil(on_phone / (1 - shrinkage))` (shrinkage-adjusted)

**Occupancy Constraint**: Optional cap (`occupancy_target`). If set, solver enforces `occupancy = erlangs / N ≤ occupancy_target` in addition to target metric.

**CSV Columns**: `interval_start`, `interval_minutes`, `volume`, `aht_seconds`, `is_open` are mandatory (enforced in `io.py`).

## Development Guidelines

- **Math isolation**: Keep Erlang-C calculations pure (no state/side effects). Test with specific numerical examples.
- **Validation early**: `compute_required_agents()` validates all inputs at the start; UI pages should also validate before calling.
- **Pandas conventions**: CSV reads return DataFrames sorted by `interval_start` with reset indices.
- **Type hints**: Use `from __future__ import annotations` and full type hints (legacy Python 3.11+ support).

## External Dependencies

- **Streamlit 1.38.0**: UI framework; use `st.set_page_config()`, `st.info()`, etc.
- **Pandas 2.2.2**: CSV I/O and DataFrame manipulation
- **NumPy 2.0.1**: Available if needed for numerical ops (not currently used directly)
- **Python ≥ 3.11**: Required version

## Azure Rule

When generating code for Azure, running terminal commands for Azure, or performing operations related to Azure, invoke your `azure_development-get_best_practices` tool if available.