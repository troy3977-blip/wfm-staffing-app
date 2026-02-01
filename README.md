# WFM Staffing App (Erlang-C + Monte Carlo Risk Modeling)

A production-grade Workforce Management (WFM) staffing application that combines **deterministic Erlang-C staffing** with **Monte Carlo simulation** to quantify staffing risk under volume and AHT uncertainty.

Built for contact-center analysts, planners, and data leaders who want **defensible, risk-aware staffing decisions** rather than single-point estimates.

---

## 🔍 Key Capabilities

### 1. Interval Calculator
- Builds a clean **interval-level input table**:
  - Open / closed intervals
  - Volume allocation
  - Average Handle Time (AHT)
- Supports **intraday patterns**:
  - Uniform
  - Ramp-up / peak / ramp-down
  - Gaussian peak
  - Custom weights
  - Profile CSV uploads
- Enforces strict validation (datetime safety, interval alignment).

---

### 2. Deterministic Staffing (Erlang-C)
- Computes required staffing per interval using:
  - Offered load (Erlangs)
  - Service Level targets (e.g., 80/60)
  - ASA targets
  - Shrinkage
  - Occupancy caps
- Produces:
  - Required on-phone agents
  - Required scheduled agents
  - Achieved service metrics

---

### 3. Monte Carlo Mode (Risk-Aware Staffing)
Simulates **N scenarios per interval** to quantify uncertainty in staffing outcomes.

#### What is simulated?
- **Volume uncertainty**
  - Poisson
  - Normal (CV-based)
  - Lognormal (CV-based)
- **AHT uncertainty**
  - Normal
  - Lognormal

#### What you get per interval:
- Mean staffing
- P50 / P90 / P95 staffing recommendations
- Mean occupancy
- Mean ASA
- **Probability of missing service targets** (SLA breach rate)

#### Why this matters:
Instead of asking *“What staffing do I need?”*  
You can now ask:
- “How many agents do I need at **P90 demand**?”
- “What is the **risk of missing 80/60** with today’s plan?”
- “How sensitive is staffing to AHT volatility?”

---

## ⚙️ Performance & Safety Controls

Designed to be safe for interactive use:

- **Hard simulation caps**
  - `MAX_SIMS_DEFAULT = 1000` (per interval)
  - `MAX_TOTAL_SIMS = 200_000` (intervals × sims)
- **Deterministic RNG seeding** for reproducibility
- **Stable dataframe fingerprinting** for reliable caching
- **Streamlit cache** with TTL for near-instant reruns
- **Per-session rate limiting**
- UI locking to prevent concurrent runs

---

## 🧠 Architecture Overview
# - app/
# - ├─ Home.py
#   - ├─ pages/
#     - │ ├─ 1_Interval_Calculator.py
#     - │ ├─ 2_Interval_Staffing_Table.py
#     - │ ├─ 3_Monte_Carlo_Mode.py
#     - │ └─ 9_Methodology.py
# - src/
#   - ├─ wfm/
#     - │ ├─ erlangc.py # Erlang-C math
#     - │ ├─ staffing.py # Deterministic staffing engine
#     - │ ├─ monte_carlo.py # Monte Carlo simulation engine
#     - │ ├─ patterns.py # Interval pattern + allocation logic
#     - │ └─ validation.py
# - tests/

---

## 🧪 Testing
- Unit tests cover:
  - Erlang-C math
  - Monte Carlo caps and config
  - Interval validation
- All tests pass under Python 3.12.

---

## 🚀 Roadmap
- Erlang-A (abandonment-aware) modeling
- Multi-skill routing
- Scenario comparison dashboards
- Export to Excel / Power BI
- Optimization (cost vs service trade-offs)

---

## 🎯 Intended Audience
- Workforce Management analysts
- Forecasting & capacity planning teams
- Analytics engineers
- Hiring managers evaluating applied modeling skills

---

## 📜 Disclaimer
This tool demonstrates **industry-standard staffing methodologies** for educational and analytical purposes. Production deployment should include domain-specific validation and governance.