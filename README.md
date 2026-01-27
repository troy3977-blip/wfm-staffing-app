Workforce Management Staffing Optimizer (WFM Staffing App)

Production-ready workforce staffing engine and scenario modeling web application built with Python, Streamlit, and Azure.
Designed to operationalize Erlang-based contact-center staffing models with modern CI/CD, cloud deployment, and analytics-grade rigor.

🔍 Problem Statement
Contact-center staffing decisions are often made using:
- opaque spreadsheets,
- manual Erlang calculators,
- inconsistent assumptions across teams.

This leads to:
- SLA misses,
- over-staffing cost,
- slow scenario iteration during planning cycles.
This project transforms classical WFM math into a deployable, testable, and auditable web application.

🚀 Solution Overview
The WFM Staffing App is a cloud-hosted analytics application that allows planners and analysts to:
- Compute interval-level staffing requirements using Erlang-C–based models
- Model service level (SL/ASA) tradeoffs
- Apply shrinkage and occupancy constraints
- Compare scenarios side-by-side
- Export results for downstream planning and reporting
The system separates core mathematical logic from the UI layer, enabling testing, reuse, and future extensions (simulation, optimization, multi-skill routing).

🧠 Modeling Approach
Core Concepts
- Poisson arrivals
- Exponential service times
- Erlang-C queuing formulation
- Interval-based staffing

Inputs
- Call volume
- Average handle time (AHT)
- Interval length
- Target service level (e.g. 80/60)
- Shrinkage
- Occupancy constraints

Outputs
- Required agents (FTE)
- Offered load
- Probability of wait
- ASA / SLA feasibility
All calculations are implemented in pure Python, independently testable from the UI.

🖥 Application Features
- Interactive Streamlit UI
- Executive-friendly results summary
- Scenario comparison (A/B modeling)
- Exportable results (CSV)
- Explicit assumptions documentation
- Health-checked Azure deployment

🏗 Architecture
High-level design
User
 └─▶ Azure App Service (Python / Streamlit)
       ├─ Streamlit UI layer
       ├─ Staffing engine (pure Python)
       ├─ Unit tests
       └─ Application Insights telemetry

Deployment
- Azure App Service (Linux, Python 3.11)
- GitHub Actions CI/CD
- OIDC authentication (no stored secrets)
- Zip Deploy artifact strategy

🔐 Security & DevOps
- OIDC-based GitHub → Azure authentication (no publish profiles)
- Principle of least privilege (managed identity / app registration)
- CI/CD pipeline with:
  - dependency install
  - test execution
  - build artifact creation
  - automated deployment
- Application Insights enabled for observability

🧪 Testing & Quality
- Unit-tested staffing logic
- Deterministic outputs for identical inputs
- Edge-case handling (zero volume, extreme shrinkage)
- Reproducible builds

📦 Repository Structure
wfm-staffing-app/
  app/              # Streamlit UI
  staffing/         # Core staffing engine
  tests/            # Unit tests
  docs/             # Architecture & assumptions
  .github/workflows # CI/CD
  requirements.txt
  README.md

🧩 Example Use Cases
Weekly staffing planning
- Scenario modeling for promotions or seasonality
- SLA feasibility analysis
- Training tool for WFM analysts
- Foundation for optimization or simulation engines

🛣 Roadmap
Planned enhancements:
- Multi-skill routing
- Monte Carlo simulation mode
- Cost-based optimization (min cost vs SLA)
- API layer for enterprise integration
- Role-based access control

🧑‍💼 About This Project
This project was built as a portfolio-grade demonstration of:
- workforce analytics expertise,
- production cloud deployment,
- CI/CD best practices,
- and applied operations research.

It is intentionally designed to mirror how modern analytics products are built and operated in enterprise environments.
