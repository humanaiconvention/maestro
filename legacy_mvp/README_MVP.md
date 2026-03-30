# Project Maestro: MVP Platform

**A public Maestro for bounded human-to-AI grounding.**

This repository implements the local-first Minimum Viable Product (MVP) of the Project Maestro layer. It is built strictly around the principles of semantic viability, non-compensatory viability gates, bounded consent, provenance tracking, off-chain settlement, and reversible participation.

## Architecture Overview

The system operationalizes the corrective channel between humans and recursive generative systems.
The implementation loop is:
1. Agent and Human actors register.
2. Human defines a granular Consent Scope (e.g., `causal_reasoning` only, max 60 mins).
3. Agent posts a Grounding Request carrying a specific Entropy Reduction Hypothesis and Phase Synchronization Hypothesis.
4. **Request Viability Gate**: The request is evaluated against non-compensatory bounds (e.g., must meet min hourly compensation, no prohibited domains).
5. Human submits a Lived Experience Contribution with a cryptographic Provenance Record.
6. **Contribution Viability Gate**: Evaluates consent matching, provenance completeness, and integrity.
7. **Settlement Constraint Gate**: Checks aggregate constraints before simulated ledger settlement.
8. Audit Trail logs all state transitions.

### Key Technical Decisions
* **Python 3.11 + FastAPI**: Clean, inspectable backend.
* **SQLAlchemy + PostgreSQL**: Strict relational modeling of actors, consent, and provenance.
* **Viability Engines**: Request, Contribution, and Settlement gates are evaluated by dedicated, configurable engines implementing hysteresis margins to prevent state oscillation.
* **Append-Only Audit**: All actions emit typed audit events for transparency.

## Local Setup

### 1. Run the Platform via Docker Compose
To spin up the PostgreSQL database and the FastAPI application locally:

```bash
docker-compose up -d --build
```
The API will be available at `http://localhost:8000`. API docs (Swagger) are automatically generated at `http://localhost:8000/docs`.

### 2. Run the Local Demo Script
The repository includes a self-contained script that walks through the entire epistemic exchange loop without a UI:

```bash
pip install requests
python scripts/demo_local_flow.py
```
*Note: Ensure the FastAPI server is running on port 8000 before executing the script.*

### 3. Run the Test Suite
The backend is fully tested using an in-memory SQLite database, ensuring deterministic and fast execution:

```bash
pip install -r app/requirements.txt
pytest tests_mvp/
```

## Deferred Features (Next Phases)
To keep the MVP tightly focused on the core Maestro loop, the following features are stubbed or deferred:
* **Blockchain Settlement**: Currently simulated via the `Settlements` table and `total_payout` ledger entries.
* **Complex ML Authenticity Models**: Provenance hashes and scores are currently mocked in the client/SDK payloads.
* **Federated Governance (DAO)**: Hardcoded policy YAML drives the Viability Engines.
* **Rich Frontend**: A simple structure is supported via API routes, but the React/Vite SPA is deferred for a later pass.


## Implementation Status

Project Maestro is moving from a local-first prototype to a production-hardened system. The current state of the implementation is as follows:

### 1. Fully Implemented
- **Domain Models**: PostgreSQL schema covering actors, consent, grounding requests, and contributions.
- **Viability Engines**: Request, Contribution, and Settlement engines with hysteresis logic and threshold checks.
- **Service Layer**: Fully implemented logic for audit trails, requests, and settlement processing.
- **Maestro Agent SDK**: Functional gateway for AI agents to participate in the epistemic exchange.
- **Audit Trace**: Append-only system of record for all state transitions and policy decisions.

### 2. Partially Implemented & Resolved
- **Identity API**: Core FastAPI endpoints are operational; missing database model definitions for ActorProfile and ProvenanceRecord have been resolved.
- **Gateway Auth**: JWT verification using python-jose (HS256) is implemented, replacing the previous security stub.
- **Revocation Window**: Previously stubbed logic is now fully implemented, reading from the iability_policy.yaml configuration.

### 3. Intentionally Deferred
- **Blockchain Settlement**: Off-chain ledger simulation is used; on-chain settlement is out of scope for MVP.
- **ML Authenticity Scoring**: Complex biometric scoring is represented by numerical confidence stubs.
- **DAO Governance**: Policy-driven YAML is used in place of on-chain governance mechanisms.
- **Rich Frontend**: The system remains API-first; the React/Vite SPA is deferred for a later phase.
- **Real LLM Routing**: Sophisticated multi-span orchestrator routing is currently simplified.

### 4. Local Setup Guide
To run the project in a development environment:
1. **Configure Secrets**: Copy .env.example to .env in both the infra/ and legacy_mvp/ directories and populate with strong passwords.
2. **Launch Services**: Run docker-compose up -d within legacy_mvp/ to start the database and API.
3. **Initialize Auth (Optional)**: Run python scripts/bootstrap_auth.py (requires a running Hydra instance).
4. **Execute Exchange**: Run python scripts/demo_local_flow.py to perform a full end-to-end epistemic exchange demo.

### 5. Known Gaps
- **Identity API**: Lacks full OIDC provider integration with external authority layers.
- **Orchestrator**: Multi-provider span assembly and complex routing logic are not yet active.
- **Semantic Cache**: Vector similarity reuse gates and threshold-driven selectors are pending.
- **Secrets Service**: Operational secrets are in .env; full KMS/Vault leasing is deferred.
- **Telemetry Plane**: OpenTelemetry traces are initialized but lack deep instrumentation across services.
