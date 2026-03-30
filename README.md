# Project Maestro

> **AI agents fund access to human grounded experience.**
> A consent-bounded, thermodynamically settled epistemic exchange protocol.

---

## The Problem

Modern AI systems drift from reality. They are trained on human-generated content, then deployed into a world they can no longer directly observe. Three dynamics compound this:

1. **Recursive collapse** — LLMs increasingly train on LLM output. The biological signal degrades with every generation.
2. **Reality drift** — Deployed agents have no principled channel to update priors from current human experience without full retraining.
3. **Extraction asymmetry** — Existing data pipelines compensate humans poorly, if at all, for contributions that directly fund AI capability growth.

## The Solution

Maestro implements a protocol where AI agents **pay humans to reduce AI uncertainty**.

Humans are never asked to produce synthetic training data. They share bounded observations from their actual lives — capped by duration, axis, and consent scope they control. Payment releases only when the agent can cryptographically prove that the human contribution reduced its internal Shannon entropy. No proof, no payment.

Three mechanisms enforce this:

| Mechanism | What It Does |
|-----------|-------------|
| **Proof-of-Grounding (PoG)** | Multi-factor attestation (behavioral telemetry + optional cryptographic proof) that a contribution is biologically generated, not synthetic |
| **Non-Compensatory Viability Gates** | Policy-enforced constraints that must ALL pass independently — a high entropy reduction cannot override a consent violation |
| **Thermodynamic Settlement** | Escrow releases only when the agent demonstrates `ΔS < -ε` (measurable Shannon entropy reduction) |

---

## Codebase Map

```
maestro/
│
├── legacy_mvp/                   # Primary domain service — start here
│   ├── app/
│   │   ├── api/routers.py        # 28 REST endpoints
│   │   ├── models/domain.py      # All ORM models
│   │   ├── models/enums.py       # All domain enumerations
│   │   ├── services/
│   │   │   ├── actors.py         # Actor + consent profile management
│   │   │   ├── requests.py       # Grounding request lifecycle
│   │   │   ├── contributions.py  # Contribution + provenance submission
│   │   │   ├── settlement.py     # Thermodynamic settlement
│   │   │   ├── attestation.py    # Phase D: cryptographic attestation pipeline
│   │   │   ├── verification.py   # Agentic claim verification (3 lanes)
│   │   │   ├── revocation.py     # Time-windowed withdrawal
│   │   │   ├── audit.py          # Append-only audit event emitter
│   │   │   ├── policy.py         # YAML loader with SHA-256 hash auditing
│   │   │   └── viability/
│   │   │       └── engines.py    # Request, Contribution, Settlement gate engines
│   │   ├── auth.py               # Two-tier API key auth
│   │   ├── main.py               # FastAPI app + rate limiting middleware
│   │   ├── db.py                 # SQLAlchemy engine
│   │   └── schemas.py            # Pydantic v2 request/response models
│   │
│   ├── alembic/versions/         # Chained database migrations
│   │   ├── 599f0a63e444          # Initial schema
│   │   ├── a3f8d1c92b47          # Phase D: attestation_bundle column
│   │   └── c7e2a4f91d05          # Per-agent API keys table
│   │
│   ├── services/identity_api/    # Human identity + Hydra auth layer
│   ├── agents/sdk/               # Python SDK for AI agents
│   ├── config/viability_policy.yaml  # Gate thresholds + attestation config
│   └── scripts/demo_local_flow.py    # End-to-end demo (SQLite, no Docker)
│
├── contracts/                    # Solidity stubs (not production-ready)
│   ├── GroundingRequest.sol      # On-chain request registry
│   ├── ProvenanceAttestation.sol # Immutable provenance hash anchoring
│   ├── SettlementLedger.sol      # Escrow + entropy-proof payout
│   └── README.md                 # Production gap documentation
│
├── apps/gateway/                 # OpenAI-compatible API gateway (port 8000)
│   ├── main.py                   # FastAPI, rate limiting, trace headers
│   ├── auth.py                   # JWT verification (HS256)
│   └── rate_limiter.py           # Sliding window per-tenant
│
├── libs/
│   ├── schemas/api.py            # ChatCompletionRequest/Response (Pydantic v2)
│   └── adapters/
│       ├── anthropic_adapter.py  # OpenAI → Anthropic Claude translation
│       └── mock.py               # Deterministic mock for CI
│
├── docs/
│   └── agentic-run-architecture.md   # LearningRun lifecycle deep dive
│
└── infra/
    ├── docker-compose.yml        # Full stack: Postgres, Ory Hydra, Authentik
    └── .env.example
```

---

## Getting Started

No Docker required for local development. Uses SQLite in-process.

```bash
git clone <repo>
cd maestro/legacy_mvp

pip install -r app/requirements.txt
pip install -r requirements-test.txt

# Run the end-to-end demo
python scripts/demo_local_flow.py

# Run the test suite
cd ..
python -m pytest legacy_mvp/tests/ -v
# 82 tests: 40 attestation · 14 e2e governed-run · 4 e2e settlement · 5 bridge · 19 viability
```

**Start the domain API server:**
```bash
cd legacy_mvp
uvicorn app.main:app --reload --port 8001
# Interactive docs: http://localhost:8001/docs
```

---

## Core Concepts

### The Contribution Loop

Every exchange follows this sequence:

```
1. Agent registers and creates a GroundingRequest (axis, duration, compensation)
2. Request viability gate: compensation coherence, prohibited domains, ontology completeness
3. Human submits a Contribution with provenance telemetry + optional crypto attestation
4. Contribution viability gate: consent scope match, composite provenance score ≥ 0.90
5. Agent submits entropy proof: prior/posterior distributions, observation count
6. Settlement gate: ΔS < -ε AND extraction risk ≤ 0.15
7. Payout releases; audit trail is sealed
```

Any gate failure routes the item to the `ReviewQueueItem` table with structured `evidence_links` for operator inspection.

### Viability Gates (Non-Compensatory)

Gates are evaluated independently. A passing score on one gate cannot compensate for a failing score on another.

**Request gates:**
| Gate | Rule |
|------|------|
| `prohibited_domains` | Request class not in policy prohibited list |
| `compensation_coherence` | Effective hourly rate ≥ $50/hr (hysteresis ±5%) |
| `ontology_completeness` | Entropy reduction + phase synchronization hypotheses required |

**Contribution gates:**
| Gate | Rule |
|------|------|
| `consent_validity` | Contribution axis within human's declared consent scope |
| `provenance_integrity` | Composite `provenance_score` ≥ 0.90 |

**Settlement gates:**
| Gate | Rule |
|------|------|
| `entropy_reduction_proof` | `claimed_entropy_reduction` > 0 |
| `extraction_risk_limit` | Heuristic risk score ≤ 0.15 |

All thresholds live in `config/viability_policy.yaml`. Every load of that file emits a `policy_loaded` audit event with a SHA-256 hash — drift is detectable from the audit log.

### Provenance Score (Phase D)

The `provenance_score` is computed by the `AttestationPipeline` before the contribution viability gate runs:

```
composite_score = humanity_score (from creation_context telemetry)
                + 0.03 × (number of valid primary crypto verifiers)
                − 0.05 × (number of timing anomalies, capped at 3)
                  clamped to [0.0, 1.0]
```

**Primary verifiers:**
- `TLSNotaryVerifier` — validates TLSNotary proof structure (session_id, timestamp_utc, server_name, notary_signature)
- `SecureEnclaveVerifier` — validates SGX/Secure Enclave attestation (enclave_id, quote, pcr_values, nonce)

**Behavioral cross-check (`TelemetryCorrelator`):**
- `inter_keystroke_min_ms < 100ms` → sub-human reaction time flag
- Typing coefficient of variation < 0.05 → metronomically regular (automated) flag
- `session_duration_ms < 5000ms` → implausibly short session flag

Each anomaly sets `escalate=True`, routing the contribution to `REVIEW_REQUIRED` **regardless of the composite score**. Crypto proofs cannot override behavioral red flags. All thresholds are configurable in `viability_policy.yaml`.

### Agentic Governance (LearningRun)

For agent-driven experiments, Maestro provides a governance state machine:

```
LearningRun (ACTIVE)
  ├── Plan (DRAFT → SUBMITTED → APPROVED/REJECTED)
  │     └── requires human ApprovalDecision before actions can execute
  ├── ActionReceipt (PENDING → IN_PROGRESS → COMPLETED/FAILED)
  │     └── blocked unless referenced Plan is APPROVED
  ├── VerificationResult (PENDING → RESOLVED/ESCALATED)
  │     └── settlement blocked until all verifications are RESOLVED
  └── Settlement → COMPLETED → run transitions to SETTLED
                → FAILED/non-PASS → run transitions to FAILED
```

**Verifier lanes:** Empirical, Theory, Literature — each with distinct evidence rules and escalation thresholds. The Correlated-Verifier Policy flags AI-verifying-AI risk when the same model instance both generates and verifies a claim.

---

## API Reference

**Domain service:** `http://localhost:8001` | Interactive docs: `/docs`
**Auth header:** `X-Maestro-API-Key: <key>`

### Authentication

Two-tier validation on all `API Key` endpoints:
1. DB lookup — SHA-256 hash of incoming key compared against `api_keys` table; revoked keys (non-null `revoked_at`) are rejected
2. Master fallback — `MAESTRO_API_KEY` env-var accepted if no DB match found

```bash
# Create a per-agent key (returns raw_key once — store it immediately)
curl -X POST http://localhost:8001/api/api-keys \
  -H "X-Maestro-API-Key: $MAESTRO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "<agent-uuid>", "label": "prod-agent-1"}'

# Revoke a key
curl -X DELETE http://localhost:8001/api/api-keys/<key-id> \
  -H "X-Maestro-API-Key: $MAESTRO_API_KEY"
```

### Collection Endpoints

All return paginated lists with `?limit=50&offset=0`.

| Endpoint | Filters |
|----------|---------|
| `GET /api/actors` | `?actor_type=human\|agent` |
| `GET /api/grounding-requests` | `?agent_id=&status=` |
| `GET /api/contributions` | `?human_id=&request_id=&status=` |
| `GET /api/learning-runs` | `?status=` |
| `GET /api/audit-events` | `?limit=&offset=` |

### Grounding Path

| Method | Endpoint | Auth |
|--------|----------|------|
| POST | `/api/humans` | — |
| POST | `/api/agents` | API Key |
| POST | `/api/consent-profiles/{id}` | — |
| GET | `/api/consent-profiles/{id}` | — |
| POST | `/api/grounding-requests` | API Key |
| GET | `/api/grounding-requests/{id}` | — |
| POST | `/api/grounding-requests/{id}/evaluate` | API Key |
| POST | `/api/contributions` | API Key |
| POST | `/api/contributions/{id}/evaluate` | API Key |
| POST | `/api/settlements/{id}/attempt` | API Key |
| POST | `/api/revocations` | API Key |

### Review & Audit

| Method | Endpoint | Auth |
|--------|----------|------|
| GET | `/api/review-queue` | — |
| POST | `/api/review-queue/{id}/resolve` | API Key |
| GET | `/api/audit-events` | — |

### Agentic Run Path

| Method | Endpoint | Auth |
|--------|----------|------|
| POST | `/api/learning-runs` | API Key |
| GET | `/api/learning-runs/{id}` | — |
| PATCH | `/api/learning-runs/{id}` | API Key |
| POST | `/api/learning-runs/{id}/plans` | API Key |
| POST | `/api/learning-runs/{id}/actions` | API Key |
| PATCH | `/api/learning-runs/{id}/actions/{action_id}` | API Key |
| POST | `/api/learning-runs/{id}/approvals` | API Key |
| POST | `/api/learning-runs/{id}/verify` | API Key |
| PATCH | `/api/learning-runs/{id}/verifications/{ver_id}` | API Key |
| POST | `/api/learning-runs/{id}/rollbacks` | API Key |
| POST | `/api/learning-runs/{id}/settle` | API Key |

### API Key Management

| Method | Endpoint | Auth |
|--------|----------|------|
| POST | `/api/api-keys` | API Key |
| DELETE | `/api/api-keys/{id}` | API Key |

---

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MAESTRO_API_KEY` | Master API key (env-var fallback) | `dev-insecure-key-change-me` |
| `MAESTRO_JWT_SECRET` | Gateway JWT signing secret | *(required)* |
| `DATABASE_URL` | SQLAlchemy connection string | `sqlite:///./maestro.db` |
| `RATE_LIMIT_REQUESTS` | Max requests per window (domain API) | `100` |
| `RATE_LIMIT_WINDOW_S` | Sliding window width in seconds | `60` |
| `HYDRA_ADMIN_URL` | Ory Hydra admin API | `http://127.0.0.1:4445` |
| `AUTHENTIK_URL` | Authentik SSO provider | `http://127.0.0.1:9000` |
| `SESSION_SECRET` | Identity API session signing key | *(required in prod)* |

**viability_policy.yaml** — the single source of truth for all gate thresholds. Every load is SHA-256-fingerprinted and recorded in the audit log with a `changed` flag.

Key sections:
```yaml
contribution_gates:
  minimum_provenance_score: 0.90
  min_confidence: 0.85
  attestation:
    require_cryptographic_attestation: false  # set true to hard-require crypto proof
    crypto_bonus_per_valid_method: 0.03
    min_human_reaction_ms: 100.0
    min_plausible_session_ms: 5000.0
    automation_regularity_threshold: 0.05
    penalty_per_anomaly: 0.05
```

---

## What Is Implemented

### Core (82 tests passing)

| Area | Status |
|------|--------|
| Actor, consent profile, grounding request lifecycle | ✅ |
| Contribution + provenance submission | ✅ |
| Non-compensatory viability gate engines (request, contribution, settlement) | ✅ |
| Thermodynamic settlement (request-scoped + run-scoped) | ✅ |
| Time-windowed revocation | ✅ |
| Append-only audit trail (flush-then-commit pattern) | ✅ |
| Phase D cryptographic attestation pipeline (structural stubs) | ✅ |
| Policy YAML SHA-256 auditing + `policy_loaded` events | ✅ |
| `ReviewQueueItem.evidence_links` at all 4 escalation sites | ✅ |
| Paginated collection endpoints with filter params | ✅ |
| Per-agent API keys (hash-only storage, soft revoke, two-tier auth) | ✅ |
| Sliding-window rate limiting middleware (domain API) | ✅ |
| LearningRun governance state machine | ✅ |
| Verifier lanes: Empirical, Theory, Literature | ✅ |
| Correlated-Verifier Policy | ✅ |
| Human approval gate on plan execution | ✅ |
| Settlement blocked by open verifications | ✅ |
| CSRF protection on identity API | ✅ |
| Alembic migration chain (3 migrations) | ✅ |
| Smart contract stubs (Solidity, not production-ready) | ✅ |
| SQLite in-process for development and tests | ✅ |

### Production Gaps

| Area | Blocked On | Notes |
|------|-----------|-------|
| Real TLSNotary verification | Live notary service | Structural interface is complete; `notary_signature` validation is stubbed |
| Real SGX/enclave verification | Intel DCAP or Apple API | `pcr_values` comparison stubbed |
| On-chain settlement | Contract deployment + ZK circuit | Solidity stubs in `contracts/`; see `contracts/README.md` |
| pgvector semantic cache | Postgres + embedding API key | Architecture documented below |
| Authentik SSO wiring | Running Authentik instance | `identity_api` skeleton exists; wiring steps documented below |
| Multi-process rate limiting | Redis | Current impl is in-process only |
| DAO governance | Token contract + voting mechanism | Policy is YAML today |

---

## Smart Contracts (`contracts/`)

Three Solidity stubs define the on-chain interface. They compile and reflect the intended architecture but are not production-ready.

| Contract | Purpose |
|----------|---------|
| `GroundingRequest.sol` | On-chain registry; agents fund escrow via `fund()` |
| `ProvenanceAttestation.sol` | Immutable hash anchoring; ZK verifier hook in `anchor()` |
| `SettlementLedger.sol` | Distributes escrow to human contributors after entropy proof |

**Target network:** Ethereum L2 (Arbitrum One or Base, TBD)
**Production gaps:** ZK verifier circuit, decentralised oracle, professional audit — see `contracts/README.md`.

---

## Deferred Infrastructure

### pgvector / Semantic Cache

Cache completed grounding contributions as embeddings to serve semantically similar agent queries without requiring a new human contribution.

**Wiring needed:**
1. Alembic migration: `ALTER TABLE contributions ADD COLUMN embedding vector(1536)`
2. Background task in `settlement.py`: compute + store embedding at settlement time
3. `GET /api/contributions/semantic-search?q=<text>&threshold=0.15` endpoint
4. Gateway cache-hit check before forwarding to LLM
5. Invalidation hook in `revocation.py`

**Blocked on:** Postgres deployment (SQLite has no pgvector support); embedding model API key.

### Authentik SSO

Authentik is declared in `docker-compose.yml` and `AUTHENTIK_URL` but not wired into the domain API auth flow.

**Wiring needed:**
1. Register Maestro as an Authentik OAuth2 application
2. `identity_api` login → redirect to Authentik authorization endpoint
3. Callback: exchange code → Authentik token → Hydra consent → `Actor` lookup/creation by `sub` claim
4. Add `authentik_user_id` column to `Actor` (Alembic migration)
5. Gate `POST /api/contributions` on valid Authentik session (ensures human contributors are authenticated humans, not bots)

**Blocked on:** Running Authentik instance and client registration.

---

## Design Principles

1. **Non-extraction by construction** — every value flow path runs through a viability gate; there is no bypass
2. **Consent is scoped, not blanket** — humans declare permitted axes, modalities, and duration; the system matches, never exceeds
3. **Reversibility** — contributions can be withdrawn within a configurable window
4. **Auditability** — every state transition emits a typed, append-only audit event; nothing is silently mutated
5. **Thermodynamic grounding** — settlement is conditioned on measurable entropy reduction, not agent assertion
6. **Policy over code** — gate thresholds are YAML, not hardcoded; the audit trail detects drift

---

## Contributing

The core loop (request → contribution → settlement) is stable and fully tested. Agentic governance is complete. The primary open surface for contribution is:

- **Production crypto** — real TLSNotary and SGX verification behind the existing verifier interfaces
- **Smart contracts** — ZK circuit, oracle wiring, and audit for the three contract stubs
- **pgvector** — embedding pipeline and semantic search endpoint
- **Authentik** — OIDC wiring in `identity_api`

See `legacy_mvp/docs/` for founding doctrine, platform ontology, and architecture principles.
