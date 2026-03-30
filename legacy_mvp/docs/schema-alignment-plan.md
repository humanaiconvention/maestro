# Project Maestro: Schema Alignment Plan

This document outlines the strategy for unifying the two divergent SQLAlchemy model definitions in the Project Maestro repository into a single, shared models package.

## 1. Side-by-Side Table Comparison

| Entity | `app/models/domain.py` Table | `services/identity_api/models.py` Table | Key Differences / Conflicts |
| :--- | :--- | :--- | :--- |
| **Actor** | `actors` | `actor` | Name mismatch (`actors` vs `actor`). Column names differ (`id` vs `actor_id`). Enum `ActorType` values differ. |
| **Consent** | `consent_profiles` | `consent_scope` | Name mismatch. `app` version is much more detailed (12+ fields). `identity` version uses JSON for preferences. |
| **Request** | `grounding_requests` | `grounding_request` | Name mismatch (plural vs singular). `app` version has more ontology fields (`entropy_reduction_hypothesis`, etc.). |
| **Contribution** | `contributions` | `experience_contribution` | Name mismatch. `app` version uses `content_payload` (JSON); `identity` uses `content_pointer` (String). |
| **Provenance** | `provenance_records` | `provenance_record` | Name mismatch. Column names differ (`id` vs `provenance_id`). `identity` version has more complex device/review context. |
| **Evaluation** | `viability_evaluations` | `viability_evaluation` | Name mismatch. `identity` version has specific boolean result columns (`harm_exclusion_result`, etc.). |
| **Settlement** | `settlements` | `settlement_event` | Name mismatch. `identity` version includes platform fee and thermodynamic evidence fields. |
| **Audit** | `audit_events` | `auth_event` / `audit_log` | Name mismatch. `app` version is generic; `identity` version is split between auth and actor merge audits. |
| **Profile** | N/A | `human_profile` / `agent_profile` / `actor_profile` | These entities exist only in the `identity_api` model for RBAC and profile management. |
| **Binding** | N/A | `credential_binding` | Exists only in `identity_api` for authentication providers. |

## 2. Source of Truth Identification

To ensure technical integrity, we identify the primary source of truth for each entity:

*   **Auth & Identity Entities**: `services/identity_api/models.py` is the source of truth for `Actor`, `HumanProfile`, `AgentProfile`, `CredentialBinding`, and `AuthEvent`. It handles complex multi-provider authentication.
*   **Epistemic Exchange Entities**: `app/models/domain.py` is the source of truth for `GroundingRequest`, `Contribution`, `ConsentProfile`, and `ViabilityEvaluation`. It contains the granular fields required for non-compensatory gates and thermodynamic proofs.

## 3. Unification Strategy (Migration Plan)

The goal is to move all models to a single package at `D:/maestro/libs/schemas/domain.py`.

### Phase 1: Enum Unification
1.  Merge `app/models/enums.py` and the enums in `identity_api/models.py` into a single `libs/schemas/enums.py`.
2.  Standardize on uppercase values for Status enums (e.g., `PASS`, `FAIL`) and lowercase for Type enums (e.g., `human`, `agent`).

### Phase 2: Table Consolidation
1.  **Actor**: Use `identity_api` model but ensure it supports the relationships required by the `app` services. Use `actor_id` as the primary key name.
2.  **ConsentProfile**: Merge `ConsentScope` and `ConsentProfile`. Use the granular 12-field schema from `app` as the base, adding `identity_api`'s JSON preference fields as optional columns.
3.  **GroundingRequest**: Standardize on the singular name `grounding_request`. Include all ontology fields from both models.
4.  **Contribution**: Standardize on `contribution`. Use `content_payload` (JSON) as it is more flexible for the MVP's current implementation.
5.  **ProvenanceRecord**: Unify into `provenance_record`. Merge the schemas to include both cryptographic hashes and capture context.

### Phase 3: Codebase Refactoring
1.  Update both `app/main.py` and `services/identity_api/main.py` to import from the new `libs/schemas/domain.py`.
2.  Use Alembic to generate a single migration script that renames tables and columns in existing databases to match the new unified schema.

## 4. Conflict Resolution

*   **Naming Collision**: Tables like `actors` and `actor` must be merged. We will standardize on the singular name (`actor`, `contribution`, etc.).
*   **Foreign Keys**: `identity_api` uses `actor_id` while `app` uses `human_id` or `agent_id` pointing to the same table. We will standardize on descriptive names (`agent_actor_id`, `human_actor_id`) for clarity.
*   **Relationships**: `identity_api` has a complex self-referential relationship for operators (AgentProfile -> Actor). This must be preserved while ensuring it doesn't break `app`'s simpler views of the actor.
*   **JSON Handling**: Both schemas use custom `JSONType` decorators. We will unify these into a single shared decorator in `libs/schemas/database_types.py` that handles both PostgreSQL `JSONB` and SQLite `JSON` gracefully.
