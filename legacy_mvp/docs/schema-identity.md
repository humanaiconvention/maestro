# Identity Schema

The `identity_api` acts as the domain owner for identity in Project Maestro. It extends the upstream capabilities of Authentik (Human ID) and Hydra (Machine ID) into a unified application schema.

## Core Entities

1. **Actor**: The central identity primitive. An actor represents a single human (`human`), a single autonomous agent (`agent`), or an internal service (`org_service`). Every entity participating in the Maestro requires an `actor_id`.
2. **CredentialBinding**: Maps external authenticators (passkeys, email links, OIDC federations, API keys) to an `Actor`. An actor can have many bindings. This decouples authentication from identity.
3. **ActorProfile**: Stores human-specific domain data like `pog_eligibility` (Proof of Grounding), `onboarding_state`, and references to payout/consent profiles.
4. **AgentProfile**: Stores machine-specific domain data like `trust_tier`, `software_name`, and references back to the `operator_actor_id` (the human who registered the agent).
5. **AuthEvent**: An immutable audit log of all security-sensitive actions (logins, failures, linking, unlinking).

## Design Philosophy

- **No Passwords**: The application database never stores passwords or password hashes. It relies entirely on `CredentialBinding` records which point to upstream cryptographic verifiable material (Passkeys, Hydra tokens, Authentik sessions).
- **Separation of Concerns**: Hydra knows about access tokens. Authentik knows about login challenges. This schema knows about **reputation, escrow eligibility, and canonical actor mapping**.

## Migrations

We use Alembic for migrations. If you add new models to `models.py`, run:
```bash
alembic revision --autogenerate -m "Description"
alembic upgrade head
```

