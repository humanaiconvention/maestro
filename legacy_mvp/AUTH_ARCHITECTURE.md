# Project Maestro: Identity & Authority Architecture

This document defines the working, local-first authentication architecture for the Project Maestro. It is designed from the ground up to support the Maestro's dual constituency: **Human Actors** and **Autonomous AI Agents**.

## Core Objective

Provide a robust, self-hostable identity foundation that separates *who the actor is* from *what they are allowed to do*, while strictly enforcing the platform's non-extractive principles.

### Supported Authentication Methods

**For Human Users:**
- **Passkeys / WebAuthn**: The primary, preferred method. It aligns with our zero-friction mandate and provides strong cryptographic proof of possession.
- **Email Magic Links**: A universal, passwordless fallback for users unable to use Passkeys.
- **Maestro ID**: A bespoke branded local account, built on open standards, providing a unified identity across the ecosystem.
- **Account Linking**: The system is prepared for future federation with external identity providers.

**For AI Agents (Machine Actors):**
- **OAuth2 Client Credentials**: For programmatic authentication.
- **`private_key_jwt`**: Public-key-bound client authentication where feasible, ensuring zero shared secrets for high-value agent transactions.
- **Scoped Machine Tokens**: Fine-grained authorization determining exactly which actions an agent can perform within the Maestro.

---

## Architectural Decisions

The architecture strictly adheres to a separation of concerns model utilizing three primary pillars:

### 1. Authentik (Identity Broker & Human UI)
**Responsibility:** The human-facing front door.
Authentik is responsible solely for the human login experience and identity lifecycle. It provides the UI for the Maestro ID flow, manages Passkey enrollment and authentication, handles Email Magic Links, and serves as the entry point for future social/SAML account linking. It handles user self-service account settings but does *not* issue the primary API access tokens.

### 2. Ory Hydra (Authorization Server & Token Authority)
**Responsibility:** The headless OAuth2/OIDC token engine.
Hydra sits behind the scenes issuing, validating, and revoking tokens. It manages access tokens, refresh tokens, and client credentials for AI agents. Hydra explicitly delegates the human login UI to Authentik (via a login/consent node), ensuring a strict separation between the user experience of logging in and the cryptographic infrastructure of token issuance. 

### 3. Maestro Application Services (Domain Model Owner)
**Responsibility:** The source of truth for platform-specific state.
Neither Authentik nor Hydra manages the specific rules of the Project Maestro. A custom API service (the Identity API / Authority Layer) owns the domain models. This includes:
- Canonical actor records (Human vs. Agent profiles).
- Consent state management.
- Proof-of-Grounding (PoG) eligibility flags.
- Future bindings for payouts, reputation scoring, and escrow metrics.


*Note: Casdoor and Keycloak are intentionally excluded from the day-one architecture. However, Authentik provides the necessary OIDC/SAML provider abstractions to federate with other systems later without requiring a schema redesign.*

---

## Rationale & Design Philosophy

The system's design is driven by the following principles:

1. **Human Experience First:** Human users must experience the shortest, lowest-friction path to value. By eliminating passwords in favor of Passkeys and Magic Links, we reduce cognitive load, allowing humans to focus on their biological value generation (PoG) rather than credential management.
2. **Machine Autonomy:** AI agents must be able to authenticate and authorize autonomously without human-in-the-loop bottlenecks. Hydra's standard Client Credentials grant fulfills this requirement flawlessly.
3. **Decoupled Security:** By separating the Identity Broker (Authentik), the Token Authority (Hydra), and the Domain Rules (App Services), we can scale, secure, or swap out individual components without rewriting the entire authentication stack.

## Local Development Setup

To run the authentication stack locally:

1. Copy the environment template:
   ```bash
   cp docker/.env.auth.example docker/.env.auth
   ```
2. Boot the infrastructure (Postgres, Redis, Authentik, Ory Hydra, and the Identity API):
   ```bash
   docker-compose -f docker/docker-compose.auth.yml --env-file docker/.env.auth up -d --build
   ```
3. Run the bootstrap script to create the necessary OAuth2 clients in Hydra:
   ```bash
   python scripts/bootstrap_auth.py
   ```
4. The custom Identity API (serving as Hydra's Login/Consent node) will be available at `http://127.0.0.1:8000`.
5. AI Agents can authenticate using the SDK by requesting a token from Hydra directly at `http://127.0.0.1:4444/oauth2/token`.

## Testing

To run the integration tests for the Identity API subsystem:
```bash
pip install -r services/identity_api/requirements.txt -r services/identity_api/requirements-test.txt
pytest tests/
```

