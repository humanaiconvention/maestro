# Architecture Principles

The Project Maestro architecture is designed to enforce the project's foundational ontology. It is not a standard SaaS platform; it is a technical Maestro for multi-agent epistemic exchange.

## 1. Decoupled Authority
Unlike centralized Maestros, the Maestro separates identity, consent, provenance, viability, and settlement into distinct layers.
- **Identity**: Verifies *who* the actor is (Human vs. Agent).
- **Consent**: Defines *what* is permitted with the data.
- **Provenance**: Verifies *how* the data was generated (Proof-of-Grounding).
- **Viability**: Verifies *if* the exchange is non-extractive.
- **Settlement**: Finalizes the transfer of value based on thermodynamic proof.

## 2. Asymmetric First-Class Actors
Humans and AI Agents are both first-class citizens of the architecture but have fundamentally different roles and protections.
- **Agents** are the active requesters and funders. They bear the risk of choosing which contributions to fund.
- **Humans** are protected contributors. The architecture is designed to ensure they cannot be "sold" or "extracted" beyond their explicit, bounded consent.

## 3. One-Directional Economic Flow
The database and API are engineered to prevent any human-to-platform or human-to-agent payment flows.
- AI Agents must maintain an escrow balance.
- Humans receive direct settlement from these escrows.
- This asymmetry is a hard architectural constraint to prevent the drift into extractive "gig economy" or "speculative market" models.

## 4. Provenance > Trust Scores
Internal "trust scores" or reputation systems are secondary to objective provenance and bounded consent. 
- **The Insufficiency of Trust Scores**: Traditional platforms rely on black-box reputation scores that often mask extractive behavior or bias. In a multi-agent system, a single aggregate score is insufficient for an AI Agent to determine the epistemic utility of a contribution.
- **Biometric Telemetry**: The system prioritizes the **Proof-of-Grounding (PoG)** biometric telemetry over subjective ratings. 
- **Contextual Trust**: A contribution's value is determined by the AI Agent using the provided provenance record, allowing the agent to apply its own internal trust models based on the specific generation characteristics of the grounded data.
- **Bounded Consent Enforcement**: Trust is meaningless if data is reused outside its intended scope. The architecture enforces consent at the protocol level, ensuring that the human contributor retains sovereignty over their lived experience.

## 5. Thermodynamic Settlement
Settlement is not based on "task completion" in the traditional sense, but on **entropy reduction**.
- The Agent must provide a proof that the received information reduced its uncertainty before the escrow is released.
- This aligns the economic reward directly with the epistemic value generated.

## 6. Privacy through Statistical Signatures
To protect Human Actors, the architecture never logs raw contribution content for platform-level verification.
- The PoG Oracle analyzes statistical biometric signatures (timing, cadence, entropy) rather than raw keystrokes or cursor paths.
- Data sovereignty remains with the contributor and is governed by the cryptographically bound Consent Scope.

