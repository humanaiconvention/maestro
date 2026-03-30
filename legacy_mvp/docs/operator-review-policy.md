# Operator Review Policy

The Viability Engine includes an **Operator Review Queue** for ambiguous cases that require human judgment to protect the integrity of the Maestro.

## 1. When Review is Required
Requests enter the review queue if they:
- Map to sensitive request classes (e.g., `medical_inquiry`, `financial_behavior`).
- Are flagged by heuristics for potential coercive or deceptive framing.
- Fall near threshold boundaries for compensation or provenance scores.

## 2. Review Criteria
Operators must evaluate the request against the **Prohibited Modes**:
- Is the framing deceptive?
- Does it resemble a human-subject experiment?
- Is the compensation fund sufficient for the biological labor required?
- Is the entropy hypothesis plausible?

## 3. Decision Transparency
Every operator decision (Approve or Block) must include a public explanation attached to the `ViabilityEvaluation` record.

