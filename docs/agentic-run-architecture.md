# Agentic Run Architecture

This document describes the architectural bridge between the legacy Maestro governance/evidence kernel and the emerging agent operating system.

## The `LearningRun`
The **`LearningRun`** is the top-level construct representing a governed execution block where an agent interacts with the physical or digital environment to reduce uncertainty. It replaces the looser request or message-level scope for settlement. All artifacts—plans, action receipts, approval decisions, verification results, and rollbacks—are linked to a single `learning_run_id`.

## Run-Scoped Artifact Flow
An agent run proceeds through typed first-class artifacts:
1. **Plan**: The agent proposes an objective and steps.
2. **ApprovalDecision**: A human or policy gate optionally approves the plan.
3. **ActionReceipt**: The agent executes steps and logs cryptographic receipts of inputs, outputs, and runtime state.
4. **VerificationResult**: Scientific or policy verifiers analyze claims and evidence against the receipts.
5. **Settlement**: If verification passes and entropy is demonstrably reduced, the run settles, releasing payout or concluding the block.
6. **RollbackRecord**: If the plan creates critical violations, the state is reverted using rollback references.

## Verifier Lanes
When an agent submits a claim of epistemic value, it is routed into one of three distinct verification lanes:
- **Empirical**: Assesses code hashes, dataset reproduction, config validation, and variance testing.
- **Theory**: Checks formalizable statements, proof statuses, and checks for counterexamples.
- **Literature**: Audits retrieved sources, conflict summaries, and rationale for synthesizing previously known facts.

### Five-Stage Verifier Pipeline
1. **Normalize**: Claim is routed to Empirical, Theory, or Literature.
2. **Collect**: Relevant receipts, code hashes, and sources are assembled.
3. **Check**: Domain-specific logic is executed (e.g., verifying a proof, or rerunning an environment).
4. **Attach**: Cryptographic provenance and rationales are bound to the result.
5. **Escalate**: Ambiguous or failing verdicts are placed in the review queue.

## Mandatory Human Review
Human validation is mandatory and cannot be bypassed when specific ambiguities are detected:
- **Empirical**: Variance is inexplicably high, reruns diverge, or artifacts are missing.
- **Theory**: The agent's assumptions cannot be automatically formalized, or open proof obligations remain.
- **Literature**: High-quality retrieved sources are in direct conflict, or the rationale is suspiciously thin.

## Correlated-Verifier Policy
A primary security concern in the agent layer is **AI-verifying-AI risk** (where synthetic models grade their own homework). The system invokes a **Correlated-Verifier Policy**:

- If a verification flow (e.g., theory or literature) relies exclusively on endogenous checks or synthetic auditors, it triggers a correlated-verifier risk flag.
- High-risk runs are given a status of `blocked_pending_exogenous_check`, meaning settlement cannot occur until external, independent validation (often human or cryptographic physical sensors) confirms the reality ground-truth.
- Claims flagged as `metaphorical_or_nonliteral` (e.g., claiming negative entropy for processing a text file without physical mapping) are rejected or skipped for thermodynamic payout.
