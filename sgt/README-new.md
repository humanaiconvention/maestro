# SGT

Status: active  
Last updated: 2026-03-20  
Scope: subsystem map for semantic grounding and recursive-drift evaluation  
Not authoritative for: current Qwen tuning run state  
Read this next: `../README-new.md`, `../STATUS.md`, `../HANDOFF.md`, `STATUS.md`, `HANDOFF.md`

## What This Subsystem Is

`sgt/` is the semantic grounding and information-preservation testbed.

It exists to study what happens under recursive learning when grounded corrective input is weakened, frozen, replaced, or refreshed.

## Why It Exists

SGT gives the project a structured place to test the viability condition:

`E_t <= C_t`

It is the benchmark and falsification layer for claims about:

- semantic drift
- recursive collapse
- early warning signals
- exogenous correction thresholds

## What Is Stable

- regime structure (`R1` through `R4`)
- benchmark framing around silent semantic drift
- evaluation families and reporting architecture
- role as research testbed rather than public runtime

## What Is Experimental

- exact regime mixes and sweep configurations
- how strongly SGT should be integrated with PRISM-backed telemetry in end-user demonstrations
- how much SGT evidence should surface directly in product/runtime interfaces

## Current Status

SGT is stable and important but not the most time-sensitive surface today.

It remains conceptually central because it is the repo’s main place for reasoning about long-run semantic preservation and recursive collapse.

## Key Areas

- benchmark scripts under `scripts/`
- configuration under `configs/`
- reporting and analysis modules described in existing `README.md`
- test suite under `tests/`

## How It Connects To Other Subsystems

- Root docs:
  - `../README-new.md`
  - `../STATUS.md`
  - `../HANDOFF.md`
- Prism:
  - aligned on geometric and early-warning telemetry
- Maestro:
  - can consume evaluation evidence in larger system stories
- Qwen grounding track:
  - separate operational path, but philosophically aligned with grounding preservation work

## Open Risks

- SGT can become conceptually central but operationally under-documented
- connections to live grounding-model work can remain implicit instead of explicit

## Next Likely Moves

1. Keep SGT as the benchmark/falsification layer, not a catch-all ML folder.
2. Clarify where SGT results should feed into public explanation versus internal evidence.
