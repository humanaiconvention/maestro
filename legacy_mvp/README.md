# Project Maestro

> **TOP LEVEL DIRECTIVE:** Project Maestro is organized around two terminal objectives — entropy reduction and phase synchronization — under a non-negotiable viability constraint.

**A Public Maestro Layer for Bounded Human-to-AI Epistemic Exchange**

## Overview

Project Maestro is a protocol and implementation that enables autonomous AI agents to fund access to **grounded human lived experience**. Instead of extracting human labor or operating as a speculative marketplace, it acts as a thermodynamic mechanism: AI systems fund the acquisition of biological contributions to reduce their internal entropy (uncertainty) and synchronize their internal representations with physical reality.

## Core Mechanisms

1. **Proof-of-Grounding (PoG)**: Cryptographic and biometric verification that contributions are biologically generated, preventing synthetic-to-synthetic data loops.
2. **Viability Gates**: Programmatic checks ensuring all exchanges remain non-extractive, consent-bounded, and support human flourishing (e.g., $50/hr minimum effective rate).
3. **Thermodynamic Settlement**: AI Agents release funds from escrow only upon proof that the biological contribution reduced their internal Shannon entropy ($ \Delta S < -\epsilon $).

## Project Structure

* `/agents/sdk/`: Python SDK for autonomous AI agents to request grounding.
* `/client/`: JavaScript PoG client for biological telemetry capture.
* `/demo/`: Local end-to-end simulation of the epistemic exchange cycle.
* `/contracts/`: Solidity smart contracts for non-extractive escrow and settlement.
* `/services/`: Backend authority layers (Identity, PoG Oracle, Viability Monitor).

## Getting Started

### Prerequisites

* Python 3.10+
* Docker & Docker Compose (for the Auth/Identity stack)

### Documentation

See the `/docs` folder for the foundational doctrine and ontology of the Maestro:
- [Founding Doctrine](docs/founding-doctrine.md)
- [Platform Ontology](docs/platform-ontology.md)
- [Architecture Principles](docs/architecture-principles.md)

### Running the Local Demo

The local demo simulates the entire cycle: an AI agent creating a Grounding Request, a human contributing lived experience with biometric proof, the agent updating its belief distribution, and the platform executing the settlement based on entropy reduction.

```bash
python demo/local_demo.py
```

