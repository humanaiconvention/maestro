# Maestro Smart Contracts

> **Status: Stub / Not Production-Ready**
>
> These contracts define the on-chain interface for Project Maestro.  They are
> structural stubs that compile and reflect the intended architecture.  ZK-proof
> verification, Chainlink oracle integration, and production key management are
> not yet implemented.

---

## Contracts

| Contract | Purpose |
|---|---|
| `GroundingRequest.sol` | Registry of human-grounding requests posted by AI agents |
| `ProvenanceAttestation.sol` | Immutable anchoring of contribution provenance hashes |
| `SettlementLedger.sol` | Escrow and payout for Proof-of-Grounding settlements |

## Architecture

```
Agent                    GroundingRequest.sol
  │  createRequest()          │
  ├──────────────────────────►│
  │                           │ stores request + escrow
  │
  │  (off-chain contributions evaluated by legacy_mvp)
  │
Oracle (trusted)         ProvenanceAttestation.sol
  │  anchor(hash, score)      │
  ├──────────────────────────►│  immutable on-chain record
  │
Oracle (trusted)         SettlementLedger.sol
  │  settle(recipients,       │
  │         shares, evalHash) │
  ├──────────────────────────►│  distributes escrow to humans
```

## Deployment (target)

- Network: Ethereum L2 (Arbitrum One or Base — TBD)
- Compiler: solc 0.8.20
- Framework: Hardhat or Foundry

## Production gaps (before mainnet)

1. **ZK verifier** — `ProvenanceAttestation.anchor()` must call an on-chain
   SNARK verifier (Groth16/PLONK) to validate the TLSNotary / Secure Enclave
   proof bundle before accepting the humanity score.

2. **Oracle trust model** — replace the `onlyOracle` modifier with a
   decentralised oracle network (e.g. Chainlink Functions) or multi-sig.

3. **Entropy proof** — `SettlementLedger.settle()` must verify the
   thermodynamic settlement claim (ΔS < -ε) on-chain before releasing escrow.

4. **Access control** — replace `owner`/`oracle` addresses with OpenZeppelin
   `AccessControl` roles.

5. **Audit** — full professional security audit before mainnet deployment.
