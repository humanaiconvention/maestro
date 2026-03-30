// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ProvenanceAttestation
 * @notice Immutable on-chain record of contribution provenance hashes.
 *
 * STUB — not production-ready.  In the full implementation a ZK-SNARK
 * verifier (Groth16 or PLONK) will validate the TLSNotary / Secure Enclave
 * proof bundle before anchoring the hash.  This stub accepts any hash from
 * the trusted oracle (off-chain attestation pipeline).
 *
 * Each ProvenanceRecord stores:
 *   - contributionId  — UUID of the Contribution (off-chain)
 *   - integrityHash   — SHA-256 of the raw contribution payload
 *   - attestationHash — SHA-256 of the AttestationBundle JSON
 *   - humanityScore   — composite provenance score (scaled 0–10000 = 0.00–1.00)
 *   - timestamp       — block.timestamp at anchoring
 */
contract ProvenanceAttestation {
    struct Attestation {
        bytes32 contributionId;
        bytes32 integrityHash;
        bytes32 attestationHash;
        uint16  humanityScore;   // 0–10000 representing 0.00–1.00
        uint256 anchoredAt;
    }

    mapping(bytes32 => Attestation) public attestations;
    address public oracle;

    event AttestationAnchored(
        bytes32 indexed contributionId,
        bytes32 integrityHash,
        uint16  humanityScore
    );

    modifier onlyOracle() {
        require(msg.sender == oracle, "Not oracle");
        _;
    }

    constructor(address _oracle) {
        oracle = _oracle;
    }

    function anchor(
        bytes32 contributionId,
        bytes32 integrityHash,
        bytes32 attestationHash,
        uint16  humanityScore
    ) external onlyOracle {
        require(attestations[contributionId].anchoredAt == 0, "Already anchored");
        require(humanityScore <= 10000, "Score out of range");

        attestations[contributionId] = Attestation({
            contributionId: contributionId,
            integrityHash: integrityHash,
            attestationHash: attestationHash,
            humanityScore: humanityScore,
            anchoredAt: block.timestamp
        });

        emit AttestationAnchored(contributionId, integrityHash, humanityScore);
    }

    function verify(bytes32 contributionId, bytes32 integrityHash)
        external view returns (bool valid, uint16 score)
    {
        Attestation storage a = attestations[contributionId];
        valid = (a.anchoredAt != 0 && a.integrityHash == integrityHash);
        score = a.humanityScore;
    }
}
