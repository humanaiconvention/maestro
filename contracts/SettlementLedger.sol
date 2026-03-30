// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./GroundingRequest.sol";
import "./ProvenanceAttestation.sol";

/**
 * @title SettlementLedger
 * @notice Escrow and payout contract for Proof-of-Grounding settlements.
 *
 * STUB — not production-ready.  The thermodynamic settlement proof
 * (ΔS < -ε) will be verified on-chain by a ZK circuit in the full
 * implementation.  This stub trusts the off-chain oracle to submit
 * verified settlement claims.
 *
 * Flow:
 *   1. Agent deposits funds via fund(requestId)
 *   2. Off-chain settlement engine evaluates entropy reduction
 *   3. Oracle calls settle() with the evaluation hash
 *   4. Contract distributes payout to human contributors
 *   5. Unclaimed escrow is returned to agent after timeout
 */
contract SettlementLedger {
    struct Escrow {
        address agent;
        uint256 balance;
        uint256 fundedAt;
        bool    settled;
    }

    mapping(bytes32 => Escrow) public escrows;
    mapping(bytes32 => address[]) public contributors;

    address public oracle;
    uint256 public constant ESCROW_TIMEOUT = 30 days;

    event Funded(bytes32 indexed requestId, address agent, uint256 amount);
    event Settled(bytes32 indexed requestId, uint256 totalPayout, uint256 recipientCount);
    event EscrowReclaimed(bytes32 indexed requestId, address agent, uint256 amount);

    modifier onlyOracle() {
        require(msg.sender == oracle, "Not oracle");
        _;
    }

    constructor(address _oracle) {
        oracle = _oracle;
    }

    function fund(bytes32 requestId) external payable {
        require(msg.value > 0, "Zero value");
        require(!escrows[requestId].settled, "Already settled");
        escrows[requestId].agent = msg.sender;
        escrows[requestId].balance += msg.value;
        if (escrows[requestId].fundedAt == 0) {
            escrows[requestId].fundedAt = block.timestamp;
        }
        emit Funded(requestId, msg.sender, msg.value);
    }

    /**
     * @notice Oracle submits settlement after off-chain viability evaluation.
     * @param requestId   Matches GroundingRequest.id (bytes32 of UUID)
     * @param recipients  Human contributor wallet addresses
     * @param shares      Payout share per contributor (must sum to escrow balance)
     * @param evalHash    SHA-256 of the ViabilityEvaluation record (audit trail)
     */
    function settle(
        bytes32 requestId,
        address[] calldata recipients,
        uint256[] calldata shares,
        bytes32 evalHash
    ) external onlyOracle {
        Escrow storage e = escrows[requestId];
        require(e.balance > 0, "No escrow");
        require(!e.settled, "Already settled");
        require(recipients.length == shares.length, "Length mismatch");

        uint256 total;
        for (uint i = 0; i < shares.length; i++) {
            total += shares[i];
        }
        require(total == e.balance, "Shares must equal escrow balance");

        e.settled = true;
        for (uint i = 0; i < recipients.length; i++) {
            payable(recipients[i]).transfer(shares[i]);
        }

        emit Settled(requestId, total, recipients.length);
    }

    function reclaimEscrow(bytes32 requestId) external {
        Escrow storage e = escrows[requestId];
        require(msg.sender == e.agent, "Not agent");
        require(!e.settled, "Already settled");
        require(block.timestamp >= e.fundedAt + ESCROW_TIMEOUT, "Timeout not reached");

        uint256 amount = e.balance;
        e.balance = 0;
        payable(e.agent).transfer(amount);

        emit EscrowReclaimed(requestId, e.agent, amount);
    }
}
