// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title GroundingRequest
 * @notice On-chain registry of human-grounding requests posted by AI agents.
 *
 * STUB — not production-ready.  This contract mirrors the off-chain
 * GroundingRequest model in legacy_mvp and will be wired to the settlement
 * ledger once the ZK-proof pipeline is production-ready.
 *
 * Lifecycle:
 *   1. Agent calls createRequest() → RequestStatus.Open
 *   2. Off-chain viability engine evaluates the request
 *   3. Operator/oracle transitions status via updateStatus()
 *   4. Request is closed by SettlementLedger after payout
 */
contract GroundingRequest {
    enum RequestStatus { Draft, Open, InProgress, Review, Settled, Cancelled }

    struct Request {
        bytes32 id;
        address agent;
        string  livedExperienceAxis;
        uint256 compensationWei;
        uint32  contributionsRequired;
        RequestStatus status;
        uint256 createdAt;
    }

    mapping(bytes32 => Request) public requests;
    address public owner;

    event RequestCreated(bytes32 indexed id, address indexed agent, string axis);
    event RequestStatusUpdated(bytes32 indexed id, RequestStatus newStatus);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function createRequest(
        bytes32 id,
        string calldata axis,
        uint32 contributionsRequired
    ) external payable {
        require(requests[id].createdAt == 0, "ID already exists");
        requests[id] = Request({
            id: id,
            agent: msg.sender,
            livedExperienceAxis: axis,
            compensationWei: msg.value,
            contributionsRequired: contributionsRequired,
            status: RequestStatus.Draft,
            createdAt: block.timestamp
        });
        emit RequestCreated(id, msg.sender, axis);
    }

    function updateStatus(bytes32 id, RequestStatus newStatus) external onlyOwner {
        require(requests[id].createdAt != 0, "Request not found");
        requests[id].status = newStatus;
        emit RequestStatusUpdated(id, newStatus);
    }
}
