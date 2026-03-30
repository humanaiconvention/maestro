from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from app.models.enums import (
    ActorType, LivedExperienceAxis, RequestStatus, ContributionStatus,
    ViabilityDecision, SettlementStatus, RevocationStatus, PrivacyLevel, AuditEventType, PayoutMethod,
    LearningRunStatus, PlanStatus, ActionStatus, ApprovalDecisionEnum, LaneType, Verdict, ReviewStatus, PolicyState
)

class ActorBase(BaseModel):
    display_name: Optional[str] = None

class HumanCreate(ActorBase):
    pass

class AgentCreate(ActorBase):
    pass

class ActorResponse(ActorBase):
    id: str
    actor_type: ActorType
    created_at: datetime

class ConsentProfileUpdate(BaseModel):
    allowed_axes: List[LivedExperienceAxis]
    max_task_duration_minutes: int
    privacy_level: PrivacyLevel
    allow_model_training: bool
    allow_public_aggregate: bool
    min_hourly_equivalent_usd: float
    payout_threshold_usd: float
    payout_method: PayoutMethod = PayoutMethod.crypto
    daily_max_contributions: int
    weekly_max_hours: int
    require_full_provenance: bool
    revocation_window_days: int

class ConsentProfileResponse(ConsentProfileUpdate):
    id: str
    actor_id: str

class GroundingRequestCreate(BaseModel):
    agent_id: str
    title: str
    description: str
    lived_experience_axis: LivedExperienceAxis
    requested_modality: str
    protocol_instructions: str
    estimated_duration_minutes: int
    compensation_amount: float
    effective_hourly_estimate: float
    entropy_reduction_hypothesis: str
    phase_synchronization_hypothesis: str
    evidence_artifact_description: str
    contributions_required: int
    viability_policy_snapshot: Dict[str, Any]

class GroundingRequestResponse(GroundingRequestCreate):
    id: str
    status: RequestStatus
    created_at: datetime

class TLSNotaryProof(BaseModel):
    session_id: str
    timestamp_utc: float
    server_name: str
    notary_signature: str

class EnclaveAttestation(BaseModel):
    enclave_id: str
    quote: str
    pcr_values: Dict[str, str]
    nonce: str

class CryptoAttestationBundle(BaseModel):
    """Phase D: cryptographic attestation payload attached to a contribution.

    Both fields are optional — the attestation pipeline runs regardless and
    awards a score bonus for each valid proof method present.
    """
    tlsnotary_proof: Optional[TLSNotaryProof] = None
    enclave_attestation: Optional[EnclaveAttestation] = None

class ContributionCreate(BaseModel):
    human_id: str
    request_id: str
    content_payload: Dict[str, Any]
    # Provenance data
    integrity_hash: str
    creation_context: Dict[str, Any]
    # Phase D: optional multi-factor cryptographic attestation
    attestation_bundle: Optional[CryptoAttestationBundle] = None

class ContributionResponse(BaseModel):
    id: str
    human_id: str
    request_id: str
    status: ContributionStatus
    created_at: datetime

class ViabilityEvaluationResult(BaseModel):
    status: str
    overall_viability: bool
    entity_type: str
    entity_id: str
    gates_evaluated: List[Dict[str, Any]]
    decision_rationale: str
    evaluation_hash: str

class SettlementAttempt(BaseModel):
    prior_distribution: Dict[str, float]
    posterior_distribution: Dict[str, float]
    observation_count: int
    claimed_entropy_reduction: float
    # Simulated ledger transfer
    total_payout: float

class RevocationCreate(BaseModel):
    human_id: str
    contribution_id: str
    reason: Optional[str] = None

class RevocationResponse(RevocationCreate):
    id: str
    status: RevocationStatus

class ReviewQueueResponse(BaseModel):
    id: str
    entity_type: str
    entity_id: str
    reason: str
    evidence_links: Optional[Dict[str, Any]] = None
    status: str
    created_at: datetime

class AuditEventResponse(BaseModel):
    id: str
    event_type: AuditEventType
    actor_id: Optional[str]
    entity_id: Optional[str]
    metadata_json: Optional[Dict[str, Any]]
    created_at: datetime


# ── Agentic Run Schemas ──────────────────────────────────────────────

class LearningRunCreate(BaseModel):
    title: str
    objective: str
    hypothesis_or_goal: Optional[str] = None
    environment: Optional[str] = None
    model_or_agent_version: Optional[str] = None

class LearningRunResponse(BaseModel):
    id: str
    title: str
    objective: str
    hypothesis_or_goal: Optional[str] = None
    environment: Optional[str] = None
    model_or_agent_version: Optional[str] = None
    status: LearningRunStatus
    created_at: datetime

class LearningRunDetailResponse(LearningRunResponse):
    plans: List["PlanResponse"] = []
    actions: List["ActionReceiptResponse"] = []
    approvals: List["ApprovalDecisionResponse"] = []
    verifications: List["VerificationResponse"] = []
    rollbacks: List["RollbackRecordResponse"] = []

class PlanCreate(BaseModel):
    title: str
    objective: str
    scope: Optional[str] = None
    assumptions: Optional[List[str]] = None
    steps: Optional[List[str]] = None
    risk_level: Optional[str] = None
    created_by: Optional[str] = None

class PlanResponse(BaseModel):
    id: str
    run_id: str
    title: str
    objective: str
    status: PlanStatus
    created_at: datetime

class ActionReceiptCreate(BaseModel):
    plan_id: Optional[str] = None
    action_type: str
    actor: Optional[str] = None
    input_refs: Optional[Dict[str, Any]] = None
    output_refs: Optional[Dict[str, Any]] = None
    environment: Optional[str] = None

class ActionReceiptResponse(BaseModel):
    id: str
    run_id: str
    plan_id: Optional[str] = None
    action_type: str
    status: ActionStatus
    started_at: datetime

class ApprovalDecisionCreate(BaseModel):
    target_ref: str
    approver: Optional[str] = None
    approval_scope: Optional[str] = None
    decision: ApprovalDecisionEnum
    rationale: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None

class ApprovalDecisionResponse(BaseModel):
    id: str
    run_id: str
    target_ref: str
    decision: ApprovalDecisionEnum
    rationale: Optional[str] = None
    created_at: datetime

class VerificationRequest(BaseModel):
    claim_text: str
    lane: str
    evidence: Dict[str, Any] = {}
    endogenous_only: bool = False
    target_ref: Optional[str] = None

class VerificationResponse(BaseModel):
    id: Optional[str] = None
    run_id: Optional[str] = None
    claim_text: Optional[str] = None
    lane: LaneType
    verdict: Verdict
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    review_status: ReviewStatus
    correlated_verifier_risk: Optional[float] = None
    policy_state: Optional[str] = None
    provenance_bundle: Optional[Dict[str, Any]] = None

class VerificationStatusPatch(BaseModel):
    review_status: Optional[ReviewStatus] = None
    policy_state: Optional[str] = None
    rationale: Optional[str] = None

class RollbackRecordCreate(BaseModel):
    action_id: Optional[str] = None
    prior_state_ref: Optional[str] = None
    rollback_trigger: Optional[str] = None
    executed_by: Optional[str] = None

class RollbackRecordResponse(BaseModel):
    id: str
    run_id: str
    action_id: Optional[str] = None
    prior_state_ref: Optional[str] = None
    rollback_trigger: Optional[str] = None
    executed_at: datetime


# ── API Key Schemas ──────────────────────────────────────────────────

class ApiKeyCreate(BaseModel):
    agent_id: str
    label: Optional[str] = None

class ApiKeyResponse(BaseModel):
    id: str
    agent_id: str
    label: Optional[str] = None
    created_at: datetime
    revoked_at: Optional[datetime] = None
    # raw_key is only populated on initial creation and never stored
    raw_key: Optional[str] = None


# ── Pagination ───────────────────────────────────────────────────────

class PageParams(BaseModel):
    """Common pagination query parameters."""
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


