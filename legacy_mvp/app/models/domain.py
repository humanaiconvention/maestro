import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Enum, Float, JSON, Integer
from sqlalchemy.orm import relationship
from app.db import Base
from app.models.enums import (
    ActorType, LivedExperienceAxis, RequestStatus, ContributionStatus,
    ViabilityDecision, SettlementStatus, RevocationStatus, PrivacyLevel, AuditEventType, PayoutMethod,
    PlanStatus, ActionStatus, ApprovalDecisionEnum, LaneType, Verdict, ReviewStatus, PolicyState, LearningRunStatus
)
from sqlalchemy.types import TypeDecorator

class JSONType(TypeDecorator):
    impl = JSON
    cache_ok = True

def get_utc_now():
    return datetime.now(timezone.utc)

def gen_uuid():
    return str(uuid.uuid4())

class Actor(Base):
    __tablename__ = "actors"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    actor_type = Column(Enum(ActorType), nullable=False)
    display_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    consent_profile = relationship("ConsentProfile", back_populates="actor", uselist=False)
    requests = relationship("GroundingRequest", back_populates="agent")
    contributions = relationship("Contribution", back_populates="human")

class ConsentProfile(Base):
    __tablename__ = "consent_profiles"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    actor_id = Column(String(36), ForeignKey("actors.id"), nullable=False, unique=True)

    allowed_axes = Column(JSONType, default=list) # List of LivedExperienceAxis
    max_task_duration_minutes = Column(Integer, default=60)
    privacy_level = Column(Enum(PrivacyLevel), default=PrivacyLevel.standard)
    allow_model_training = Column(Boolean, default=True)
    allow_public_aggregate = Column(Boolean, default=False)

    min_hourly_equivalent_usd = Column(Float, default=50.0)
    payout_threshold_usd = Column(Float, default=25.0)
    payout_method = Column(Enum(PayoutMethod), default=PayoutMethod.crypto)

    daily_max_contributions = Column(Integer, default=10)
    weekly_max_hours = Column(Integer, default=8)

    require_full_provenance = Column(Boolean, default=True)
    revocation_window_days = Column(Integer, default=90)

    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    actor = relationship("Actor", back_populates="consent_profile")

class GroundingRequest(Base):
    __tablename__ = "grounding_requests"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    agent_id = Column(String(36), ForeignKey("actors.id"), nullable=False)

    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    lived_experience_axis = Column(Enum(LivedExperienceAxis), nullable=False)
    requested_modality = Column(String, nullable=False)
    protocol_instructions = Column(String, nullable=False)
    estimated_duration_minutes = Column(Integer, nullable=False)

    compensation_amount = Column(Float, nullable=False)
    effective_hourly_estimate = Column(Float, nullable=False)

    entropy_reduction_hypothesis = Column(String, nullable=False)
    phase_synchronization_hypothesis = Column(String, nullable=False)
    evidence_artifact_description = Column(String, nullable=False)

    contributions_required = Column(Integer, default=50)
    viability_policy_snapshot = Column(JSONType, nullable=True)

    status = Column(Enum(RequestStatus), default=RequestStatus.draft)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    agent = relationship("Actor", back_populates="requests")
    contributions = relationship("Contribution", back_populates="request")

class Contribution(Base):
    __tablename__ = "contributions"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    human_id = Column(String(36), ForeignKey("actors.id"), nullable=False)
    request_id = Column(String(36), ForeignKey("grounding_requests.id"), nullable=False)

    content_payload = Column(JSONType, nullable=False) # The actual observation
    consent_snapshot = Column(JSONType, nullable=False)

    status = Column(Enum(ContributionStatus), default=ContributionStatus.submitted)
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    human = relationship("Actor", back_populates="contributions")
    request = relationship("GroundingRequest", back_populates="contributions")
    provenance = relationship("ProvenanceRecord", back_populates="contribution", uselist=False)

class ProvenanceRecord(Base):
    __tablename__ = "provenance_records"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    contribution_id = Column(String(36), ForeignKey("contributions.id"), nullable=False, unique=True)
    human_id = Column(String(36), ForeignKey("actors.id"), nullable=False)
    request_id = Column(String(36), ForeignKey("grounding_requests.id"), nullable=False)

    integrity_hash = Column(String, nullable=False)
    creation_context = Column(JSONType, nullable=False) # e.g. telemetry metrics, timestamps
    attestation_bundle = Column(JSONType, nullable=True) # Phase D: cryptographic proof payload
    transformation_log = Column(JSONType, default=list)
    withdrawal_status = Column(Enum(ContributionStatus), default=ContributionStatus.submitted)
    usage_rights = Column(JSONType, nullable=False)
    expiration_date = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=get_utc_now)

    contribution = relationship("Contribution", back_populates="provenance")

class ViabilityEvaluation(Base):
    __tablename__ = "viability_evaluations"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    entity_type = Column(String, nullable=False) # "request", "contribution", "settlement"
    entity_id = Column(String(36), nullable=False)

    overall_viability = Column(Boolean, nullable=False)
    status = Column(Enum(ViabilityDecision), nullable=False)
    gates_evaluated = Column(JSONType, nullable=False)
    decision_rationale = Column(String, nullable=True)
    evaluation_hash = Column(String, nullable=True)

    created_at = Column(DateTime, default=get_utc_now)

class LearningRun(Base):
    __tablename__ = "learning_runs"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    title = Column(String, nullable=False)
    objective = Column(String, nullable=False)
    hypothesis_or_goal = Column(String, nullable=True)
    environment = Column(String, nullable=True)
    model_or_agent_version = Column(String, nullable=True)
    result_classification = Column(String, nullable=True)
    start_time = Column(DateTime, default=get_utc_now)
    end_time = Column(DateTime, nullable=True)
    status = Column(Enum(LearningRunStatus), default=LearningRunStatus.active)
    
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

    plans = relationship("Plan", back_populates="run")
    actions = relationship("ActionReceipt", back_populates="run")
    approvals = relationship("ApprovalDecision", back_populates="run")
    verifications = relationship("VerificationResult", back_populates="run")
    rollbacks = relationship("RollbackRecord", back_populates="run")
    settlements = relationship("Settlement", back_populates="learning_run")

class Plan(Base):
    __tablename__ = "plans"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    run_id = Column(String(36), ForeignKey("learning_runs.id"), nullable=False)
    title = Column(String, nullable=False)
    objective = Column(String, nullable=False)
    scope = Column(String, nullable=True)
    assumptions = Column(JSONType, nullable=True)
    steps = Column(JSONType, nullable=True)
    risk_level = Column(String, nullable=True)
    created_by = Column(String, nullable=True)
    status = Column(Enum(PlanStatus), default=PlanStatus.draft)
    
    created_at = Column(DateTime, default=get_utc_now)
    
    run = relationship("LearningRun", back_populates="plans")
    actions = relationship("ActionReceipt", back_populates="plan")

class ActionReceipt(Base):
    __tablename__ = "action_receipts"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    run_id = Column(String(36), ForeignKey("learning_runs.id"), nullable=False)
    plan_id = Column(String(36), ForeignKey("plans.id"), nullable=True)
    action_type = Column(String, nullable=False)
    actor = Column(String, nullable=True)
    input_refs = Column(JSONType, nullable=True)
    output_refs = Column(JSONType, nullable=True)
    environment = Column(String, nullable=True)
    started_at = Column(DateTime, default=get_utc_now)
    completed_at = Column(DateTime, nullable=True)
    status = Column(Enum(ActionStatus), default=ActionStatus.pending)
    provenance_bundle = Column(JSONType, nullable=True)

    run = relationship("LearningRun", back_populates="actions")
    plan = relationship("Plan", back_populates="actions")
    rollbacks = relationship("RollbackRecord", back_populates="action")

class ApprovalDecision(Base):
    __tablename__ = "approval_decisions"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    run_id = Column(String(36), ForeignKey("learning_runs.id"), nullable=False)
    target_ref = Column(String(36), nullable=False)
    approver = Column(String, nullable=True)
    approval_scope = Column(String, nullable=True)
    decision = Column(Enum(ApprovalDecisionEnum), nullable=False)
    rationale = Column(String, nullable=True)
    conditions = Column(JSONType, nullable=True)
    
    created_at = Column(DateTime, default=get_utc_now)

    run = relationship("LearningRun", back_populates="approvals")

class VerificationResult(Base):
    __tablename__ = "verification_results"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    run_id = Column(String(36), ForeignKey("learning_runs.id"), nullable=False)
    target_ref = Column(String(36), nullable=True)
    claim_id = Column(String, nullable=True)
    claim_text = Column(String, nullable=False)
    claim_type = Column(String, nullable=True)
    lane = Column(Enum(LaneType), nullable=False)
    verdict = Column(Enum(Verdict), nullable=False)
    confidence = Column(Float, nullable=True)
    rationale = Column(String, nullable=True)
    scope = Column(String, nullable=True)
    assumptions = Column(JSONType, nullable=True)
    evidence_items = Column(JSONType, nullable=True)
    evidence_refs = Column(JSONType, nullable=True)
    provenance_bundle = Column(JSONType, nullable=True)
    
    review_status = Column(Enum(ReviewStatus), default=ReviewStatus.pending)
    correlated_verifier_risk = Column(Float, nullable=True)
    policy_state = Column(Enum(PolicyState), nullable=True)
    
    lane_specific_record = Column(JSONType, nullable=True)

    created_at = Column(DateTime, default=get_utc_now)

    run = relationship("LearningRun", back_populates="verifications")

class RollbackRecord(Base):
    __tablename__ = "rollback_records"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    run_id = Column(String(36), ForeignKey("learning_runs.id"), nullable=False)
    action_id = Column(String(36), ForeignKey("action_receipts.id"), nullable=True)
    prior_state_ref = Column(String, nullable=True)
    rollback_trigger = Column(String, nullable=True)
    executed_by = Column(String, nullable=True)
    executed_at = Column(DateTime, default=get_utc_now)
    outcome = Column(String, nullable=True)

    run = relationship("LearningRun", back_populates="rollbacks")
    action = relationship("ActionReceipt", back_populates="rollbacks")

class Settlement(Base):
    __tablename__ = "settlements"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    request_id = Column(String(36), ForeignKey("grounding_requests.id"), nullable=True)
    learning_run_id = Column(String(36), ForeignKey("learning_runs.id"), nullable=True)

    learning_run = relationship("LearningRun", back_populates="settlements")

    epistemic_value_proof = Column(JSONType, nullable=False)
    viability_proof_hash = Column(String, nullable=False)

    status = Column(Enum(SettlementStatus), default=SettlementStatus.pending)
    total_payout = Column(Float, nullable=False)
    contributions_settled = Column(Integer, nullable=False)

    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

class RevocationRequest(Base):
    __tablename__ = "revocation_requests"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    contribution_id = Column(String(36), ForeignKey("contributions.id"), nullable=False)
    human_id = Column(String(36), ForeignKey("actors.id"), nullable=False)

    reason = Column(String, nullable=True)
    status = Column(Enum(RevocationStatus), default=RevocationStatus.requested)

    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

class ReviewQueueItem(Base):
    __tablename__ = "review_queue"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    entity_type = Column(String, nullable=False) # request, contribution, settlement
    entity_id = Column(String(36), nullable=False)

    reason = Column(String, nullable=False)
    evidence_links = Column(JSONType, nullable=True)

    status = Column(String, default="pending") # pending, resolved
    operator_id = Column(String(36), ForeignKey("actors.id"), nullable=True)
    action_taken = Column(String, nullable=True)
    rationale = Column(String, nullable=True)

    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)

class AuditEvent(Base):
    __tablename__ = "audit_events"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    event_type = Column(Enum(AuditEventType), nullable=False)
    actor_id = Column(String(36), nullable=True)
    entity_id = Column(String(36), nullable=True)
    metadata_json = Column(JSONType, nullable=True)
    created_at = Column(DateTime, default=get_utc_now)


class ApiKey(Base):
    """Per-agent API key.  The raw key is shown once at creation and never stored;
    only the SHA-256 hex digest is persisted so a compromised database reveals no
    usable secrets.  The master env-var key continues to work as a fallback.
    """
    __tablename__ = "api_keys"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    agent_id = Column(String(36), ForeignKey("actors.id"), nullable=False)
    key_hash = Column(String(64), nullable=False, unique=True)  # SHA-256 hex of raw key
    label = Column(String, nullable=True)
    created_at = Column(DateTime, default=get_utc_now)
    revoked_at = Column(DateTime, nullable=True)

    agent = relationship("Actor", foreign_keys=[agent_id])
