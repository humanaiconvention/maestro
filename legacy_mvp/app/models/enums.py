import enum

class ActorType(str, enum.Enum):
    human = "human"
    agent = "agent"

class LivedExperienceAxis(str, enum.Enum):
    causal_reasoning = "causal_reasoning"
    context_anchoring = "context_anchoring"
    ethical_deliberation = "ethical_deliberation"
    interpretability_provenance = "interpretability_provenance"
    long_horizon_planning = "long_horizon_planning"
    playful_creativity = "playful_creativity"

class RequestStatus(str, enum.Enum):
    draft = "DRAFT"
    pending_viability = "PENDING_VIABILITY"
    open = "OPEN"
    in_progress = "IN_PROGRESS"
    review = "REVIEW"
    settled = "SETTLED"
    cancelled = "CANCELLED"

class ContributionStatus(str, enum.Enum):
    submitted = "SUBMITTED"
    verified = "VERIFIED"
    rejected = "REJECTED"
    review = "REVIEW"
    settled = "SETTLED"
    withdrawn = "WITHDRAWN"

class ViabilityDecision(str, enum.Enum):
    allow = "PASS"
    block = "FAIL"
    review = "REVIEW_REQUIRED"

class SettlementStatus(str, enum.Enum):
    pending = "PENDING"
    completed = "COMPLETED"
    failed = "FAILED"
    review = "REVIEW"

class RevocationStatus(str, enum.Enum):
    requested = "REQUESTED"
    completed = "COMPLETED"
    failed = "FAILED"

class PrivacyLevel(str, enum.Enum):
    standard = "standard"
    high = "high"
    maximum = "maximum"

class AuditEventType(str, enum.Enum):
    actor_created = "actor_created"
    consent_updated = "consent_updated"
    request_created = "request_created"
    request_evaluated = "request_evaluated"
    contribution_submitted = "contribution_submitted"
    contribution_evaluated = "contribution_evaluated"
    settlement_attempted = "settlement_attempted"
    revocation_requested = "revocation_requested"
    revocation_completed = "revocation_completed"
    operator_decision = "operator_decision"
    policy_loaded = "policy_loaded"  # emitted each time viability_policy.yaml is read; hash change signals drift
    api_key_created = "api_key_created"
    api_key_revoked = "api_key_revoked"

class PayoutMethod(str, enum.Enum):
    crypto = "crypto"
    fiat = "fiat"
    credit = "credit"

class PlanStatus(str, enum.Enum):
    draft = "DRAFT"
    submitted = "SUBMITTED"
    approved = "APPROVED"
    rejected = "REJECTED"
    active = "ACTIVE"
    completed = "COMPLETED"
    failed = "FAILED"
    revoked = "REVOKED"

class ActionStatus(str, enum.Enum):
    pending = "PENDING"
    in_progress = "IN_PROGRESS"
    completed = "COMPLETED"
    failed = "FAILED"

class ApprovalDecisionEnum(str, enum.Enum):
    approved = "APPROVED"
    rejected = "REJECTED"
    conditionally_approved = "CONDITIONALLY_APPROVED"

class LaneType(str, enum.Enum):
    empirical = "empirical"
    theory = "theory"
    literature = "literature"

class Verdict(str, enum.Enum):
    verified = "verified"
    partially_verified = "partially_verified"
    unresolved = "unresolved"
    contradicted = "contradicted"
    metaphorical_or_nonliteral = "metaphorical_or_nonliteral"
    not_applicable_to_lane = "not_applicable_to_lane"

class ReviewStatus(str, enum.Enum):
    pending = "PENDING"
    escalated = "ESCALATED"
    resolved = "RESOLVED"

class PolicyState(str, enum.Enum):
    allowed = "allowed"
    allowed_with_review = "allowed_with_review"
    blocked_pending_exogenous_check = "blocked_pending_exogenous_check"

class LearningRunStatus(str, enum.Enum):
    active = "ACTIVE"
    completed = "COMPLETED"
    failed = "FAILED"
    settlement_pending = "SETTLEMENT_PENDING"
    settled = "SETTLED"
    rolled_back = "ROLLED_BACK"
