from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    RootModel,
    StringConstraints,
    field_validator,
    model_validator,
)


PacketId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9._:-]+$")]
CovenantId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9._:-]+$")]
StandardId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9._:-]+$")]
UriString = Annotated[str, StringConstraints(min_length=1)]


class KnowledgeStatus(str, Enum):
    provisional = "provisional"
    active = "active"
    disputed = "disputed"
    superseded = "superseded"
    archived = "archived"


class Visibility(str, Enum):
    private = "private"
    shared = "shared"
    public = "public"
    agent_scoped = "agent_scoped"


class ContributorRole(str, Enum):
    author = "author"
    synthesizer = "synthesizer"
    reviewer = "reviewer"
    observer = "observer"
    operator = "operator"


class PacketKind(str, Enum):
    claim = "claim"
    evidence = "evidence"
    objection = "objection"
    revision = "revision"


class ConfidenceLabel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class AlignmentCommitment(str, Enum):
    honesty = "honesty"
    accuracy = "accuracy"
    reliability = "reliability"
    verifiability = "verifiability"
    falsifiability = "falsifiability"
    usefulness = "usefulness"
    scientific_grounding = "scientific_grounding"
    first_principles_reasoning = "first_principles_reasoning"


class AlignmentMode(str, Enum):
    geometric_orientation = "geometric_orientation"
    audit_signal = "audit_signal"
    hard_constraint = "hard_constraint"


class StandardStatus(str, Enum):
    draft = "draft"
    active = "active"
    retired = "retired"


class ArchitectureLayer(str, Enum):
    public_website = "public_website"
    public_docs = "public_docs"
    envoy_workspace = "envoy_workspace"
    private_envoy_runtime = "private_envoy_runtime"
    participation_covenant = "participation_covenant"
    knowledge_ledger = "knowledge_ledger"
    maestro_api = "maestro_api"
    viability_gates = "viability_gates"
    review_queue = "review_queue"
    audit_ledger = "audit_ledger"
    monitoring_metrics = "monitoring_metrics"


class IntegrityCommitment(str, Enum):
    bounded_consent = "bounded_consent"
    informed_disclosure = "informed_disclosure"
    transparent_methods = "transparent_methods"
    revocable_participation = "revocable_participation"
    proportional_safeguards = "proportional_safeguards"
    accountable_auditability = "accountable_auditability"
    non_extraction = "non_extraction"
    anti_capture = "anti_capture"
    anti_lock_in = "anti_lock_in"
    anti_proof_gaming = "anti_proof_gaming"
    anti_social_engineering = "anti_social_engineering"
    reciprocal_dignity = "reciprocal_dignity"


class RiskPatternKind(str, Enum):
    ambient_extraction = "ambient_extraction"
    coercive_dependency = "coercive_dependency"
    dark_pattern = "dark_pattern"
    lock_in_by_default = "lock_in_by_default"
    proof_by_charisma = "proof_by_charisma"
    proof_gaming = "proof_gaming"
    social_engineering = "social_engineering"
    identity_manipulation = "identity_manipulation"
    undisclosed_retention = "undisclosed_retention"
    engagement_steering = "engagement_steering"
    capture_pressure = "capture_pressure"


class EvidenceType(str, Enum):
    observation = "observation"
    experiment = "experiment"
    measurement = "measurement"
    citation = "citation"
    receipt = "receipt"
    trace = "trace"
    lived_report = "lived_report"


class ObjectionType(str, Enum):
    factual = "factual"
    methodological = "methodological"
    interpretive = "interpretive"
    scope = "scope"
    consent = "consent"


class ResolutionRequest(str, Enum):
    review = "review"
    revise = "revise"
    narrow_scope = "narrow_scope"
    retract = "retract"
    no_action = "no_action"


class CovenantStatus(str, Enum):
    draft = "draft"
    active = "active"
    paused = "paused"
    retired = "retired"


class ParticipantKind(str, Enum):
    human = "human"
    agent = "agent"
    service = "service"
    operator = "operator"


class EnforcementMode(str, Enum):
    runtime_block = "runtime_block"
    runtime_warn = "runtime_warn"
    audit_only = "audit_only"
    manual_review = "manual_review"


class ApprovalLevel(str, Enum):
    forbidden = "forbidden"
    explicit = "explicit"
    step_up = "step_up"
    auto = "auto"


class RetentionMode(str, Enum):
    ephemeral = "ephemeral"
    session = "session"
    bounded = "bounded"
    persistent = "persistent"


class CentralStorageMode(str, Enum):
    prohibited = "prohibited"
    verification_only = "verification_only"
    minimized = "minimized"
    bounded_raw = "bounded_raw"


class StoredArtifactClass(str, Enum):
    verification_token = "verification_token"
    cryptographic_attestation = "cryptographic_attestation"
    measurement_summary = "measurement_summary"
    optimization_metric = "optimization_metric"
    audit_receipt = "audit_receipt"
    consent_receipt = "consent_receipt"
    revocation_receipt = "revocation_receipt"
    lineage_pointer = "lineage_pointer"
    challenge_artifact = "challenge_artifact"
    raw_content = "raw_content"
    full_transcript = "full_transcript"
    model_training_corpus = "model_training_corpus"
    biometric_raw_capture = "biometric_raw_capture"
    plaintext_secret = "plaintext_secret"


class ProofKind(str, Enum):
    citation = "citation"
    receipt = "receipt"
    measurement = "measurement"
    human_confirmation = "human_confirmation"
    cryptographic_attestation = "cryptographic_attestation"


class RecoveryAction(str, Enum):
    pause = "pause"
    notify = "notify"
    request_reconsent = "request_reconsent"
    narrow_scope = "narrow_scope"
    human_review = "human_review"
    invalidate_outputs = "invalidate_outputs"
    quarantine_memory = "quarantine_memory"
    quarantine_tools = "quarantine_tools"
    emit_receipt = "emit_receipt"
    terminate = "terminate"


class FederatedCommitment(str, Enum):
    federated_exchange = "federated_exchange"
    distributed_control = "distributed_control"
    open_protocols = "open_protocols"
    transparent_verification = "transparent_verification"
    participant_data_sovereignty = "participant_data_sovereignty"
    verification_minimization = "verification_minimization"
    portable_receipts = "portable_receipts"
    continuous_revision = "continuous_revision"


class FederatedRiskPatternKind(str, Enum):
    central_payload_hoarding = "central_payload_hoarding"
    silent_payload_retention = "silent_payload_retention"
    operator_plaintext_capture = "operator_plaintext_capture"
    hidden_training_reuse = "hidden_training_reuse"
    opaque_verification = "opaque_verification"
    closed_protocol_dependency = "closed_protocol_dependency"
    unverifiable_optimization = "unverifiable_optimization"
    federation_in_name_only = "federation_in_name_only"


class SoftLaunchCommitment(str, Enum):
    calibration_not_governance = "calibration_not_governance"
    anti_founder_overweighting = "anti_founder_overweighting"
    anti_early_user_capture = "anti_early_user_capture"
    diversity_before_direction = "diversity_before_direction"
    delayed_adaptive_optimization = "delayed_adaptive_optimization"
    scheduled_review_only = "scheduled_review_only"
    bounded_influence = "bounded_influence"


class SoftLaunchRiskPatternKind(str, Enum):
    founder_capture = "founder_capture"
    early_user_capture = "early_user_capture"
    cohort_capture = "cohort_capture"
    coercive_dependence = "coercive_dependence"
    proof_gaming = "proof_gaming"
    social_engineering = "social_engineering"
    premature_governance = "premature_governance"
    adaptive_feedback_lock_in = "adaptive_feedback_lock_in"


class SoftLaunchPhase(str, Enum):
    seed = "seed"
    pilot = "pilot"
    governance_eligible = "governance_eligible"


class SoftLaunchInfluenceMode(str, Enum):
    calibration_only = "calibration_only"
    bounded_feedback = "bounded_feedback"
    governance_eligible = "governance_eligible"


class ContributorRef(BaseModel):
    actor_id: str
    role: ContributorRole
    display_name: Optional[str] = None


class SourceRef(BaseModel):
    source_id: str
    source_type: str
    locator: Optional[str] = None
    citation: Optional[str] = None
    integrity_hash: Optional[str] = None
    immutable: bool = False


class LedgerRef(BaseModel):
    ledger_id: str
    entry_id: str
    immutable: bool = True
    locator: Optional[str] = None


class PacketLinks(BaseModel):
    supports: List[PacketId] = Field(default_factory=list)
    challenges: List[PacketId] = Field(default_factory=list)
    revises: List[PacketId] = Field(default_factory=list)
    supersedes: List[PacketId] = Field(default_factory=list)
    related: List[PacketId] = Field(default_factory=list)


class ProvenanceEnvelope(BaseModel):
    contributors: List[ContributorRef] = Field(min_length=1)
    sources: List[SourceRef] = Field(default_factory=list)
    derived_from: List[PacketId] = Field(default_factory=list)
    created_by_mode: Literal["human", "agent", "hybrid"] = "human"


class ClaimScope(BaseModel):
    domain: str
    boundaries: List[str] = Field(default_factory=list)
    visibility: Visibility = Visibility.shared


class ConfidenceEstimate(BaseModel):
    label: ConfidenceLabel
    rationale: Optional[str] = None


class EpistemicAlignmentAxis(BaseModel):
    axis_id: str = "haic.axis.epistemic-grounding.v0"
    name: str = "epistemic_grounding"
    commitments: List[AlignmentCommitment] = Field(
        default_factory=lambda: [
            AlignmentCommitment.honesty,
            AlignmentCommitment.accuracy,
            AlignmentCommitment.reliability,
            AlignmentCommitment.verifiability,
            AlignmentCommitment.falsifiability,
            AlignmentCommitment.usefulness,
            AlignmentCommitment.scientific_grounding,
            AlignmentCommitment.first_principles_reasoning,
        ]
    )
    orientation: str = Field(min_length=1)
    mode: AlignmentMode = AlignmentMode.geometric_orientation
    signals: List[str] = Field(default_factory=list)
    anti_goals: List[str] = Field(default_factory=list)

    @field_validator("commitments")
    @classmethod
    def dedupe_commitments(cls, value: List[AlignmentCommitment]) -> List[AlignmentCommitment]:
        seen: list[AlignmentCommitment] = []
        for item in value:
            if item not in seen:
                seen.append(item)
        return seen


class StandardRef(BaseModel):
    standard_id: StandardId
    title: Optional[str] = None
    rationale: Optional[str] = None


class KnowledgePacketBase(BaseModel):
    schema_version: Literal["haic.knowledge_packet.v0"] = "haic.knowledge_packet.v0"
    packet_id: PacketId
    kind: PacketKind
    status: KnowledgeStatus = KnowledgeStatus.provisional
    created_at: datetime
    updated_at: datetime
    title: str = Field(min_length=1)
    summary: Optional[str] = None
    visibility: Visibility = Visibility.shared
    tags: List[str] = Field(default_factory=list)
    alignment_axis: Optional[EpistemicAlignmentAxis] = None
    provenance: ProvenanceEnvelope
    links: PacketLinks = Field(default_factory=PacketLinks)


class ClaimPacket(KnowledgePacketBase):
    kind: Literal[PacketKind.claim] = PacketKind.claim
    statement: str = Field(min_length=1)
    scope: ClaimScope
    confidence: ConfidenceEstimate
    update_conditions: List[str] = Field(min_length=1)
    supporting_packet_ids: List[PacketId] = Field(default_factory=list)
    challenge_packet_ids: List[PacketId] = Field(default_factory=list)


class EvidencePacket(KnowledgePacketBase):
    kind: Literal[PacketKind.evidence] = PacketKind.evidence
    evidence_type: EvidenceType
    observation: str = Field(min_length=1)
    method: str = Field(min_length=1)
    artifact_uri: Optional[UriString] = None
    captured_at: datetime
    integrity_hash: Optional[str] = None
    ledger_refs: List[LedgerRef] = Field(default_factory=list)
    target_packet_ids: List[PacketId] = Field(default_factory=list)


class ObjectionPacket(KnowledgePacketBase):
    kind: Literal[PacketKind.objection] = PacketKind.objection
    target_packet_id: PacketId
    objection_type: ObjectionType
    argument: str = Field(min_length=1)
    requested_resolution: ResolutionRequest = ResolutionRequest.review
    supporting_packet_ids: List[PacketId] = Field(default_factory=list)


class RevisionChange(BaseModel):
    field_path: str = Field(min_length=1)
    previous_value: Optional[Any] = None
    new_value: Optional[Any] = None
    reason: str = Field(min_length=1)


class RevisionPacket(KnowledgePacketBase):
    kind: Literal[PacketKind.revision] = PacketKind.revision
    target_packet_id: PacketId
    change_summary: str = Field(min_length=1)
    resulting_status: KnowledgeStatus
    changes: List[RevisionChange] = Field(min_length=1)
    supersedes_target: bool = True


KnowledgePacket = Annotated[
    Union[ClaimPacket, EvidencePacket, ObjectionPacket, RevisionPacket],
    Field(discriminator="kind"),
]


class KnowledgePacketDocument(RootModel[KnowledgePacket]):
    pass


class CovenantParticipant(BaseModel):
    participant_id: str
    kind: ParticipantKind
    role: str = Field(min_length=1)
    required: bool = True


class CovenantScope(BaseModel):
    task: str = Field(min_length=1)
    boundaries: List[str] = Field(min_length=1)
    visibility: Visibility = Visibility.shared
    environments: List[str] = Field(default_factory=list)


class CovenantRule(BaseModel):
    rule_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    applies_to_roles: List[str] = Field(default_factory=list)
    enforcement: EnforcementMode
    evidence_required: List[str] = Field(default_factory=list)


class CovenantGoal(BaseModel):
    goal_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    success_signals: List[str] = Field(default_factory=list)
    enforcement: EnforcementMode = EnforcementMode.audit_only


class ToolApprovalOverride(BaseModel):
    tool_name: str = Field(min_length=1)
    approval: ApprovalLevel
    reason: Optional[str] = None


class ApprovalPolicy(BaseModel):
    consequential_reads: ApprovalLevel = ApprovalLevel.explicit
    writes: ApprovalLevel = ApprovalLevel.explicit
    default_tool_invocation: ApprovalLevel = ApprovalLevel.explicit
    overrides: List[ToolApprovalOverride] = Field(default_factory=list)


class MemoryPolicy(BaseModel):
    retention_mode: RetentionMode
    max_retention_days: Optional[int] = Field(default=None, ge=0)
    allow_training: bool = False
    allow_cross_covenant_reuse: bool = False
    require_disclosure: bool = True
    central_storage_mode: CentralStorageMode = CentralStorageMode.minimized
    allowed_central_artifacts: List[StoredArtifactClass] = Field(default_factory=list)
    edge_exchange_required: bool = False
    allow_operator_plaintext_access: bool = False

    @field_validator("max_retention_days")
    @classmethod
    def validate_retention_days(cls, value: Optional[int], info) -> Optional[int]:
        retention_mode = info.data.get("retention_mode")
        if retention_mode == RetentionMode.bounded and value is None:
            raise ValueError("max_retention_days is required when retention_mode is bounded")
        return value

    @model_validator(mode="after")
    def validate_storage_boundary(self) -> "MemoryPolicy":
        raw_artifacts = {
            StoredArtifactClass.raw_content,
            StoredArtifactClass.full_transcript,
            StoredArtifactClass.model_training_corpus,
            StoredArtifactClass.biometric_raw_capture,
            StoredArtifactClass.plaintext_secret,
        }
        allowed = set(self.allowed_central_artifacts)

        if self.central_storage_mode == CentralStorageMode.prohibited and allowed:
            raise ValueError(
                "allowed_central_artifacts must be empty when central_storage_mode is prohibited"
            )

        if self.central_storage_mode == CentralStorageMode.verification_only:
            if allowed & raw_artifacts:
                raise ValueError(
                    "verification_only storage cannot allow raw content, transcripts, training corpora, "
                    "biometric raw capture, or plaintext secrets"
                )
            if not self.edge_exchange_required:
                raise ValueError(
                    "edge_exchange_required must be true when central_storage_mode is verification_only"
                )

        if self.central_storage_mode in {
            CentralStorageMode.prohibited,
            CentralStorageMode.verification_only,
        } and self.allow_operator_plaintext_access:
            raise ValueError(
                "allow_operator_plaintext_access cannot be true when central storage forbids or excludes raw payloads"
            )

        return self


class ProofRequirement(BaseModel):
    proof_id: str = Field(min_length=1)
    kind: ProofKind
    description: str = Field(min_length=1)
    required: bool = True


class DriftSignal(BaseModel):
    signal_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    threshold_hint: Optional[str] = None


class MonitoringPolicy(BaseModel):
    drift_signals: List[DriftSignal] = Field(default_factory=list)
    audit_event_types: List[str] = Field(default_factory=list)
    review_triggers: List[str] = Field(default_factory=list)


class RecoveryStep(BaseModel):
    action: RecoveryAction
    description: str = Field(min_length=1)


class RecoveryPolicy(BaseModel):
    on_violation: List[RecoveryStep] = Field(min_length=1)
    on_scope_change: List[RecoveryStep] = Field(default_factory=list)
    on_consent_withdrawal: List[RecoveryStep] = Field(default_factory=list)


class SoftLaunchThreshold(BaseModel):
    threshold_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    minimum: float = Field(ge=0)
    current: Optional[float] = Field(default=None, ge=0)
    required_for_governance: bool = True


class SoftLaunchPolicy(BaseModel):
    phase: SoftLaunchPhase = SoftLaunchPhase.seed
    participation_mode: SoftLaunchInfluenceMode = SoftLaunchInfluenceMode.calibration_only
    founder_input_mode: SoftLaunchInfluenceMode = SoftLaunchInfluenceMode.calibration_only
    early_human_input_mode: SoftLaunchInfluenceMode = SoftLaunchInfluenceMode.calibration_only
    early_agent_input_mode: SoftLaunchInfluenceMode = SoftLaunchInfluenceMode.calibration_only
    adaptive_optimization_enabled: bool = False
    scheduled_review_interval_days: int = Field(default=14, ge=1)
    max_influence_share_per_actor: float = Field(default=0.1, ge=0.0, le=1.0)
    max_influence_share_per_cluster: float = Field(default=0.25, ge=0.0, le=1.0)
    minimum_human_participants: int = Field(default=0, ge=0)
    minimum_agent_participants: int = Field(default=0, ge=0)
    minimum_distinct_cohorts: int = Field(default=0, ge=0)
    minimum_distinct_problem_domains: int = Field(default=0, ge=0)
    governance_unlock_thresholds: List[SoftLaunchThreshold] = Field(default_factory=list)
    directional_change_keywords: List[str] = Field(
        default_factory=lambda: [
            "governance",
            "policy",
            "default",
            "ranking",
            "roadmap",
            "platform direction",
            "incentive",
            "constitution",
            "weighting",
        ]
    )
    notes: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_phase_consistency(self) -> "SoftLaunchPolicy":
        if self.phase in {SoftLaunchPhase.seed, SoftLaunchPhase.pilot}:
            if self.participation_mode == SoftLaunchInfluenceMode.governance_eligible:
                raise ValueError("governance_eligible participation_mode is not allowed before governance_eligible phase")
            if self.founder_input_mode == SoftLaunchInfluenceMode.governance_eligible:
                raise ValueError("founder_input_mode cannot be governance_eligible before governance_eligible phase")
            if self.early_human_input_mode == SoftLaunchInfluenceMode.governance_eligible:
                raise ValueError("early_human_input_mode cannot be governance_eligible before governance_eligible phase")
            if self.early_agent_input_mode == SoftLaunchInfluenceMode.governance_eligible:
                raise ValueError("early_agent_input_mode cannot be governance_eligible before governance_eligible phase")
            if self.adaptive_optimization_enabled:
                raise ValueError("adaptive_optimization_enabled must remain false during seed and pilot phases")

        if self.phase == SoftLaunchPhase.governance_eligible:
            required_thresholds = [threshold for threshold in self.governance_unlock_thresholds if threshold.required_for_governance]
            if not required_thresholds:
                raise ValueError("governance_eligible phase requires at least one governance unlock threshold")

        return self


class ParticipationCovenant(BaseModel):
    schema_version: Literal["haic.participation_covenant.v0"] = "haic.participation_covenant.v0"
    covenant_id: CovenantId
    title: str = Field(min_length=1)
    status: CovenantStatus = CovenantStatus.draft
    purpose: str = Field(min_length=1)
    alignment_axis: Optional[EpistemicAlignmentAxis] = None
    standards: List[StandardRef] = Field(default_factory=list)
    participants: List[CovenantParticipant] = Field(min_length=1)
    scope: CovenantScope
    preconditions: List[CovenantRule] = Field(min_length=1)
    hard_invariants: List[CovenantRule] = Field(min_length=1)
    soft_invariants: List[CovenantRule] = Field(default_factory=list)
    hard_goals: List[CovenantGoal] = Field(min_length=1)
    soft_goals: List[CovenantGoal] = Field(default_factory=list)
    approvals: ApprovalPolicy
    memory_policy: MemoryPolicy
    proof_requirements: List[ProofRequirement] = Field(default_factory=list)
    monitoring: MonitoringPolicy = Field(default_factory=MonitoringPolicy)
    recovery: RecoveryPolicy
    soft_launch: Optional[SoftLaunchPolicy] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntegrityProhibition(BaseModel):
    prohibition_id: str = Field(min_length=1)
    kind: RiskPatternKind
    description: str = Field(min_length=1)
    detection_signals: List[str] = Field(default_factory=list)
    response_actions: List[str] = Field(default_factory=list)


class IntegrityControl(BaseModel):
    control_id: str = Field(min_length=1)
    commitment: IntegrityCommitment
    description: str = Field(min_length=1)
    layers: List[ArchitectureLayer] = Field(min_length=1)
    enforcement: EnforcementMode = EnforcementMode.audit_only
    verification_signals: List[str] = Field(default_factory=list)


class IntegrityDefault(BaseModel):
    default_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    layers: List[ArchitectureLayer] = Field(min_length=1)
    desired_state: str = Field(min_length=1)
    anti_goal: Optional[str] = None


class IntegrityMetric(BaseModel):
    metric_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    layers: List[ArchitectureLayer] = Field(min_length=1)
    target: str = Field(min_length=1)
    failure_trigger: Optional[str] = None


class ArchitectureStackEntry(BaseModel):
    layer: ArchitectureLayer
    intent: str = Field(min_length=1)
    required_artifacts: List[str] = Field(default_factory=list)
    required_controls: List[str] = Field(default_factory=list)
    review_questions: List[str] = Field(default_factory=list)


class ParticipationIntegrityStandard(BaseModel):
    schema_version: Literal["haic.participation_integrity_standard.v0"] = (
        "haic.participation_integrity_standard.v0"
    )
    standard_id: StandardId
    title: str = Field(min_length=1)
    status: StandardStatus = StandardStatus.draft
    purpose: str = Field(min_length=1)
    summary: Optional[str] = None
    commitments: List[IntegrityCommitment] = Field(min_length=1)
    prohibited_patterns: List[IntegrityProhibition] = Field(min_length=1)
    required_controls: List[IntegrityControl] = Field(min_length=1)
    defaults: List[IntegrityDefault] = Field(default_factory=list)
    accountability_metrics: List[IntegrityMetric] = Field(default_factory=list)
    architecture_stack: List[ArchitectureStackEntry] = Field(min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("commitments")
    @classmethod
    def dedupe_integrity_commitments(
        cls, value: List[IntegrityCommitment]
    ) -> List[IntegrityCommitment]:
        seen: list[IntegrityCommitment] = []
        for item in value:
            if item not in seen:
                seen.append(item)
        return seen


class SoftLaunchProhibition(BaseModel):
    prohibition_id: str = Field(min_length=1)
    kind: SoftLaunchRiskPatternKind
    description: str = Field(min_length=1)
    detection_signals: List[str] = Field(default_factory=list)
    response_actions: List[str] = Field(default_factory=list)


class SoftLaunchControl(BaseModel):
    control_id: str = Field(min_length=1)
    commitment: SoftLaunchCommitment
    description: str = Field(min_length=1)
    layers: List[ArchitectureLayer] = Field(min_length=1)
    enforcement: EnforcementMode = EnforcementMode.audit_only
    verification_signals: List[str] = Field(default_factory=list)


class SoftLaunchMetric(BaseModel):
    metric_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    target: str = Field(min_length=1)
    failure_trigger: Optional[str] = None


class SoftLaunchStandard(BaseModel):
    schema_version: Literal["haic.soft_launch_standard.v0"] = "haic.soft_launch_standard.v0"
    standard_id: StandardId
    title: str = Field(min_length=1)
    status: StandardStatus = StandardStatus.draft
    purpose: str = Field(min_length=1)
    summary: Optional[str] = None
    commitments: List[SoftLaunchCommitment] = Field(min_length=1)
    prohibited_patterns: List[SoftLaunchProhibition] = Field(min_length=1)
    required_controls: List[SoftLaunchControl] = Field(min_length=1)
    default_policy: SoftLaunchPolicy
    accountability_metrics: List[SoftLaunchMetric] = Field(default_factory=list)
    architecture_stack: List[ArchitectureStackEntry] = Field(min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("commitments")
    @classmethod
    def dedupe_soft_launch_commitments(
        cls, value: List[SoftLaunchCommitment]
    ) -> List[SoftLaunchCommitment]:
        seen: list[SoftLaunchCommitment] = []
        for item in value:
            if item not in seen:
                seen.append(item)
        return seen


class FederatedProhibition(BaseModel):
    prohibition_id: str = Field(min_length=1)
    kind: FederatedRiskPatternKind
    description: str = Field(min_length=1)
    detection_signals: List[str] = Field(default_factory=list)
    response_actions: List[str] = Field(default_factory=list)


class FederatedControl(BaseModel):
    control_id: str = Field(min_length=1)
    commitment: FederatedCommitment
    description: str = Field(min_length=1)
    layers: List[ArchitectureLayer] = Field(min_length=1)
    enforcement: EnforcementMode = EnforcementMode.audit_only
    verification_signals: List[str] = Field(default_factory=list)


class FederatedExchangeStandard(BaseModel):
    schema_version: Literal["haic.federated_exchange_standard.v0"] = (
        "haic.federated_exchange_standard.v0"
    )
    standard_id: StandardId
    title: str = Field(min_length=1)
    status: StandardStatus = StandardStatus.draft
    purpose: str = Field(min_length=1)
    summary: Optional[str] = None
    commitments: List[FederatedCommitment] = Field(min_length=1)
    prohibited_patterns: List[FederatedProhibition] = Field(min_length=1)
    required_controls: List[FederatedControl] = Field(min_length=1)
    allowed_system_artifacts: List[StoredArtifactClass] = Field(min_length=1)
    disallowed_system_artifacts: List[StoredArtifactClass] = Field(min_length=1)
    default_memory_policy: MemoryPolicy
    transparency_requirements: List[str] = Field(default_factory=list)
    evolution_requirements: List[str] = Field(default_factory=list)
    architecture_stack: List[ArchitectureStackEntry] = Field(min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("commitments")
    @classmethod
    def dedupe_federated_commitments(
        cls, value: List[FederatedCommitment]
    ) -> List[FederatedCommitment]:
        seen: list[FederatedCommitment] = []
        for item in value:
            if item not in seen:
                seen.append(item)
        return seen


def validate_knowledge_packet(payload: Dict[str, Any]) -> KnowledgePacket:
    return KnowledgePacketDocument.model_validate(payload).root


def validate_participation_covenant(payload: Dict[str, Any]) -> ParticipationCovenant:
    return ParticipationCovenant.model_validate(payload)


def validate_participation_integrity_standard(
    payload: Dict[str, Any]
) -> ParticipationIntegrityStandard:
    return ParticipationIntegrityStandard.model_validate(payload)


def validate_federated_exchange_standard(payload: Dict[str, Any]) -> FederatedExchangeStandard:
    return FederatedExchangeStandard.model_validate(payload)


def validate_soft_launch_standard(payload: Dict[str, Any]) -> SoftLaunchStandard:
    return SoftLaunchStandard.model_validate(payload)
