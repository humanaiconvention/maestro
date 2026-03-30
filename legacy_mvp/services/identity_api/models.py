import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Enum, JSON, Integer, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
from sqlalchemy.orm import relationship
from database import Base
import enum

# Use generic JSON if PostgreSQL JSONB isn't available (e.g., in SQLite tests)
from sqlalchemy.types import TypeDecorator
class JSONType(TypeDecorator):
    impl = JSON
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(JSON())

class ActorType(str, enum.Enum):
    human = 'human'
    agent = 'agent'
    org_service = 'org_service'
    operator = 'operator'

class ActorStatus(str, enum.Enum):
    active = 'active'
    pending = 'pending'
    suspended = 'suspended'
    deleted = 'deleted'

class AssuranceLevel(str, enum.Enum):
    aal1 = 'aal1'
    aal2 = 'aal2'
    aal3 = 'aal3'

class ProviderType(str, enum.Enum):
    Maestro_local = 'Maestro_local'
    passkey = 'passkey'
    email_link = 'email_link'
    oauth_oidc = 'oauth_oidc'
    saml = 'saml'
    api_key = 'api_key'
    client_credentials = 'client_credentials'
    private_key_jwt = 'private_key_jwt'
    wallet_sig = 'wallet_sig'

class OnboardingState(str, enum.Enum):
    new = 'new'
    basic = 'basic'
    passkey_prompted = 'passkey_prompted'
    passkey_enrolled = 'passkey_enrolled'
    complete = 'complete'

class ParticipationState(str, enum.Enum):
    eligible = 'eligible'
    ineligible = 'ineligible'
    suspended = 'suspended'

class TrustTier(str, enum.Enum):
    sandbox = 'sandbox'
    basic = 'basic'
    verified = 'verified'
    elevated = 'elevated'

class AuthEventType(str, enum.Enum):
    login_success = 'login_success'
    login_failure = 'login_failure'
    magic_link_sent = 'magic_link_sent'
    magic_link_consumed = 'magic_link_consumed'
    passkey_registered = 'passkey_registered'
    passkey_login = 'passkey_login'
    provider_link_started = 'provider_link_started'
    provider_link_completed = 'provider_link_completed'
    provider_link_failed = 'provider_link_failed'
    provider_unlinked = 'provider_unlinked'
    recovery_started = 'recovery_started'
    recovery_completed = 'recovery_completed'
    agent_registered = 'agent_registered'
    token_issued = 'token_issued'

class RequestStatus(str, enum.Enum):
    draft = 'draft'
    open = 'open'
    fulfilled = 'fulfilled'
    cancelled = 'cancelled'

class ContributionStatus(str, enum.Enum):
    submitted = 'submitted'
    verified = 'verified'
    rejected = 'rejected'
    settled = 'settled'
    withdrawn = 'withdrawn'

class ViabilityDecision(str, enum.Enum):
    allow = 'allow'
    block = 'block'
    review = 'review'

class SettlementStatus(str, enum.Enum):
    pending = 'pending'
    completed = 'completed'
    failed = 'failed'

class Actor(Base):
    __tablename__ = "actor"

    actor_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    actor_type = Column(Enum(ActorType), nullable=False, index=True)
    display_name = Column(String, nullable=True)
    primary_handle = Column(String, unique=True, nullable=True)
    status = Column(Enum(ActorStatus), default=ActorStatus.active, index=True)
    assurance_level = Column(Enum(AssuranceLevel), default=AssuranceLevel.aal1)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_seen_at = Column(DateTime(timezone=True), nullable=True)

    bindings = relationship("CredentialBinding", back_populates="actor", cascade="all, delete-orphan")
    human_profile = relationship("HumanProfile", back_populates="actor", uselist=False, cascade="all, delete-orphan")
    agent_profile = relationship("AgentProfile", foreign_keys="[AgentProfile.actor_id]", back_populates="actor", uselist=False, cascade="all, delete-orphan")

class CredentialBinding(Base):
    __tablename__ = "credential_binding"

    binding_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    actor_id = Column(String(36), ForeignKey("actor.actor_id"), index=True, nullable=False)
    provider_type = Column(Enum(ProviderType), nullable=False)
    provider_name = Column(String, nullable=False)
    external_subject = Column(String, nullable=True)
    external_email = Column(String, nullable=True, index=True)
    external_username = Column(String, nullable=True)
    public_key_thumbprint = Column(String, nullable=True)
    credential_label = Column(String, nullable=True)
    metadata_json = Column(JSONType, nullable=True)
    verified_at = Column(DateTime(timezone=True), nullable=True)
    linked_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_used_at = Column(DateTime(timezone=True), nullable=True, index=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)

    actor = relationship("Actor", back_populates="bindings")

class HumanProfile(Base):
    __tablename__ = "human_profile"

    actor_id = Column(String(36), ForeignKey("actor.actor_id"), primary_key=True)
    email = Column(String, nullable=True, index=True)
    email_verified = Column(Boolean, default=False)
    public_handle = Column(String, unique=True, nullable=True)
    locale = Column(String, nullable=True)
    participation_state = Column(Enum(ParticipationState), default=ParticipationState.eligible)
    consent_profile_id = Column(String(36), ForeignKey("consent_scope.consent_scope_id"), nullable=True)
    recovery_state = Column(JSONType, nullable=True)
    provenance_visibility_preferences = Column(JSONType, nullable=True)
    onboarding_state = Column(Enum(OnboardingState), default=OnboardingState.new)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    actor = relationship("Actor", back_populates="human_profile")

class AgentProfile(Base):
    __tablename__ = "agent_profile"

    actor_id = Column(String(36), ForeignKey("actor.actor_id"), primary_key=True)
    operator_actor_id = Column(String(36), ForeignKey("actor.actor_id"), nullable=True)
    software_name = Column(String, nullable=False)
    software_version = Column(String, nullable=True)
    trust_tier = Column(Enum(TrustTier), default=TrustTier.sandbox)
    allowed_request_classes = Column(JSONType, nullable=True)
    funding_profile_id = Column(String(36), nullable=True)
    jwks_uri = Column(String, nullable=True)
    webhook_url = Column(String, nullable=True)
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    actor = relationship("Actor", foreign_keys=[actor_id], back_populates="agent_profile")

class ConsentScope(Base):
    __tablename__ = "consent_scope"

    consent_scope_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    actor_id = Column(String(36), ForeignKey("actor.actor_id"), nullable=False)
    permitted_request_classes = Column(JSONType, nullable=True)
    prohibited_request_classes = Column(JSONType, nullable=True)
    modality_permissions = Column(JSONType, nullable=True)
    retention_preferences = Column(JSONType, nullable=True)
    revocation_policy = Column(JSONType, nullable=True)
    visibility_policy = Column(JSONType, nullable=True)
    compensation_preferences = Column(JSONType, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class GroundingRequest(Base):
    __tablename__ = "grounding_request"

    request_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_actor_id = Column(String(36), ForeignKey("actor.actor_id"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    request_class = Column(String, nullable=False)
    hypothesis_entropy_reduction = Column(Float, nullable=False)
    hypothesis_phase_sync = Column(String, nullable=True)
    success_signal_type = Column(String, nullable=True)
    evidence_artifact_type = Column(String, nullable=True)
    requested_modalities = Column(JSONType, nullable=True)
    target_population_description = Column(String, nullable=True)
    geographic_scope = Column(String, nullable=True)
    compensation_budget = Column(Float, nullable=False)
    settlement_policy_id = Column(String(36), nullable=True)
    viability_policy_id = Column(String(36), nullable=True)
    status = Column(Enum(RequestStatus), default=RequestStatus.draft)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class ExperienceContribution(Base):
    __tablename__ = "experience_contribution"

    contribution_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    actor_id = Column(String(36), ForeignKey("actor.actor_id"), nullable=False)
    request_id = Column(String(36), ForeignKey("grounding_request.request_id"), nullable=False)
    modality = Column(String, nullable=False)
    content_pointer = Column(String, nullable=False)
    structured_metadata = Column(JSONType, nullable=True)
    contribution_status = Column(Enum(ContributionStatus), default=ContributionStatus.submitted)
    submitted_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    withdrawn_at = Column(DateTime(timezone=True), nullable=True)

class ProvenanceRecord(Base):
    __tablename__ = "provenance_record"

    provenance_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    contribution_id = Column(String(36), ForeignKey("experience_contribution.contribution_id"), nullable=False)
    consent_scope_id = Column(String(36), ForeignKey("consent_scope.consent_scope_id"), nullable=False)
    capture_context = Column(JSONType, nullable=False)
    device_context = Column(JSONType, nullable=True)
    transformation_chain = Column(JSONType, nullable=True)
    review_events = Column(JSONType, nullable=True)
    signatures = Column(JSONType, nullable=False)
    provenance_status = Column(String, default='verified')
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ViabilityEvaluation(Base):
    __tablename__ = "viability_evaluation"

    evaluation_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(String(36), ForeignKey("grounding_request.request_id"), nullable=True)
    contribution_id = Column(String(36), ForeignKey("experience_contribution.contribution_id"), nullable=True)
    gate_results = Column(JSONType, nullable=False)
    harm_exclusion_result = Column(Boolean, nullable=False)
    extraction_risk_result = Column(Boolean, nullable=False)
    consent_validity_result = Column(Boolean, nullable=False)
    provenance_validity_result = Column(Boolean, nullable=False)
    sustainability_result = Column(Boolean, nullable=False)
    final_decision = Column(Enum(ViabilityDecision), nullable=False)
    explanation = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class SettlementEvent(Base):
    __tablename__ = "settlement_event"

    settlement_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(String(36), ForeignKey("grounding_request.request_id"), nullable=False)
    agent_actor_id = Column(String(36), ForeignKey("actor.actor_id"), nullable=False)
    contribution_id = Column(String(36), ForeignKey("experience_contribution.contribution_id"), nullable=True)
    settlement_type = Column(String, nullable=False)
    gross_amount = Column(Float, nullable=False)
    platform_fee = Column(Float, nullable=False)
    human_amount = Column(Float, nullable=False)
    holdback_amount = Column(Float, default=0.0)
    
    # Evidence & Hypothesis fields for thermodynamic settlement
    claimed_outcome_summary = Column(String, nullable=True)
    evidence_artifact_metadata = Column(JSONType, nullable=True)
    confidence_level = Column(Float, nullable=True)
    settlement_justification_text = Column(String, nullable=True)

    status = Column(Enum(SettlementStatus), default=SettlementStatus.pending)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class AuthEvent(Base):
    __tablename__ = "auth_event"

    event_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    actor_id = Column(String(36), nullable=True, index=True)
    binding_id = Column(String(36), nullable=True)
    event_type = Column(Enum(AuthEventType), nullable=False)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    request_id = Column(String, nullable=True)
    metadata_json = Column(JSONType, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ActorMergeAudit(Base):
    __tablename__ = "actor_merge_audit"

    merge_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_actor_id = Column(String(36), nullable=False)
    target_actor_id = Column(String(36), nullable=False)
    merge_reason = Column(String, nullable=False)
    approved_by_actor_id = Column(String(36), nullable=True)
    metadata_json = Column(JSONType, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ProviderLinkToken(Base):
    __tablename__ = "provider_link_token"

    link_token_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    actor_id = Column(String(36), ForeignKey("actor.actor_id"), nullable=False)
    provider_name = Column(String, nullable=False)
    state = Column(String, unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    consumed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))



class ActorProfile(Base):
    __tablename__ = "actor_profile"
    actor_id = Column(String(36), ForeignKey("actor.actor_id"), primary_key=True)
    pog_eligibility = Column(Boolean, default=False)
    actor = relationship("Actor")

