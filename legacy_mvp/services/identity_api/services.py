from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple
import uuid
import logging
from sqlalchemy.orm import Session
from sqlalchemy import or_

from models import (
    Actor, CredentialBinding, HumanProfile, AgentProfile,
    AuthEvent, ProviderLinkToken, ActorType, ProviderType,
    OnboardingState, AuthEventType, GroundingRequest, ConsentScope,
    ExperienceContribution, ProvenanceRecord, ViabilityEvaluation, SettlementEvent,
    RequestStatus, ContributionStatus, ViabilityDecision, SettlementStatus
)
from exceptions import LinkingError, MaestroDomainError
from viability_engine import ViabilityEngine

logger = logging.getLogger(__name__)

class LinkingService:
    def __init__(self, db: Session):
        self.db = db

    def resolve_login_identity(
        self,
        provider_type: ProviderType,
        provider_name: str,
        external_subject: str,
        external_email: str = None,
        current_actor_id: str = None,
        request_ip: str = None,
        user_agent: str = None
    ) -> tuple[Actor, bool]:
        binding = self.db.query(CredentialBinding).filter(
            CredentialBinding.provider_name == provider_name,
            CredentialBinding.external_subject == external_subject
        ).first()

        if binding:
            actor = binding.actor
            binding.last_used_at = datetime.now(timezone.utc)
            actor.last_seen_at = datetime.now(timezone.utc)
            self._log_event(actor.actor_id, binding.binding_id, AuthEventType.login_success, request_ip, user_agent)
            self.db.commit()
            return actor, False

        if current_actor_id:
            actor = self.db.query(Actor).filter(Actor.actor_id == current_actor_id).first()
            if not actor:
                raise LinkingError("Current authenticated actor not found.")

            new_binding = CredentialBinding(
                actor_id=actor.actor_id,
                provider_type=provider_type,
                provider_name=provider_name,
                external_subject=external_subject,
                external_email=external_email,
                verified_at=datetime.now(timezone.utc)
            )
            self.db.add(new_binding)
            self._log_event(actor.actor_id, new_binding.binding_id, AuthEventType.provider_link_completed, request_ip, user_agent)
            self.db.commit()
            return actor, False

        if external_email:
            existing_profile = self.db.query(HumanProfile).filter(HumanProfile.email == external_email).first()
            if existing_profile:
                logger.warning(f"Ambiguous email collision for {external_email}. Silent merge rejected.") 
                raise LinkingError(f"An account with {external_email} already exists. Please sign in with your original method and link this new provider from Account Settings.")

        new_actor = Actor(
            actor_type=ActorType.human,
            display_name=external_email.split('@')[0] if external_email else "New User",
            last_seen_at=datetime.now(timezone.utc)
        )
        self.db.add(new_actor)
        self.db.flush()

        profile = HumanProfile(
            actor_id=new_actor.actor_id,
            email=external_email,
            email_verified=bool(external_email),
            onboarding_state=OnboardingState.new
        )
        self.db.add(profile)

        new_binding = CredentialBinding(
            actor_id=new_actor.actor_id,
            provider_type=provider_type,
            provider_name=provider_name,
            external_subject=external_subject,
            external_email=external_email,
            verified_at=datetime.now(timezone.utc)
        )
        self.db.add(new_binding)
        self.db.flush()

        self._log_event(new_actor.actor_id, new_binding.binding_id, AuthEventType.login_success, request_ip, user_agent)
        self.db.commit()
        return new_actor, True

    def unlink_provider(self, actor_id: str, binding_id: str) -> bool:
        actor = self.db.query(Actor).filter(Actor.actor_id == actor_id).first()
        if not actor:
            raise LinkingError("Actor not found")

        active_bindings = [b for b in actor.bindings if b.revoked_at is None]
        if len(active_bindings) <= 1:
            raise LinkingError("Cannot unlink the last remaining credential. Please add another sign-in method first.")

        target_binding = next((b for b in active_bindings if b.binding_id == binding_id), None)
        if not target_binding:
            raise LinkingError("Binding not found or already revoked.")

        target_binding.revoked_at = datetime.now(timezone.utc)
        self._log_event(actor_id, target_binding.binding_id, AuthEventType.provider_unlinked)
        self.db.commit()
        return True

    def _log_event(self, actor_id: str, binding_id: str, event_type: AuthEventType, ip: str = None, ua: str = None):
        event = AuthEvent(
            actor_id=actor_id,
            binding_id=binding_id,
            event_type=event_type,
            ip_address=ip,
            user_agent=ua
        )
        self.db.add(event)

class MaestroDomainService:
    def __init__(self, db: Session):
        self.db = db
        self.viability_engine = ViabilityEngine(db)

    def get_human_dashboard_data(self, actor_id: str) -> Dict[str, Any]:
        """Fetch human dashboard aggregates."""
        profile = self.db.query(HumanProfile).filter_by(actor_id=actor_id).first()
        if not profile:
            raise MaestroDomainError("Human profile not found")

        contributions = self.db.query(ExperienceContribution).filter_by(actor_id=actor_id).all()
        settlements = self.db.query(SettlementEvent).filter_by(agent_actor_id=actor_id).all() # Simple mock mapping for demo
        
        settled_total = sum(s.human_amount for s in settlements if s.status == SettlementStatus.completed)
        
        return {
            "profile": profile,
            "total_contributions": len(contributions),
            "pending_settlements": len([c for c in contributions if c.contribution_status == ContributionStatus.verified]),
            "settled_contributions": len([c for c in contributions if c.contribution_status == ContributionStatus.settled]),
            "settled_total": settled_total
        }

    def get_compensation_data(self, actor_id: str) -> Dict[str, Any]:
        """Fetch full compensation ledger."""
        settlements = self.db.query(SettlementEvent).order_by(SettlementEvent.created_at.desc()).all() # Demo filter
        
        pending_total = sum(s.human_amount for s in settlements if s.status == SettlementStatus.pending)
        settled_total = sum(s.human_amount for s in settlements if s.status == SettlementStatus.completed)
        
        return {
            "pending_total": pending_total,
            "settled_total": settled_total,
            "settlements": settlements
        }

    def update_consent_scope(self, actor_id: str, permitted_classes: list, modality_permissions: dict) -> ConsentScope:
        """Create or update a human's bounded consent scope."""
        profile = self.db.query(HumanProfile).filter_by(actor_id=actor_id).first()
        if not profile:
            raise MaestroDomainError("Human profile not found")

        scope = None
        if profile.consent_profile_id:
            scope = self.db.query(ConsentScope).filter_by(consent_scope_id=profile.consent_profile_id).first()

        if not scope:
            scope = ConsentScope(actor_id=actor_id)
            self.db.add(scope)
            self.db.flush()
            profile.consent_profile_id = scope.consent_scope_id

        scope.permitted_request_classes = permitted_classes
        scope.modality_permissions = modality_permissions
        scope.updated_at = datetime.now(timezone.utc)
        self.db.commit()
        return scope

    def get_open_opportunities(self, actor_id: str) -> List[GroundingRequest]:
        """Find active requests matching the human's consent scope."""
        profile = self.db.query(HumanProfile).filter_by(actor_id=actor_id).first()
        if not profile or not profile.consent_profile_id:
            return []
            
        scope = self.db.query(ConsentScope).filter_by(consent_scope_id=profile.consent_profile_id).first()
        if not scope or not scope.permitted_request_classes:
            return []

        # Simple matching: return open requests where the class is permitted
        requests = self.db.query(GroundingRequest).filter(
            GroundingRequest.status == RequestStatus.open
        ).all()
        
        matched = [r for r in requests if r.request_class in scope.permitted_request_classes]
        return matched

    def create_grounding_request(
        self, agent_id: str, title: str, description: str, request_class: str,
        entropy_target: float, budget: float, required_contribs: int
    ) -> GroundingRequest:
        """Agent creates a new grounding request."""
        # Note: One-way AI-to-human funding is represented by the `compensation_budget`.
        # Agents must provide this upfront (in a real system, via an escrow smart contract).
        req = GroundingRequest(
            agent_actor_id=agent_id,
            title=title,
            description=description,
            request_class=request_class,
            hypothesis_entropy_reduction=entropy_target,
            compensation_budget=budget,
            target_population_description="General",
            status=RequestStatus.open
        )
        self.db.add(req)
        self.db.commit()
        return req

    def submit_contribution(
        self, actor_id: str, request_id: str, modality: str, content: str, provenance_data: dict
    ) -> ExperienceContribution:
        """Human submits a lived experience with biometric provenance."""
        req = self.db.query(GroundingRequest).filter_by(request_id=request_id).first()
        if not req or req.status != RequestStatus.open:
            raise MaestroDomainError("Request is not open")
            
        profile = self.db.query(HumanProfile).filter_by(actor_id=actor_id).first()
        if not profile or not profile.consent_profile_id:
            raise MaestroDomainError("Must configure consent scope before contributing")

        # Create contribution
        contrib = ExperienceContribution(
            actor_id=actor_id,
            request_id=request_id,
            modality=modality,
            content_pointer=content,
            contribution_status=ContributionStatus.submitted
        )
        self.db.add(contrib)
        self.db.flush()

        # Generate Provenance Record
        # This makes provenance a first-class citizen
        prov = ProvenanceRecord(
            contribution_id=contrib.contribution_id,
            consent_scope_id=profile.consent_profile_id,
            capture_context=provenance_data,
            signatures={"telemetry_hash": str(uuid.uuid4())} # Mock signature
        )
        self.db.add(prov)
        self.db.commit()
        
        return contrib

    def evaluate_viability(self, request_id: str) -> ViabilityEvaluation:
        """Evaluate if a GroundingRequest is non-extractive and viable."""
        req = self.db.query(GroundingRequest).filter_by(request_id=request_id).first()
        if not req:
            raise MaestroDomainError("Request not found")
            
        eval_record = self.viability_engine.evaluate_request(req)
        self.db.add(eval_record)
        self.db.commit()
        return eval_record

    def evaluate_contribution_viability(self, contribution_id: str) -> ViabilityEvaluation:
        """Evaluate if a contribution meets provenance and consent gates."""
        contrib = self.db.query(ExperienceContribution).filter_by(contribution_id=contribution_id).first()
        if not contrib:
            raise MaestroDomainError("Contribution not found")
        prov = self.db.query(ProvenanceRecord).filter_by(contribution_id=contribution_id).first()
        if not prov:
            raise MaestroDomainError("Provenance record not found")
            
        eval_record = self.viability_engine.evaluate_contribution(contrib, prov)
        self.db.add(eval_record)
        self.db.commit()
        return eval_record

    def evaluate_settlement_viability(self, request_id: str, evidence_metadata: dict, confidence: float) -> Tuple[bool, str]:
        """Check if settlement conditions are satisfied."""
        req = self.db.query(GroundingRequest).filter_by(request_id=request_id).first()
        if not req:
            raise MaestroDomainError("Request not found")
            
        return self.viability_engine.evaluate_settlement(req, evidence_metadata, confidence)



