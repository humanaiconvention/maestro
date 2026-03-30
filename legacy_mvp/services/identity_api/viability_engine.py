import yaml
import logging
from typing import Dict, Any, Tuple
from sqlalchemy.orm import Session
from models import (
    GroundingRequest, ExperienceContribution, ConsentScope, 
    ProvenanceRecord, SettlementEvent, ViabilityEvaluation, ViabilityDecision,
    HumanProfile
)

import os

logger = logging.getLogger(__name__)

class ViabilityEngine:
    def __init__(self, db: Session, policy_path: str = None):
        self.db = db
        if policy_path is None:
            # Load from ENV or relative canonical path
            policy_path = os.environ.get("VIABILITY_POLICY_PATH")
            if not policy_path:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                # Canonical path: ../../config/viability_policy.yaml
                policy_path = os.path.normpath(os.path.join(base_dir, "..", "..", "config", "viability_policy.yaml"))
        
        logger.info(f"Loading viability policy from: {policy_path}")
        try:
            with open(policy_path, 'r') as f:
                self.policy = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load viability policy: {e}")
            # Fallback to defaults matching canonical schema
            self.policy = {
                "request_gates": {
                    "prohibited_classes": [], 
                    "review_required_classes": [], 
                    "compensation": {"min_total_budget_usd": 10.0}, 
                    "ontology_requirements": {}
                },
                "contribution_gates": {
                    "minimum_provenance_score": 0.5, 
                    "require_consent_match": True, 
                    "allow_withdrawn_contributions": False
                },
                "settlement_gates": {
                    "require_request_viability": True, 
                    "require_contribution_viability": True, 
                    "require_evidence_artifact": True, 
                    "minimum_confidence_level": 0.5
                }
            }

    def evaluate_request(self, request: GroundingRequest) -> ViabilityEvaluation:
        gates = self.policy.get("request_gates", {})

        results = {}

        # 1. Prohibited Domain Gate
        prohibited = request.request_class in gates.get("prohibited_classes", [])
        results["prohibited_domain"] = not prohibited

        # 2. Compensation Viability Gate (Canonical lookup)
        min_budget = gates.get("compensation", {}).get("min_total_budget_usd", 0.0)
        has_budget = request.compensation_budget >= min_budget
        results["compensation_viable"] = has_budget

        # 3. Ontology Completeness Gate
        ont_req = gates.get("ontology_requirements", {})
        ont_pass = True
        if ont_req.get("require_entropy_hypothesis") and not request.hypothesis_entropy_reduction:        
            ont_pass = False
        if ont_req.get("require_phase_hypothesis") and not request.hypothesis_phase_sync:
            ont_pass = False
        if ont_req.get("require_success_signal_definition") and not request.success_signal_type:
            ont_pass = False
        results["ontology_complete"] = ont_pass

        # 4. Review Required Check
        needs_review = request.request_class in gates.get("review_required_classes", [])

        # Aggregate
        harm_exclusion = not prohibited
        extraction_risk = has_budget

        # Non-compensatory logic: if any hard gate fails, block.
        if not harm_exclusion or not extraction_risk or not ont_pass:
            decision = ViabilityDecision.block
            explanation = "Failed hard constraint gate."
        elif needs_review:
            decision = ViabilityDecision.review
            explanation = "Request class requires manual operator review."
        else:
            decision = ViabilityDecision.allow
            explanation = "Passed all viability gates."

        eval_record = ViabilityEvaluation(
            request_id=request.request_id,
            gate_results=results,
            harm_exclusion_result=harm_exclusion,
            extraction_risk_result=extraction_risk,
            consent_validity_result=True, # N/A for request
            provenance_validity_result=True, # N/A for request
            sustainability_result=extraction_risk,
            final_decision=decision,
            explanation=explanation
        )
        return eval_record

    def evaluate_contribution(self, contribution: ExperienceContribution, provenance: ProvenanceRecord) -> ViabilityEvaluation:
        gates = self.policy.get("contribution_gates", {})
        results = {}

        # 1. Provenance Integrity
        # Assume provenance.signatures contains a 'humanity_score' for now.
        score = provenance.capture_context.get("humanity_score", 1.0)
        prov_pass = score >= gates.get("minimum_provenance_score", 0.0)
        results["provenance_pass"] = prov_pass

        # 2. Consent Match
        consent_pass = True
        if gates.get("require_consent_match"):
            consent = self.db.query(ConsentScope).filter_by(consent_scope_id=provenance.consent_scope_id).first()
            req = self.db.query(GroundingRequest).filter_by(request_id=contribution.request_id).first()   
            if consent and req:
                if req.request_class not in (consent.permitted_request_classes or []):
                    consent_pass = False
        results["consent_pass"] = consent_pass

        # 3. Withdrawal Status
        status_pass = contribution.withdrawn_at is None or gates.get("allow_withdrawn_contributions", False)
        results["status_pass"] = status_pass

        # Aggregate
        if not prov_pass or not consent_pass or not status_pass:
            decision = ViabilityDecision.block
            explanation = "Contribution failed provenance, consent, or status checks."
        else:
            decision = ViabilityDecision.allow
            explanation = "Contribution is viable."

        eval_record = ViabilityEvaluation(
            contribution_id=contribution.contribution_id,
            request_id=contribution.request_id,
            gate_results=results,
            harm_exclusion_result=True,
            extraction_risk_result=True,
            consent_validity_result=consent_pass,
            provenance_validity_result=prov_pass,
            sustainability_result=True,
            final_decision=decision,
            explanation=explanation
        )
        return eval_record

    def evaluate_settlement(self, request: GroundingRequest, evidence_metadata: dict, confidence: float) -> Tuple[bool, str]:
        gates = self.policy.get("settlement_gates", {})

        if gates.get("require_request_viability"):
            req_eval = self.db.query(ViabilityEvaluation).filter_by(request_id=request.request_id).order_by(ViabilityEvaluation.created_at.desc()).first()
            if not req_eval or req_eval.final_decision == ViabilityDecision.block:
                return False, "Request viability blocked."

        if gates.get("require_evidence_artifact") and not evidence_metadata:
            return False, "Missing required evidence artifact."

        if confidence < gates.get("minimum_confidence_level", 0.0):
            return False, "Confidence level below threshold."

        return True, "Settlement eligible."
