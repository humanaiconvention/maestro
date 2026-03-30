import uuid
from datetime import datetime, timezone
from app.models.enums import ReviewStatus, Verdict, LaneType, PolicyState

# Correlated Verifier Policy thresholds
CORRELATED_VERIFIER_RISK_THRESHOLD = 0.8

def get_utc_now():
    return datetime.now(timezone.utc)

class VerificationPipeline:
    def __init__(self, policy: dict | None = None):
        self.policy = policy or {}

    def run_pipeline(self, claim_text: str, lane: str, evidence: dict, endogenous_only: bool = False) -> dict:
        """
        Five-stage verifier pipeline:
        1. normalize claim into lane
        2. collect lane-specific evidence (passed in)
        3. run lane-specific checks
        4. attach provenance and rationale
        5. escalate ambiguous cases to human review
        """
        # Stage 1: Normalize
        normalized_lane = self._normalize_claim_lane(lane)
        
        # Claim-safety hook
        is_metaphorical = self._detect_metaphorical_or_nonliteral(claim_text)
        if is_metaphorical:
            return {
                "lane": normalized_lane,
                "verdict": Verdict.metaphorical_or_nonliteral,
                "confidence": 1.0,
                "rationale": "Claim flagged as metaphorical or non-literal.",
                "review_status": ReviewStatus.pending,
                "correlated_verifier_risk": 0.0,
                "policy_state": PolicyState.allowed.value,
                "provenance_bundle": {"checked_at": get_utc_now().isoformat()}
            }
            
        # Correlated-verifier policy risk
        risk_score = 1.0 if endogenous_only else 0.1
        policy_state = PolicyState.allowed
        if endogenous_only:
            policy_state = PolicyState.blocked_pending_exogenous_check
            
        # Stage 2 & 3: Run checks based on evidence
        verdict, confidence, rationale, needs_review = self._run_lane_checks(normalized_lane, evidence)
        
        # Stage 4: Attach provenance (mocked for pipeline logic)
        provenance = {"verified_at": get_utc_now().isoformat()}
        
        # Stage 5: Escalate ambiguous cases
        review_status = ReviewStatus.pending
        if needs_review or policy_state == PolicyState.blocked_pending_exogenous_check:
            review_status = ReviewStatus.escalated
            
        return {
            "lane": normalized_lane,
            "verdict": verdict,
            "confidence": confidence,
            "rationale": rationale,
            "review_status": review_status,
            "correlated_verifier_risk": risk_score,
            "policy_state": policy_state.value if isinstance(policy_state, PolicyState) else policy_state,
            "provenance_bundle": provenance
        }

    def _normalize_claim_lane(self, lane: str) -> LaneType:
        if lane.lower() == "empirical":
            return LaneType.empirical
        elif lane.lower() == "theory":
            return LaneType.theory
        elif lane.lower() == "literature":
            return LaneType.literature
        return LaneType.empirical # Default

    def _detect_metaphorical_or_nonliteral(self, claim_text: str) -> bool:
        lower_claim = claim_text.lower()
        if "literal negative entropy in grounding" in lower_claim:
            return True
        if "second-law claims without physical mapping" in lower_claim:
            return True
        if "autonomous verification sufficiency without exogenous grounding" in lower_claim:
            return True
        return False

    def _run_lane_checks(self, lane: LaneType, evidence: dict):
        # Mock logic based on requirements
        needs_review = False
        verdict = Verdict.verified
        confidence = 0.9
        rationale = "Verification passed."
        
        if lane == LaneType.empirical:
            if evidence.get("variance_high") or evidence.get("rerun_disagreement"):
                needs_review = True
                verdict = Verdict.unresolved
                confidence = 0.5
                rationale = "Empirical variance high or disagreement."
        elif lane == LaneType.theory:
            if not evidence.get("formalizable") or evidence.get("proof_failed"):
                needs_review = True
                verdict = Verdict.unresolved
                confidence = 0.5
                rationale = "Proof failed or unformalizable assumptions."
        elif lane == LaneType.literature:
            if evidence.get("sources_conflict") or evidence.get("weak_retrieval"):
                needs_review = True
                verdict = Verdict.unresolved
                confidence = 0.5
                rationale = "Conflicting sources or weak retrieval."
                
        return verdict, confidence, rationale, needs_review
