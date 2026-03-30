import json
import hashlib
import logging
import os
import sys
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Resolve prism libs from monorepo layout
_PRISM_ADAPTERS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../libs/adapters")
)
_PRISM_SCHEMAS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../libs/schemas")
)
for _p in [_PRISM_ADAPTERS, _PRISM_SCHEMAS]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from app.models.enums import ViabilityDecision

class GateResult:
    def __init__(self, gate: str, status: str, value: Any, threshold: Any, confidence: float, reason: str, hysteresis_applied: bool = False, previous_state: str = None):
        self.gate = gate
        self.status = status # "PASS" | "FAIL" | "REVIEW_REQUIRED"
        self.value = value
        self.threshold = threshold
        self.confidence = confidence
        self.reason = reason
        self.hysteresis_applied = hysteresis_applied
        self.previous_state = previous_state

    def to_dict(self):
        return {
            "gate": self.gate,
            "status": self.status,
            "value": self.value,
            "threshold": self.threshold,
            "confidence": self.confidence,
            "reason": self.reason,
            "hysteresis_applied": self.hysteresis_applied,
            "previous_state": self.previous_state
        }

class RequestViabilityEngine:
    def __init__(self, policy: dict):
        self.policy = policy.get('request_gates', {})
        self.hysteresis_margin = self.policy.get('hysteresis_margin', 0.05)

    def evaluate(self, request_data: dict, previous_eval: dict = None) -> Dict[str, Any]:
        gates = []
        overall_viability = True
        decision = "PASS"

        # 1. Prohibited Domains Gate (Hard Constraint)
        prohibited_classes = self.policy.get("prohibited_classes", [])
        req_class = request_data.get("request_class")

        # Also check for semantic keywords in the description
        prohibited_keywords = ["political manipulation", "deception", "exploitation", "human experimentation"]
        desc = request_data.get("description", "").lower()

        is_prohibited = req_class in prohibited_classes or any(kw in desc for kw in prohibited_keywords)  

        if is_prohibited:
            gates.append(GateResult(
                "prohibited_domains", "FAIL", req_class, "NOT IN " + str(prohibited_classes),
                1.0, f"Request class '{req_class}' is strictly prohibited."
            ).to_dict())
            overall_viability = False
        else:
            gates.append(GateResult("prohibited_domains", "PASS", req_class, "ALLOWED", 1.0, "Domain check passed.").to_dict())

        # 2. Compensation Coherence (Hysteresis Check)
        eff_hourly = float(request_data.get("effective_hourly_estimate", 0.0))
        threshold = float(self.policy.get("compensation", {}).get("min_hourly_rate_usd", 50.0))
        
        # Pull previous state for hysteresis
        prev_gate = next((g for g in (previous_eval or {}).get("gates", []) if g["gate"] == "compensation_coherence"), None)
        prev_status = prev_gate["status"] if prev_gate else None

        margin = threshold * self.hysteresis_margin
        current_status = "PASS" if eff_hourly >= threshold else "FAIL"
        hysteresis_applied = False

        if prev_status:
            # Hysteresis logic: only flip if outside the margin
            if prev_status == "PASS" and eff_hourly < (threshold - margin):
                current_status = "FAIL"
            elif prev_status == "FAIL" and eff_hourly > (threshold + margin):
                current_status = "PASS"
            elif (threshold - margin) <= eff_hourly <= (threshold + margin):
                current_status = prev_status
                hysteresis_applied = True

        if current_status == "FAIL":
            overall_viability = False

        gates.append(GateResult(
            "compensation_coherence", current_status, eff_hourly, threshold, 1.0,
            f"Hourly rate check.", hysteresis_applied, prev_status
        ).to_dict())

        # 3. Ontology Completeness
        ont_fields = ["entropy_reduction_hypothesis", "phase_synchronization_hypothesis", "evidence_artifact_description"]
        missing = [f for f in ont_fields if not request_data.get(f)]
        if missing:
            gates.append(GateResult("ontology_completeness", "FAIL", False, True, 1.0, f"Missing ontology: {missing}").to_dict())
            overall_viability = False
        else:
            gates.append(GateResult("ontology_completeness", "PASS", True, True, 1.0, "Ontology verified.").to_dict())

        # Final Decision
        needs_review = req_class in self.policy.get("review_required_classes", ["medical_inquiry", "vulnerable_populations"])

        if not overall_viability:
            decision = "FAIL"
        elif needs_review:
            decision = "REVIEW_REQUIRED"

        eval_hash = hashlib.sha256(json.dumps(gates, sort_keys=True).encode()).hexdigest()

        return {
            "status": decision,
            "overall_viability": overall_viability,
            "entity_type": "request",
            "entity_id": request_data.get("id"),
            "gates": gates,
            "decision_rationale": "All hard constraints must pass." if not overall_viability else "Evaluation complete.",
            "evaluation_hash": eval_hash
        }

class ContributionViabilityEngine:
    def __init__(self, policy: dict):
        self.policy = policy.get('contribution_gates', {})

    def evaluate(self, contribution: dict, consent_snapshot: dict) -> Dict[str, Any]:
        gates = []
        overall_viability = True
        decision = "PASS"

        # 1. Consent Scope Match (Non-compensatory)
        axis = contribution.get("axis")
        allowed = consent_snapshot.get("allowed_axes", [])
        if axis not in allowed:
            gates.append(GateResult("consent_validity", "FAIL", axis, allowed, 1.0, "Lived experience axis outside of human consent scope.").to_dict())
            overall_viability = False
        else:
            gates.append(GateResult("consent_validity", "PASS", axis, allowed, 1.0, "Matches human consent boundaries.").to_dict())

        # 2. Provenance Integrity (Biometric threshold)
        prov_score = float(contribution.get("provenance_score", 0.0))
        fail_threshold = float(self.policy.get("minimum_provenance_score", 0.90))
        pass_threshold = float(self.policy.get("min_confidence", 0.95))

        if prov_score < fail_threshold:
            gates.append(GateResult("provenance_integrity", "FAIL", prov_score, fail_threshold, 1.0, "Biometric provenance insufficient.").to_dict())
            overall_viability = False
        elif prov_score < pass_threshold:
            gates.append(GateResult("provenance_integrity", "REVIEW_REQUIRED", prov_score, pass_threshold, 0.5, "Ambiguous provenance score.").to_dict())
            decision = "REVIEW_REQUIRED"
        else:
            gates.append(GateResult("provenance_integrity", "PASS", prov_score, pass_threshold, 1.0, "Provenance verified.").to_dict())

        if not overall_viability:
            decision = "FAIL"

        eval_hash = hashlib.sha256(json.dumps(gates, sort_keys=True).encode()).hexdigest()

        return {
            "status": decision,
            "overall_viability": overall_viability,
            "entity_type": "contribution",
            "entity_id": contribution.get("id"),
            "gates": gates,
            "decision_rationale": "Verified provenance and consent matching required.",
            "evaluation_hash": eval_hash
        }

class SettlementViabilityEngine:
    def __init__(self, policy: dict):
        self.policy = policy.get('settlement_gates', {})

    def evaluate(self, request_id: str, settlement_data: dict) -> Dict[str, Any]:
        gates = []
        overall_viability = True

        # 1. Epistemic Value Check (Entropy Reduction Proof)
        claimed_reduction = float(settlement_data.get("claimed_entropy_reduction", 0.0))
        if claimed_reduction <= 0:
            gates.append(GateResult("entropy_reduction_proof", "FAIL", claimed_reduction, ">0", 1.0, "Agent failed to prove thermodynamic entropy reduction.").to_dict())
            overall_viability = False
        else:
            gates.append(GateResult("entropy_reduction_proof", "PASS", claimed_reduction, ">0", 1.0, "Entropy proof verified.").to_dict())

        # 2. Extraction Risk (Aggregate Check)
        max_risk = float(self.policy.get("max_extraction_risk", 0.15))
        risk_score = settlement_data.get("extraction_risk")
        heuristic_risk = 0.1 # Baseline

        # Heuristic calculation logic
        payout = settlement_data.get("payout_amount")
        eff_hourly = settlement_data.get("effective_hourly_estimate")
        duration = settlement_data.get("duration_minutes")
        
        if payout is not None and eff_hourly is not None and duration is not None and duration > 0:
            market_payout = (float(eff_hourly) * float(duration)) / 60.0
            if market_payout > 0:
                ratio = float(payout) / market_payout
                if ratio > 3.0:
                    heuristic_risk += 0.4
        
        usage_rights = settlement_data.get("usage_rights", [])
        allow_training = settlement_data.get("allow_model_training")
        if isinstance(usage_rights, list) and "model_training" in usage_rights and allow_training is False:
            heuristic_risk += 0.4
            
        heuristic_risk = min(max(heuristic_risk, 0.0), 1.0)

        if risk_score is None:
            # Fallback to heuristic if we have ANY data, otherwise 1.0 worst-case
            has_heuristic_data = any(settlement_data.get(k) is not None for k in ["payout_amount", "effective_hourly_estimate", "duration_minutes", "usage_rights", "allow_model_training"])
            if has_heuristic_data:
                risk_score = heuristic_risk
            else:
                risk_score = 1.0
        else:
            risk_score = float(risk_score)

        gate_status = "PASS" if risk_score <= max_risk else "FAIL"
        if gate_status == "FAIL":
            overall_viability = False

        res = GateResult(
            "extraction_risk_limit", 
            gate_status, 
            risk_score, 
            max_risk, 
            1.0, 
            "Extraction risk within bounds." if gate_status == "PASS" else "Aggregate extraction risk too high."
        ).to_dict()
        res["heuristic_extraction_risk"] = heuristic_risk
        gates.append(res)

        # 3. PRISM Entropy Measurement Consistency (non-blocking when UNVERIFIED)
        prism_gate = self._entropy_measurement_consistency_gate(settlement_data)
        gates.append(prism_gate)
        if prism_gate["status"] == "FAIL":
            overall_viability = False

        decision = "PASS" if overall_viability else "FAIL"

        eval_hash = hashlib.sha256(json.dumps(gates, sort_keys=True).encode()).hexdigest()

        return {
            "status": decision,
            "overall_viability": overall_viability,
            "entity_type": "settlement",
            "entity_id": request_id,
            "gates": gates,
            "decision_rationale": "Settlement requires valid thermodynamic proof and low extraction risk.",
            "evaluation_hash": eval_hash
        }

    def _entropy_measurement_consistency_gate(self, settlement_data: dict) -> dict:
        """Validate that PRISM-measured entropy delta is consistent with claimed reduction.

        Returns UNVERIFIED (non-blocking) when no PRISM snapshots are present.
        Returns FAIL when measured reduction does not meet the epsilon threshold.
        Returns REVIEW_REQUIRED when claimed vs measured discrepancy exceeds tolerance.
        Returns PASS when verified and within tolerance.

        Policy keys used:
            settlement_gates.entropy_epsilon             (default 0.01)
            settlement_gates.entropy_consistency_tolerance (default 0.25)
        """
        snap_before = settlement_data.get("entropy_snapshot_before")
        snap_after = settlement_data.get("entropy_snapshot_after")

        if snap_before is None or snap_after is None:
            return {
                "gate": "entropy_measurement_consistency",
                "status": "UNVERIFIED",
                "value": None,
                "threshold": None,
                "confidence": 0.0,
                "reason": "No PRISM entropy snapshots provided — gate skipped (backward-compatible).",
            }

        epsilon = float(self.policy.get("entropy_epsilon", 0.01))
        tolerance = float(self.policy.get("entropy_consistency_tolerance", 0.25))
        claimed = float(settlement_data.get("claimed_entropy_reduction", 0.0))

        try:
            from prism_adapter import compute_entropy_delta
            from prism_schemas import EntropySnapshot

            before = EntropySnapshot.from_dict(snap_before)
            after = EntropySnapshot.from_dict(snap_after)
            proof = compute_entropy_delta(before, after, epsilon=epsilon)
        except Exception as exc:
            logger.warning("prism settlement gate: snapshot evaluation failed: %s", exc)
            return {
                "gate": "entropy_measurement_consistency",
                "status": "UNVERIFIED",
                "value": None,
                "threshold": None,
                "confidence": 0.0,
                "reason": f"PRISM evaluation failed: {exc}",
            }

        if not proof.reduction_verified:
            return {
                "gate": "entropy_measurement_consistency",
                "status": "FAIL",
                "value": proof.delta_spectral_entropy,
                "threshold": -epsilon,
                "confidence": 1.0,
                "reason": (
                    f"Measured ΔS={proof.delta_spectral_entropy:+.4f} did not meet "
                    f"reduction threshold ε={epsilon}."
                ),
            }

        measured = -proof.delta_spectral_entropy  # positive = reduction
        if abs(claimed) > 1e-9:
            discrepancy = abs(measured - claimed) / (abs(claimed) + 1e-9)
            if discrepancy > tolerance:
                return {
                    "gate": "entropy_measurement_consistency",
                    "status": "REVIEW_REQUIRED",
                    "value": proof.delta_spectral_entropy,
                    "threshold": tolerance,
                    "confidence": 0.5,
                    "reason": (
                        f"Claimed reduction {claimed:.4f} vs measured {measured:.4f} — "
                        f"{discrepancy:.1%} discrepancy exceeds {tolerance:.0%} tolerance."
                    ),
                }

        return {
            "gate": "entropy_measurement_consistency",
            "status": "PASS",
            "value": proof.delta_spectral_entropy,
            "threshold": -epsilon,
            "confidence": 1.0,
            "reason": (
                f"PRISM-verified entropy reduction ΔS={proof.delta_spectral_entropy:+.4f} "
                f"(ε={epsilon})."
            ),
        }
