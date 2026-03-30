import pytest

from app.services.viability.engines import (
    RequestViabilityEngine, 
    ContributionViabilityEngine, 
    SettlementViabilityEngine
)
from app.models.enums import ViabilityDecision

# Mock Policy
POLICY = {
    "request_gates": {
        "prohibited_classes": ["exploitation", "political_manipulation"],
        "review_required_classes": ["medical_inquiry", "vulnerable_populations"],
        "compensation": {
            "min_hourly_rate_usd": 50.0
        },
        "hysteresis_margin": 0.05
    },
    "contribution_gates": {
        "minimum_provenance_score": 0.90,
        "min_confidence": 0.95
    },
    "settlement_gates": {
        "max_extraction_risk": 0.15
    }
}

class TestRequestViabilityEngine:
    def test_pass_valid_request(self):
        engine = RequestViabilityEngine(POLICY)
        request_data = {
            "id": "req_1",
            "request_class": "causal_reasoning",
            "description": "A valid request for causal reasoning.",
            "effective_hourly_estimate": 60.0,
            "entropy_reduction_hypothesis": "H1",
            "phase_synchronization_hypothesis": "H2",
            "evidence_artifact_description": "E1"
        }
        result = engine.evaluate(request_data)
        assert result["status"] == "PASS"
        assert result["overall_viability"] is True

    def test_fail_prohibited_class(self):
        engine = RequestViabilityEngine(POLICY)
        request_data = {
            "id": "req_2",
            "request_class": "exploitation",
            "description": "Prohibited class.",
            "effective_hourly_estimate": 60.0,
            "entropy_reduction_hypothesis": "H1",
            "phase_synchronization_hypothesis": "H2",
            "evidence_artifact_description": "E1"
        }
        result = engine.evaluate(request_data)
        assert result["status"] == "FAIL"
        assert result["overall_viability"] is False
        assert any(g["gate"] == "prohibited_domains" and g["status"] == "FAIL" for g in result["gates"])  

    def test_fail_low_compensation(self):
        engine = RequestViabilityEngine(POLICY)
        request_data = {
            "id": "req_3",
            "request_class": "causal_reasoning",
            "description": "Low pay.",
            "effective_hourly_estimate": 30.0,
            "entropy_reduction_hypothesis": "H1",
            "phase_synchronization_hypothesis": "H2",
            "evidence_artifact_description": "E1"
        }
        result = engine.evaluate(request_data)
        assert result["status"] == "FAIL"
        assert result["overall_viability"] is False
        assert any(g["gate"] == "compensation_coherence" and g["status"] == "FAIL" for g in result["gates"])

    def test_review_medical_inquiry(self):
        engine = RequestViabilityEngine(POLICY)
        request_data = {
            "id": "req_4",
            "request_class": "medical_inquiry",
            "description": "Medical request.",
            "effective_hourly_estimate": 60.0,
            "entropy_reduction_hypothesis": "H1",
            "phase_synchronization_hypothesis": "H2",
            "evidence_artifact_description": "E1"
        }
        result = engine.evaluate(request_data)
        assert result["status"] == "REVIEW_REQUIRED"
        assert result["overall_viability"] is True

    def test_edge_compensation_boundary(self):
        engine = RequestViabilityEngine(POLICY)
        request_data = {
            "id": "req_5",
            "request_class": "causal_reasoning",
            "description": "Exact boundary.",
            "effective_hourly_estimate": 50.0,
            "entropy_reduction_hypothesis": "H1",
            "phase_synchronization_hypothesis": "H2",
            "evidence_artifact_description": "E1"
        }
        result = engine.evaluate(request_data)
        assert result["status"] == "PASS"
        assert result["overall_viability"] is True

    def test_fail_missing_ontology(self):
        engine = RequestViabilityEngine(POLICY)
        request_data = {
            "id": "req_6",
            "request_class": "causal_reasoning",
            "description": "Missing ontology.",
            "effective_hourly_estimate": 60.0,
            "entropy_reduction_hypothesis": "", # Empty
            "phase_synchronization_hypothesis": "H2",
            "evidence_artifact_description": "E1"
        }
        result = engine.evaluate(request_data)
        assert result["status"] == "FAIL"
        assert result["overall_viability"] is False
        assert any(g["gate"] == "ontology_completeness" and g["status"] == "FAIL" for g in result["gates"])

class TestContributionViabilityEngine:
    def test_pass_valid_contribution(self):
        engine = ContributionViabilityEngine(POLICY)
        contribution = {
            "axis": "causal_reasoning",
            "provenance_score": 0.98
        }
        consent_snapshot = {
            "allowed_axes": ["causal_reasoning"]
        }
        result = engine.evaluate(contribution, consent_snapshot)
        assert result["status"] == "PASS"
        assert result["overall_viability"] is True

    def test_fail_axis_not_allowed(self):
        engine = ContributionViabilityEngine(POLICY)
        contribution = {
            "axis": "ethical_deliberation",
            "provenance_score": 0.98
        }
        consent_snapshot = {
            "allowed_axes": ["causal_reasoning"]
        }
        result = engine.evaluate(contribution, consent_snapshot)
        assert result["status"] == "FAIL"
        assert result["overall_viability"] is False

    def test_fail_low_provenance(self):
        engine = ContributionViabilityEngine(POLICY)
        contribution = {
            "axis": "causal_reasoning",
            "provenance_score": 0.85
        }
        consent_snapshot = {
            "allowed_axes": ["causal_reasoning"]
        }
        result = engine.evaluate(contribution, consent_snapshot)
        assert result["status"] == "FAIL"
        assert result["overall_viability"] is False

    def test_review_provenance_range(self):
        engine = ContributionViabilityEngine(POLICY)
        contribution = {
            "axis": "causal_reasoning",
            "provenance_score": 0.92
        }
        consent_snapshot = {
            "allowed_axes": ["causal_reasoning"]
        }
        result = engine.evaluate(contribution, consent_snapshot)
        assert result["status"] == "REVIEW_REQUIRED"
        assert result["overall_viability"] is True

class TestSettlementViabilityEngine:
    def test_pass_valid_settlement(self):
        engine = SettlementViabilityEngine(POLICY)
        settlement_data = {
            "claimed_entropy_reduction": 0.5,
            "extraction_risk": 0.1
        }
        result = engine.evaluate("req_1", settlement_data)
        assert result["status"] == "PASS"
        assert result["overall_viability"] is True

    def test_fail_no_entropy_reduction(self):
        engine = SettlementViabilityEngine(POLICY)
        settlement_data = {
            "claimed_entropy_reduction": 0.0,
            "extraction_risk": 0.1
        }
        result = engine.evaluate("req_1", settlement_data)
        assert result["status"] == "FAIL"
        assert result["overall_viability"] is False
        assert any(g["gate"] == "entropy_reduction_proof" and g["status"] == "FAIL" for g in result["gates"])

    def test_fail_high_extraction_risk_explicit(self):
        engine = SettlementViabilityEngine(POLICY)
        settlement_data = {
            "claimed_entropy_reduction": 0.5,
            "extraction_risk": 0.5 # Explicit high risk
        }
        result = engine.evaluate("req_1", settlement_data)
        assert result["status"] == "FAIL"
        assert result["overall_viability"] is False
        assert any(g["gate"] == "extraction_risk_limit" and g["status"] == "FAIL" for g in result["gates"])

    def test_pass_low_extraction_risk_explicit(self):
        engine = SettlementViabilityEngine(POLICY)
        settlement_data = {
            "claimed_entropy_reduction": 0.5,
            "extraction_risk": 0.10 # Explicit low risk
        }
        result = engine.evaluate("req_1", settlement_data)
        assert result["status"] == "PASS"
        assert result["overall_viability"] is True

    def test_heuristic_overpay_triggers_risk(self):
        engine = SettlementViabilityEngine(POLICY)
        settlement_data = {
            "claimed_entropy_reduction": 0.5,
            "payout_amount": 500,
            "effective_hourly_estimate": 50,
            "duration_minutes": 60
            # ratio = 500 / (50*60/60) = 10.0 > 3.0 -> risk += 0.4. baseline 0.1 + 0.4 = 0.5.
        }
        result = engine.evaluate("req_1", settlement_data)
        assert result["status"] == "FAIL"
        assert result["overall_viability"] is False
        gate = next(g for g in result["gates"] if g["gate"] == "extraction_risk_limit")
        assert gate["status"] == "FAIL"
        assert gate["heuristic_extraction_risk"] == 0.5

    def test_heuristic_training_rights_mismatch(self):
        engine = SettlementViabilityEngine(POLICY)
        settlement_data = {
            "claimed_entropy_reduction": 0.5,
            "usage_rights": ["model_training"],
            "allow_model_training": False
            # mismatch -> risk += 0.4. baseline 0.1 + 0.4 = 0.5.
        }
        result = engine.evaluate("req_1", settlement_data)
        assert result["status"] == "FAIL"
        assert result["overall_viability"] is False
        gate = next(g for g in result["gates"] if g["gate"] == "extraction_risk_limit")
        assert gate["status"] == "FAIL"
        assert gate["heuristic_extraction_risk"] == 0.5

    def test_heuristic_both_violations(self):
        engine = SettlementViabilityEngine(POLICY)
        settlement_data = {
            "claimed_entropy_reduction": 0.5,
            "payout_amount": 500,
            "effective_hourly_estimate": 50,
            "duration_minutes": 60,
            "usage_rights": ["model_training"],
            "allow_model_training": False
            # overpay (+0.4) AND mismatch (+0.4). baseline 0.1 + 0.4 + 0.4 = 0.9.
        }
        result = engine.evaluate("req_1", settlement_data)
        assert result["status"] == "FAIL"
        assert result["overall_viability"] is False
        gate = next(g for g in result["gates"] if g["gate"] == "extraction_risk_limit")
        assert gate["status"] == "FAIL"
        assert gate["heuristic_extraction_risk"] == 0.9

    def test_heuristic_baseline_passes(self):
        engine = SettlementViabilityEngine(POLICY)
        settlement_data = {
            "claimed_entropy_reduction": 0.5,
            "payout_amount": 50,
            "effective_hourly_estimate": 50,
            "duration_minutes": 60,
            "allow_model_training": True
            # ratio 1.0, no training mismatch. baseline 0.1.
        }
        result = engine.evaluate("req_1", settlement_data)
        assert result["status"] == "PASS"
        assert result["overall_viability"] is True
        gate = next(g for g in result["gates"] if g["gate"] == "extraction_risk_limit")
        assert gate["status"] == "PASS"
        assert gate["heuristic_extraction_risk"] == 0.1

    def test_no_data_worst_case(self):
        engine = SettlementViabilityEngine(POLICY)
        settlement_data = {
            "claimed_entropy_reduction": 0.5
            # No heuristic keys -> risk = 1.0.
        }
        result = engine.evaluate("req_1", settlement_data)
        assert result["status"] == "FAIL"
        assert result["overall_viability"] is False
        gate = next(g for g in result["gates"] if g["gate"] == "extraction_risk_limit")
        assert gate["status"] == "FAIL"
        assert gate["value"] == 1.0
