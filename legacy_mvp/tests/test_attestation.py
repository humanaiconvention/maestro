"""Tests for Phase D — Cryptographic Attestation Pipeline (services/attestation.py)

Coverage:
  - TLSNotaryVerifier: missing proof, missing fields, empty signature, invalid
    timestamp, valid proof.
  - SecureEnclaveVerifier: missing attestation, missing fields, empty quote,
    bad pcr_values, valid attestation.
  - TelemetryCorrelator: sub-human keystroke speed, metronomic typing, short
    session, multiple anomalies, clean telemetry.
  - AttestationPipeline: no bundle, single valid method, both methods, timing
    anomaly escalation, score clamping.
  - E2E integration: contribution with spoofed timing routes to REVIEW_REQUIRED
    even when humanity_score would otherwise pass.
"""
import os
import pytest

os.environ["TESTING"] = "true"

from app.services.attestation import (
    TLSNotaryVerifier,
    SecureEnclaveVerifier,
    TelemetryCorrelator,
    AttestationPipeline,
    MIN_HUMAN_REACTION_MS,
    MIN_PLAUSIBLE_SESSION_MS,
    AUTOMATION_REGULARITY_THRESHOLD,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_TLS_PROOF = {
    "session_id": "sess-abc123",
    "timestamp_utc": 1_700_000_000.0,
    "server_name": "api.example.com",
    "notary_signature": "sig_deadbeef",
}

VALID_ENCLAVE_ATTESTATION = {
    "enclave_id": "enclave-prod-1",
    "quote": "AQIDBA==",
    "pcr_values": {"pcr0": "aabbcc", "pcr1": "ddeeff"},
    "nonce": "nonce-xyz",
}

CLEAN_CONTEXT = {
    "humanity_score": 0.96,
    "typing_cadence_std": 85.0,
    "typing_cadence_mean_ms": 210.0,
    "inter_keystroke_min_ms": 130.0,
    "session_duration_ms": 45_000.0,
}


# ── TLSNotaryVerifier ─────────────────────────────────────────────────────────

class TestTLSNotaryVerifier:
    def test_no_proof_returns_not_present(self):
        result = TLSNotaryVerifier().verify({})
        assert result["present"] is False
        assert result["valid"] is False

    def test_missing_fields(self):
        result = TLSNotaryVerifier().verify({"tlsnotary_proof": {"session_id": "x"}})
        assert result["present"] is True
        assert result["valid"] is False
        assert "missing required fields" in result["detail"]

    def test_empty_signature(self):
        proof = {**VALID_TLS_PROOF, "notary_signature": ""}
        result = TLSNotaryVerifier().verify({"tlsnotary_proof": proof})
        assert result["valid"] is False
        assert "empty" in result["detail"]

    def test_invalid_timestamp(self):
        proof = {**VALID_TLS_PROOF, "timestamp_utc": -1}
        result = TLSNotaryVerifier().verify({"tlsnotary_proof": proof})
        assert result["valid"] is False

    def test_valid_proof(self):
        result = TLSNotaryVerifier().verify({"tlsnotary_proof": VALID_TLS_PROOF})
        assert result["present"] is True
        assert result["valid"] is True
        assert result["confidence"] == 1.0


# ── SecureEnclaveVerifier ─────────────────────────────────────────────────────

class TestSecureEnclaveVerifier:
    def test_no_attestation_returns_not_present(self):
        result = SecureEnclaveVerifier().verify({})
        assert result["present"] is False
        assert result["valid"] is False

    def test_missing_fields(self):
        result = SecureEnclaveVerifier().verify(
            {"enclave_attestation": {"enclave_id": "x"}}
        )
        assert result["present"] is True
        assert result["valid"] is False
        assert "missing required fields" in result["detail"]

    def test_empty_quote(self):
        att = {**VALID_ENCLAVE_ATTESTATION, "quote": ""}
        result = SecureEnclaveVerifier().verify({"enclave_attestation": att})
        assert result["valid"] is False

    def test_empty_pcr_values(self):
        att = {**VALID_ENCLAVE_ATTESTATION, "pcr_values": {}}
        result = SecureEnclaveVerifier().verify({"enclave_attestation": att})
        assert result["valid"] is False

    def test_pcr_not_dict(self):
        att = {**VALID_ENCLAVE_ATTESTATION, "pcr_values": "not-a-dict"}
        result = SecureEnclaveVerifier().verify({"enclave_attestation": att})
        assert result["valid"] is False

    def test_valid_attestation(self):
        result = SecureEnclaveVerifier().verify(
            {"enclave_attestation": VALID_ENCLAVE_ATTESTATION}
        )
        assert result["present"] is True
        assert result["valid"] is True
        assert result["confidence"] == 1.0


# ── TelemetryCorrelator ───────────────────────────────────────────────────────

class TestTelemetryCorrelator:
    def test_clean_context_no_anomalies(self):
        result = TelemetryCorrelator().correlate(CLEAN_CONTEXT, [])
        assert result["timing_valid"] is True
        assert result["anomalies"] == []
        assert result["penalty"] == 0.0

    def test_sub_human_keystroke_speed(self):
        ctx = {**CLEAN_CONTEXT, "inter_keystroke_min_ms": MIN_HUMAN_REACTION_MS - 1}
        result = TelemetryCorrelator().correlate(ctx, [])
        assert result["timing_valid"] is False
        assert len(result["anomalies"]) == 1
        assert "sub-human reaction time" in result["anomalies"][0]
        assert result["penalty"] == pytest.approx(0.05)

    def test_metronomic_typing(self):
        # CV = 2 / 1000 = 0.002 < threshold 0.05
        ctx = {**CLEAN_CONTEXT, "typing_cadence_std": 2.0, "typing_cadence_mean_ms": 1000.0}
        result = TelemetryCorrelator().correlate(ctx, [])
        assert result["timing_valid"] is False
        assert any("metronomically regular" in a for a in result["anomalies"])

    def test_short_session(self):
        ctx = {**CLEAN_CONTEXT, "session_duration_ms": MIN_PLAUSIBLE_SESSION_MS - 1}
        result = TelemetryCorrelator().correlate(ctx, [])
        assert result["timing_valid"] is False
        assert any("implausibly short" in a for a in result["anomalies"])

    def test_multiple_anomalies_penalty_capped(self):
        ctx = {
            "humanity_score": 0.96,
            "inter_keystroke_min_ms": 10.0,       # anomaly 1
            "typing_cadence_std": 1.0,             # anomaly 2
            "typing_cadence_mean_ms": 1000.0,
            "session_duration_ms": 100.0,          # anomaly 3
        }
        result = TelemetryCorrelator().correlate(ctx, [])
        assert len(result["anomalies"]) == 3
        # 3 × 0.05 = 0.15, which equals MAX_PENALTY
        assert result["penalty"] == pytest.approx(0.15)

    def test_missing_telemetry_fields_ignored(self):
        # Only humanity_score — no timing fields — should produce no anomalies
        result = TelemetryCorrelator().correlate({"humanity_score": 0.95}, [])
        assert result["timing_valid"] is True
        assert result["penalty"] == 0.0


# ── AttestationPipeline ───────────────────────────────────────────────────────

class TestAttestationPipeline:
    def test_no_bundle_uses_humanity_score(self):
        result = AttestationPipeline().run(CLEAN_CONTEXT, None)
        assert result["composite_provenance_score"] == pytest.approx(0.96)
        assert result["escalate"] is False
        assert result["crypto_bonus"] == 0.0

    def test_valid_tls_proof_adds_bonus(self):
        bundle = {"tlsnotary_proof": VALID_TLS_PROOF}
        result = AttestationPipeline().run(CLEAN_CONTEXT, bundle)
        assert result["composite_provenance_score"] == pytest.approx(0.96 + 0.03)
        assert result["escalate"] is False

    def test_both_valid_methods_add_double_bonus(self):
        bundle = {
            "tlsnotary_proof": VALID_TLS_PROOF,
            "enclave_attestation": VALID_ENCLAVE_ATTESTATION,
        }
        result = AttestationPipeline().run(CLEAN_CONTEXT, bundle)
        # 0.96 + 0.06 = 1.02, clamped to 1.0
        assert result["composite_provenance_score"] == pytest.approx(1.0)
        assert result["crypto_bonus"] == pytest.approx(0.06)

    def test_timing_anomaly_sets_escalate(self):
        ctx = {**CLEAN_CONTEXT, "inter_keystroke_min_ms": 10.0}
        result = AttestationPipeline().run(ctx, None)
        assert result["escalate"] is True
        assert result["timing_penalty"] == pytest.approx(0.05)
        assert result["composite_provenance_score"] == pytest.approx(0.96 - 0.05)

    def test_score_clamped_to_zero(self):
        ctx = {"humanity_score": 0.0, "inter_keystroke_min_ms": 10.0,
               "session_duration_ms": 100.0, "typing_cadence_std": 1.0,
               "typing_cadence_mean_ms": 1000.0}
        result = AttestationPipeline().run(ctx, None)
        assert result["composite_provenance_score"] == pytest.approx(0.0)

    def test_score_clamped_to_one(self):
        ctx = {"humanity_score": 1.0}
        bundle = {
            "tlsnotary_proof": VALID_TLS_PROOF,
            "enclave_attestation": VALID_ENCLAVE_ATTESTATION,
        }
        result = AttestationPipeline().run(ctx, bundle)
        assert result["composite_provenance_score"] == pytest.approx(1.0)


# ── E2E integration: timing anomaly forces REVIEW_REQUIRED ───────────────────

from fastapi.testclient import TestClient
from app.main import app
from app.db import Base, engine

client = TestClient(app)
API_HEADERS = {"X-Maestro-API-Key": "dev-insecure-key-change-me"}


@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def _setup_agent_human_request():
    """Helper: registers an agent, human, consent, and grounding request. Returns (req_id, human_id)."""
    agent_id = client.post("/api/agents", json={"display_name": "A"}, headers=API_HEADERS).json()["id"]
    human_id = client.post("/api/humans", json={"display_name": "H"}).json()["id"]
    consent = {
        "allowed_axes": ["causal_reasoning"], "max_task_duration_minutes": 60,
        "privacy_level": "standard", "allow_model_training": True,
        "allow_public_aggregate": False, "min_hourly_equivalent_usd": 50.0,
        "payout_threshold_usd": 25.0, "payout_method": "crypto",
        "daily_max_contributions": 10, "weekly_max_hours": 8,
        "require_full_provenance": True, "revocation_window_days": 90,
    }
    client.post(f"/api/consent-profiles/{human_id}", json=consent)
    req_payload = {
        "agent_id": agent_id, "title": "T", "description": "D",
        "lived_experience_axis": "causal_reasoning", "requested_modality": "text",
        "protocol_instructions": "P", "estimated_duration_minutes": 10,
        "compensation_amount": 10.0, "effective_hourly_estimate": 60.0,
        "entropy_reduction_hypothesis": "H1", "phase_synchronization_hypothesis": "H2",
        "evidence_artifact_description": "E1", "contributions_required": 1,
        "viability_policy_snapshot": {},
    }
    req_id = client.post("/api/grounding-requests", json=req_payload, headers=API_HEADERS).json()["id"]
    client.post(f"/api/grounding-requests/{req_id}/evaluate", headers=API_HEADERS)
    return req_id, human_id


def test_clean_contribution_passes():
    """A contribution with clean telemetry and high humanity_score should PASS."""
    req_id, human_id = _setup_agent_human_request()
    contrib = {
        "human_id": human_id, "request_id": req_id,
        "content_payload": {"obs": 1}, "integrity_hash": "h1",
        "creation_context": {
            "humanity_score": 0.97,
            "inter_keystroke_min_ms": 150.0,
            "typing_cadence_std": 80.0,
            "typing_cadence_mean_ms": 200.0,
            "session_duration_ms": 30_000.0,
        },
    }
    contrib_id = client.post("/api/contributions", json=contrib, headers=API_HEADERS).json()["id"]
    res = client.post(f"/api/contributions/{contrib_id}/evaluate", headers=API_HEADERS)
    assert res.status_code == 200
    assert res.json()["status"] == "PASS"


def test_spoofed_timing_escalates_to_review():
    """A contribution whose humanity_score passes (0.97) but whose timing data
    indicates automation should be routed to REVIEW_REQUIRED."""
    req_id, human_id = _setup_agent_human_request()
    contrib = {
        "human_id": human_id, "request_id": req_id,
        "content_payload": {"obs": 1}, "integrity_hash": "h1",
        "creation_context": {
            "humanity_score": 0.97,
            # Sub-human reaction time — impossible for a genuine human contributor
            "inter_keystroke_min_ms": 5.0,
        },
    }
    contrib_id = client.post("/api/contributions", json=contrib, headers=API_HEADERS).json()["id"]
    res = client.post(f"/api/contributions/{contrib_id}/evaluate", headers=API_HEADERS)
    assert res.status_code == 200
    assert res.json()["status"] == "REVIEW_REQUIRED"


def test_valid_tlsnotary_proof_boosts_score():
    """A contribution with a valid TLSNotary proof should receive a score bonus
    reflected in the audit trail (composite_provenance_score > humanity_score)."""
    req_id, human_id = _setup_agent_human_request()
    contrib = {
        "human_id": human_id, "request_id": req_id,
        "content_payload": {"obs": 1}, "integrity_hash": "h1",
        "creation_context": {"humanity_score": 0.96},
        "attestation_bundle": {
            "tlsnotary_proof": {
                "session_id": "sess-001",
                "timestamp_utc": 1_700_000_000.0,
                "server_name": "api.maestro.io",
                "notary_signature": "sig_abc",
            }
        },
    }
    contrib_id = client.post("/api/contributions", json=contrib, headers=API_HEADERS).json()["id"]
    res = client.post(f"/api/contributions/{contrib_id}/evaluate", headers=API_HEADERS)
    assert res.status_code == 200
    # Score = 0.96 + 0.03 = 0.99 → well above both thresholds → PASS
    assert res.json()["status"] == "PASS"


def test_timing_anomaly_with_crypto_proof_still_escalates():
    """Timing anomalies escalate to REVIEW_REQUIRED even when cryptographic
    proofs are valid — crypto cannot override behavioral red flags."""
    req_id, human_id = _setup_agent_human_request()
    contrib = {
        "human_id": human_id, "request_id": req_id,
        "content_payload": {"obs": 1}, "integrity_hash": "h1",
        "creation_context": {
            "humanity_score": 0.97,
            "inter_keystroke_min_ms": 5.0,  # anomaly
        },
        "attestation_bundle": {
            "tlsnotary_proof": {
                "session_id": "sess-002",
                "timestamp_utc": 1_700_000_001.0,
                "server_name": "api.maestro.io",
                "notary_signature": "sig_xyz",
            }
        },
    }
    contrib_id = client.post("/api/contributions", json=contrib, headers=API_HEADERS).json()["id"]
    res = client.post(f"/api/contributions/{contrib_id}/evaluate", headers=API_HEADERS)
    assert res.status_code == 200
    assert res.json()["status"] == "REVIEW_REQUIRED"


# -- Bug-fix tests: policy-driven thresholds (Bug 2) --------------------------

class TestPolicyDrivenThresholds:
    """Verify TelemetryCorrelator and AttestationPipeline read thresholds
    from the attestation_policy dict rather than hardcoded constants."""

    def test_custom_penalty_rate_applied(self):
        policy = {"penalty_per_anomaly": 0.20, "min_human_reaction_ms": 100.0}
        result = TelemetryCorrelator(attestation_policy=policy).correlate(
            {**CLEAN_CONTEXT, "inter_keystroke_min_ms": 10.0}, []
        )
        assert result["penalty"] == pytest.approx(0.20)

    def test_raised_min_human_reaction_ms_flags_200ms_keystroke(self):
        policy = {"min_human_reaction_ms": 300.0}
        result = TelemetryCorrelator(attestation_policy=policy).correlate(
            {**CLEAN_CONTEXT, "inter_keystroke_min_ms": 200.0}, []
        )
        assert result["timing_valid"] is False
        assert len(result["anomalies"]) == 1

    def test_default_threshold_passes_200ms_keystroke(self):
        result = TelemetryCorrelator().correlate(
            {**CLEAN_CONTEXT, "inter_keystroke_min_ms": 200.0}, []
        )
        assert result["timing_valid"] is True

    def test_custom_max_penalty_is_three_times_per_anomaly(self):
        policy = {"penalty_per_anomaly": 0.10}
        ctx = {
            "humanity_score": 0.96,
            "inter_keystroke_min_ms": 10.0,
            "typing_cadence_std": 1.0,
            "typing_cadence_mean_ms": 1000.0,
            "session_duration_ms": 100.0,
        }
        result = TelemetryCorrelator(attestation_policy=policy).correlate(ctx, [])
        assert len(result["anomalies"]) == 3
        assert result["penalty"] == pytest.approx(0.30)

    def test_pipeline_reads_crypto_bonus_from_policy(self):
        policy = {"crypto_bonus_per_valid_method": 0.10}
        result = AttestationPipeline(attestation_policy=policy).run(
            CLEAN_CONTEXT, {"tlsnotary_proof": VALID_TLS_PROOF}
        )
        assert result["crypto_bonus"] == pytest.approx(0.10)
        assert result["composite_provenance_score"] == pytest.approx(
            min(1.0, 0.96 + 0.10)
        )

    def test_pipeline_propagates_thresholds_to_correlator(self):
        policy = {"min_human_reaction_ms": 300.0, "penalty_per_anomaly": 0.20}
        result = AttestationPipeline(attestation_policy=policy).run(
            {**CLEAN_CONTEXT, "inter_keystroke_min_ms": 200.0}, None
        )
        assert result["escalate"] is True
        assert result["timing_penalty"] == pytest.approx(0.20)


# -- Bug-fix tests: require_cryptographic_attestation flag (Bug 3) ------------

class TestRequireCryptographicAttestation:
    """Verify that the require_cryptographic_attestation flag is enforced."""

    def test_flag_false_no_proof_no_escalation(self):
        result = AttestationPipeline(
            attestation_policy={"require_cryptographic_attestation": False}
        ).run(CLEAN_CONTEXT, None)
        assert result["escalate"] is False
        assert result["crypto_requirement_unmet"] is False

    def test_flag_true_no_bundle_escalates(self):
        result = AttestationPipeline(
            attestation_policy={"require_cryptographic_attestation": True}
        ).run(CLEAN_CONTEXT, None)
        assert result["escalate"] is True
        assert result["no_valid_crypto_proof"] is True
        assert result["crypto_requirement_unmet"] is True

    def test_flag_true_structurally_invalid_proof_still_escalates(self):
        result = AttestationPipeline(
            attestation_policy={"require_cryptographic_attestation": True}
        ).run(CLEAN_CONTEXT, {"tlsnotary_proof": {"session_id": "incomplete"}})
        assert result["escalate"] is True
        assert result["crypto_requirement_unmet"] is True

    def test_flag_true_valid_tls_proof_satisfies_requirement(self):
        result = AttestationPipeline(
            attestation_policy={"require_cryptographic_attestation": True}
        ).run(CLEAN_CONTEXT, {"tlsnotary_proof": VALID_TLS_PROOF})
        assert result["crypto_requirement_unmet"] is False
        assert result["escalate"] is False

    def test_flag_true_valid_enclave_satisfies_requirement(self):
        result = AttestationPipeline(
            attestation_policy={"require_cryptographic_attestation": True}
        ).run(CLEAN_CONTEXT, {"enclave_attestation": VALID_ENCLAVE_ATTESTATION})
        assert result["crypto_requirement_unmet"] is False
        assert result["escalate"] is False

    def test_timing_anomaly_escalates_independently_of_crypto_flag(self):
        result = AttestationPipeline(
            attestation_policy={"require_cryptographic_attestation": True}
        ).run(
            {**CLEAN_CONTEXT, "inter_keystroke_min_ms": 5.0},
            {"tlsnotary_proof": VALID_TLS_PROOF},
        )
        assert result["crypto_requirement_unmet"] is False
        assert result["escalate"] is True

    def test_score_unaffected_by_crypto_requirement_flag(self):
        score_on = AttestationPipeline(
            attestation_policy={"require_cryptographic_attestation": True}
        ).run(CLEAN_CONTEXT, None)["composite_provenance_score"]
        score_off = AttestationPipeline(
            attestation_policy={"require_cryptographic_attestation": False}
        ).run(CLEAN_CONTEXT, None)["composite_provenance_score"]
        assert score_on == pytest.approx(score_off)
