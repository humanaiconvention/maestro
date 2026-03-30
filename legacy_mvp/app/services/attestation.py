"""Phase D — Cryptographic Attestation Pipeline

Implements the three-stage defense-in-depth verification described in the
Phase D architecture: primary cryptographic verifiers, behavioral cross-checks,
and escalation fallback.

All verifiers are *simulated*: they validate the structural shape of incoming
attestation payloads and apply deterministic scoring rules that mirror the
interface real cryptographic libraries will eventually satisfy.  No real ZK,
enclave, or TLS verification is performed in this stub implementation.

Pipeline stages:
  1. TLSNotaryVerifier   — validates presence and structure of a TLSNotary proof
  2. SecureEnclaveVerifier — validates presence and structure of an enclave quote
  3. TelemetryCorrelator  — cross-checks behavioral timing against crypto results
  4. AttestationPipeline  — orchestrates all three → composite provenance_score

All tuneable parameters are read from the attestation sub-section of
viability_policy.yaml (contribution_gates.attestation) at runtime.  Module-level
constants serve as fallback defaults so the pipeline remains usable in unit
tests and environments where no policy file is present.
"""
from __future__ import annotations

from typing import Any

# ── Module-level defaults (mirrors viability_policy.yaml contribution_gates.attestation) ──
# These values are used when no policy dict is provided (unit tests, standalone
# use).  The policy YAML is the single source of truth for production runs.
MIN_HUMAN_REACTION_MS: float = 100.0
AUTOMATION_REGULARITY_THRESHOLD: float = 0.05
MIN_PLAUSIBLE_SESSION_MS: float = 5_000.0
PENALTY_PER_ANOMALY: float = 0.05
CRYPTO_BONUS_PER_VALID_METHOD: float = 0.03


class TLSNotaryVerifier:
    """Validates a simulated TLSNotary proof.

    A real TLSNotary proof cryptographically binds a TLS session transcript to
    a specific server and timestamp, proving the data was observed over a live
    network connection rather than fabricated offline.

    Simulated validation rules:
      - Proof must be a dict with keys: session_id, timestamp_utc, server_name,
        notary_signature.
      - notary_signature must be a non-empty string (would be verified against
        a trusted notary public key in production).
      - timestamp_utc must be a positive numeric epoch value.
    """

    def verify(self, bundle: dict[str, Any]) -> dict[str, Any]:
        proof = bundle.get("tlsnotary_proof")
        if not proof:
            return {
                "method": "tlsnotary", "present": False, "valid": False,
                "confidence": 0.0, "detail": "No TLSNotary proof supplied.",
            }

        required_keys = {"session_id", "timestamp_utc", "server_name", "notary_signature"}
        missing = required_keys - set(proof.keys())
        if missing:
            return {
                "method": "tlsnotary", "present": True, "valid": False,
                "confidence": 0.0,
                "detail": f"Proof missing required fields: {sorted(missing)}",
            }

        if not proof.get("notary_signature"):
            return {
                "method": "tlsnotary", "present": True, "valid": False,
                "confidence": 0.0, "detail": "Notary signature is empty.",
            }

        try:
            ts = float(proof["timestamp_utc"])
            if ts <= 0:
                raise ValueError("non-positive timestamp")
        except (TypeError, ValueError):
            return {
                "method": "tlsnotary", "present": True, "valid": False,
                "confidence": 0.0, "detail": "Invalid timestamp_utc.",
            }

        return {
            "method": "tlsnotary", "present": True, "valid": True,
            "confidence": 1.0,
            "detail": f"TLSNotary proof valid for server '{proof['server_name']}'.",
        }


class SecureEnclaveVerifier:
    """Validates a simulated Secure Enclave attestation quote.

    A real enclave attestation (e.g. Intel SGX DCAP or Apple Secure Enclave)
    proves that a piece of code ran inside tamper-resistant hardware and that
    its measurements match an expected binary.  This makes it extremely
    difficult to inject synthetic data at the hardware boundary.

    Simulated validation rules:
      - Attestation must be a dict with keys: enclave_id, quote, pcr_values, nonce.
      - pcr_values must be a non-empty dict (would be compared against a
        known-good measurement in production).
      - quote must be a non-empty string.
      - nonce must be present (prevents replay attacks).
    """

    def verify(self, bundle: dict[str, Any]) -> dict[str, Any]:
        attestation = bundle.get("enclave_attestation")
        if not attestation:
            return {
                "method": "secure_enclave", "present": False, "valid": False,
                "confidence": 0.0, "detail": "No enclave attestation supplied.",
            }

        required_keys = {"enclave_id", "quote", "pcr_values", "nonce"}
        missing = required_keys - set(attestation.keys())
        if missing:
            return {
                "method": "secure_enclave", "present": True, "valid": False,
                "confidence": 0.0,
                "detail": f"Attestation missing required fields: {sorted(missing)}",
            }

        if not attestation.get("quote"):
            return {
                "method": "secure_enclave", "present": True, "valid": False,
                "confidence": 0.0, "detail": "Enclave quote is empty.",
            }

        pcr = attestation.get("pcr_values")
        if not isinstance(pcr, dict) or not pcr:
            return {
                "method": "secure_enclave", "present": True, "valid": False,
                "confidence": 0.0,
                "detail": "pcr_values must be a non-empty dict.",
            }

        return {
            "method": "secure_enclave", "present": True, "valid": True,
            "confidence": 1.0,
            "detail": f"Enclave attestation valid for enclave '{attestation['enclave_id']}'.",
        }


class TelemetryCorrelator:
    """Cross-checks behavioral telemetry against cryptographic proof results.

    Even valid cryptographic proofs can be spoofed if an attacker controls the
    hardware or the notary.  The correlator adds a secondary signal by checking
    whether the *behavioral* telemetry in creation_context is consistent with
    genuine human interaction.

    Anomaly detection (applied to creation_context keys):
      - inter_keystroke_min_ms < min_human_reaction_ms:
            keystrokes faster than human reaction time → automation flag
      - typing_cadence_std / typing_cadence_mean_ms < automation_regularity_threshold:
            typing too metronomically regular → automation flag
      - session_duration_ms < min_plausible_session_ms:
            session implausibly short → automation flag

    All thresholds are read from the attestation policy dict at construction
    time, falling back to module-level defaults when no policy is provided.

    The score penalty scales with the number of anomalies detected:
    penalty_per_anomaly per anomaly, capped at penalty_per_anomaly * 3.
    """

    def __init__(self, attestation_policy: dict[str, Any] | None = None) -> None:
        p = attestation_policy or {}
        self._min_human_reaction_ms = float(
            p.get("min_human_reaction_ms", MIN_HUMAN_REACTION_MS)
        )
        self._min_plausible_session_ms = float(
            p.get("min_plausible_session_ms", MIN_PLAUSIBLE_SESSION_MS)
        )
        self._automation_regularity_threshold = float(
            p.get("automation_regularity_threshold", AUTOMATION_REGULARITY_THRESHOLD)
        )
        self._penalty_per_anomaly = float(
            p.get("penalty_per_anomaly", PENALTY_PER_ANOMALY)
        )
        # Maximum penalty capped at three simultaneous anomalies
        self._max_penalty = self._penalty_per_anomaly * 3

    def correlate(
        self,
        creation_context: dict[str, Any],
        crypto_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        anomalies: list[str] = []

        # 1. Minimum inter-keystroke interval
        min_ik = creation_context.get("inter_keystroke_min_ms")
        if min_ik is not None:
            try:
                if float(min_ik) < self._min_human_reaction_ms:
                    anomalies.append(
                        f"inter_keystroke_min_ms={min_ik} < {self._min_human_reaction_ms}ms "
                        f"(sub-human reaction time)"
                    )
            except (TypeError, ValueError):
                pass

        # 2. Typing regularity — coefficient of variation (std / mean)
        cadence_std = creation_context.get("typing_cadence_std")
        cadence_mean = creation_context.get("typing_cadence_mean_ms")
        if cadence_std is not None and cadence_mean is not None:
            try:
                std_f = float(cadence_std)
                mean_f = float(cadence_mean)
                if mean_f > 0:
                    cv = std_f / mean_f
                    if cv < self._automation_regularity_threshold:
                        anomalies.append(
                            f"typing CV={cv:.4f} < {self._automation_regularity_threshold} "
                            f"(metronomically regular — likely automated)"
                        )
            except (TypeError, ValueError):
                pass

        # 3. Minimum plausible session duration
        session_ms = creation_context.get("session_duration_ms")
        if session_ms is not None:
            try:
                if float(session_ms) < self._min_plausible_session_ms:
                    anomalies.append(
                        f"session_duration_ms={session_ms} < {self._min_plausible_session_ms}ms "
                        f"(implausibly short session)"
                    )
            except (TypeError, ValueError):
                pass

        penalty = min(len(anomalies) * self._penalty_per_anomaly, self._max_penalty)

        return {
            "timing_valid": len(anomalies) == 0,
            "anomalies": anomalies,
            "penalty": penalty,
        }


class AttestationPipeline:
    """Orchestrates the multi-factor cryptographic attestation evaluation.

    All tuneable parameters are read from the attestation policy dict passed at
    construction time.  When no policy is provided (e.g. in unit tests) the
    module-level defaults are used.

    Stages:
      1. Run TLSNotaryVerifier on attestation_bundle (if present).
      2. Run SecureEnclaveVerifier on attestation_bundle (if present).
      3. Optionally enforce require_cryptographic_attestation: if the policy
         flag is True and no valid proof is present, escalate regardless of score.
      4. Run TelemetryCorrelator cross-check against creation_context.
      5. Compute composite provenance_score:
             base_score  = creation_context.get("humanity_score", 0.0)
             + crypto_bonus_per_valid_method per valid primary verifier
             − timing_penalty
             clamped to [0.0, 1.0].
      6. Set escalate=True if timing anomalies detected OR crypto requirement
         is unmet — the caller routes escalated contributions to REVIEW_REQUIRED.

    Returns a dict that the caller uses to replace the raw humanity_score with
    the composite score before passing the contribution to the viability engine.
    """

    def __init__(self, attestation_policy: dict[str, Any] | None = None) -> None:
        p = attestation_policy or {}
        self._require_crypto = bool(
            p.get("require_cryptographic_attestation", False)
        )
        self._crypto_bonus = float(
            p.get("crypto_bonus_per_valid_method", CRYPTO_BONUS_PER_VALID_METHOD)
        )
        self._tlsnotary = TLSNotaryVerifier()
        self._enclave = SecureEnclaveVerifier()
        self._correlator = TelemetryCorrelator(attestation_policy=p)

    def run(
        self,
        creation_context: dict[str, Any],
        attestation_bundle: dict[str, Any] | None,
    ) -> dict[str, Any]:
        bundle = attestation_bundle or {}

        # Stages 1 & 2: primary cryptographic verifiers
        tls_result = self._tlsnotary.verify(bundle)
        enclave_result = self._enclave.verify(bundle)
        crypto_results = [tls_result, enclave_result]

        # Stage 3: enforce crypto requirement flag
        no_valid_crypto = not any(r.get("valid") for r in crypto_results)
        crypto_requirement_unmet = self._require_crypto and no_valid_crypto

        # Stage 4: behavioral cross-check
        correlation = self._correlator.correlate(creation_context, crypto_results)

        # Stage 5: composite score
        base_score = float(creation_context.get("humanity_score", 0.0))
        crypto_bonus = sum(
            self._crypto_bonus for r in crypto_results if r.get("valid")
        )
        composite_score = max(
            0.0, min(1.0, base_score + crypto_bonus - correlation["penalty"])
        )

        # Stage 6: escalation — timing anomalies OR unmet crypto requirement
        escalate = (not correlation["timing_valid"]) or crypto_requirement_unmet

        return {
            "composite_provenance_score": composite_score,
            "escalate": escalate,
            "verifier_results": crypto_results,
            "correlation": correlation,
            "base_humanity_score": base_score,
            "crypto_bonus": crypto_bonus,
            "timing_penalty": correlation["penalty"],
            "no_valid_crypto_proof": no_valid_crypto,
            "crypto_requirement_unmet": crypto_requirement_unmet,
        }
