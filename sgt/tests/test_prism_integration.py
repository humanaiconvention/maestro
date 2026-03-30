"""Tests for the PRISM integration layer.

All tests in this file are GPU-free and PRISM-free.
PRISM itself is mocked everywhere so these tests pass in any environment.

Coverage:
  TestPrismSchemas                   — shared data contracts (pure Python)
  TestRecursiveTrainerPrismHook      — hook observer (mocked take_sgt_snapshot)
  TestEarlyWarningDetectorGeometric  — geometric trajectory extension in drift_metrics.py
  TestSettlementGate                 — entropy_measurement_consistency_gate in engines.py
"""

import os
import sys
import time
import math
import json
import pytest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Path setup — resolve imports from the monorepo without pip install
# ---------------------------------------------------------------------------
_SGT_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "../src"))
_MAESTRO_ADAPTERS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../libs/adapters")
)
_MAESTRO_SCHEMAS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../libs/schemas")
)
_MAESTRO_APP = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../legacy_mvp")
)
for _p in [_SGT_SRC, _MAESTRO_ADAPTERS, _MAESTRO_SCHEMAS, _MAESTRO_APP]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from prism_schemas import (
    EntropySnapshot, EntropyDeltaProof, GeometricHealthScore, LayerEntropyProfile,
)
from semantic_grounding.reporting.drift_metrics import EarlyWarningDetector
from app.services.viability.engines import SettlementViabilityEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(gen_idx: int = 0, spectral_entropy: float = 3.0,
                   effective_dimension: float = 200.0, viability: float = 0.5,
                   coherence: float = 0.8) -> EntropySnapshot:
    return EntropySnapshot(
        timestamp=time.time(),
        model_name="test-model",
        generation_idx=gen_idx,
        regime="R1",
        mean_spectral_entropy=spectral_entropy,
        mean_effective_dimension=effective_dimension,
        mean_viability_score=viability,
        mean_fisher_curvature=1.5,
        layer_profiles=[
            LayerEntropyProfile(
                layer_idx=4,
                spectral_entropy=spectral_entropy,
                effective_dimension=effective_dimension,
                viability_score=viability,
                fisher_curvature=1.5,
            )
        ],
        mean_phase_coherence=coherence,
        noise_sensitivity=0.1,
        n_samples=5,
    )


def _make_delta(delta_se: float, epsilon: float = 0.01) -> EntropyDeltaProof:
    before = _make_snapshot(gen_idx=0, spectral_entropy=3.0)
    after = _make_snapshot(gen_idx=1, spectral_entropy=3.0 + delta_se)
    return EntropyDeltaProof(
        snapshot_before=before,
        snapshot_after=after,
        delta_spectral_entropy=delta_se,
        delta_effective_dimension=-delta_se * 10,  # correlated
        delta_viability_score=-delta_se * 0.01,
        delta_phase_coherence=-delta_se * 0.05,
        epsilon_threshold=epsilon,
        reduction_verified=delta_se < -epsilon,
    )


# ---------------------------------------------------------------------------
# TestPrismSchemas — pure data contract tests (no mocks needed)
# ---------------------------------------------------------------------------

class TestPrismSchemas:
    """Tests for EntropySnapshot, EntropyDeltaProof, GeometricHealthScore."""

    def test_entropy_snapshot_roundtrip(self):
        """to_dict() / from_dict() round-trip preserves all scalar fields."""
        snap = _make_snapshot(gen_idx=2, spectral_entropy=4.1, effective_dimension=150.0)
        d = snap.to_dict()
        snap2 = EntropySnapshot.from_dict(d)

        assert snap2.generation_idx == snap.generation_idx
        assert snap2.regime == snap.regime
        assert abs(snap2.mean_spectral_entropy - snap.mean_spectral_entropy) < 1e-9
        assert abs(snap2.mean_effective_dimension - snap.mean_effective_dimension) < 1e-9
        assert abs(snap2.mean_viability_score - snap.mean_viability_score) < 1e-9
        assert abs(snap2.mean_phase_coherence - snap.mean_phase_coherence) < 1e-9
        assert abs(snap2.noise_sensitivity - snap.noise_sensitivity) < 1e-9
        assert snap2.n_samples == snap.n_samples

    def test_entropy_snapshot_layer_profiles_roundtrip(self):
        """Layer profiles are preserved through serialization."""
        snap = _make_snapshot()
        d = snap.to_dict()
        snap2 = EntropySnapshot.from_dict(d)
        assert len(snap2.layer_profiles) == 1
        lp = snap2.layer_profiles[0]
        assert lp.layer_idx == 4
        assert abs(lp.spectral_entropy - snap.mean_spectral_entropy) < 1e-9

    def test_entropy_delta_verified_when_negative(self):
        """delta_spectral_entropy < −ε → reduction_verified=True."""
        proof = _make_delta(delta_se=-0.5, epsilon=0.01)
        assert proof.reduction_verified is True

    def test_entropy_delta_not_verified_when_positive(self):
        """delta_spectral_entropy > 0 → reduction_verified=False."""
        proof = _make_delta(delta_se=0.2, epsilon=0.01)
        assert proof.reduction_verified is False

    def test_entropy_delta_not_verified_below_epsilon(self):
        """delta_spectral_entropy = −0.005 (< ε=0.01) → reduction_verified=False."""
        proof = _make_delta(delta_se=-0.005, epsilon=0.01)
        assert proof.reduction_verified is False

    def test_entropy_delta_roundtrip(self):
        """to_dict() preserves delta and verification flag."""
        proof = _make_delta(delta_se=-0.3)
        d = proof.to_dict()
        assert abs(d["delta_spectral_entropy"] - (-0.3)) < 1e-9
        assert d["reduction_verified"] is True

    def test_geometric_health_score_range(self):
        """composite_score is always in [0, 1]."""
        for vs, se, ns in [(0.9, 1.0, 0.05), (0.1, 7.0, 0.9), (0.5, 4.0, 0.5)]:
            ghs = GeometricHealthScore(
                viability_score=vs,
                spectral_entropy=se,
                noise_sensitivity=ns,
                composite_score=max(0.0, min(1.0, 1.0 - (0.4 * vs + 0.3 * 0.5 + 0.3 * 0.5))),
            )
            assert 0.0 <= ghs.composite_score <= 1.0, f"score={ghs.composite_score} out of range"

    def test_layer_entropy_profile_roundtrip(self):
        lp = LayerEntropyProfile(layer_idx=7, spectral_entropy=2.3, effective_dimension=128.0,
                                  viability_score=0.63, fisher_curvature=0.9)
        lp2 = LayerEntropyProfile.from_dict(lp.to_dict())
        assert lp2.layer_idx == 7
        assert abs(lp2.spectral_entropy - 2.3) < 1e-9


# ---------------------------------------------------------------------------
# TestRecursiveTrainerPrismHook — hook observer tests (mocked PRISM)
# ---------------------------------------------------------------------------

class TestRecursiveTrainerPrismHook:
    """Tests for RecursiveTrainerPrismHook — no model or GPU needed."""

    def _make_hook_with_mock(self, n_snapshots_to_generate=0):
        """Create a hook with take_sgt_snapshot fully mocked."""
        from semantic_grounding.prism_integration.hooks import RecursiveTrainerPrismHook

        trainer_mock = MagicMock()
        trainer_mock.model = MagicMock()
        trainer_mock.tokenizer = MagicMock()
        trainer_mock.device = "cpu"

        eval_data = [{"prompt": f"What colour is the sky? Q{i}"} for i in range(10)]
        hook = RecursiveTrainerPrismHook(trainer=trainer_mock, eval_data=eval_data,
                                         regime="R1", n_samples=3)

        def fake_snapshot(trainer, eval_data, generation_idx, regime, **kwargs):
            return _make_snapshot(gen_idx=generation_idx)

        hook._snapshot_fn = fake_snapshot
        return hook, fake_snapshot

    def test_hook_records_one_snapshot_per_generation(self):
        """3 calls to on_generation_end → 3 snapshots recorded."""
        from semantic_grounding.prism_integration.hooks import RecursiveTrainerPrismHook

        trainer_mock = MagicMock()
        eval_data = [{"prompt": "Q1"}, {"prompt": "Q2"}]

        with patch("semantic_grounding.prism_integration.hooks.take_sgt_snapshot",
                   side_effect=lambda **kw: _make_snapshot(gen_idx=kw["generation_idx"])):
            hook = RecursiveTrainerPrismHook(trainer_mock, eval_data, regime="R1", n_samples=2)
            for i in range(3):
                hook.on_generation_end(i)

        assert len(hook.snapshots) == 3
        assert [s.generation_idx for s in hook.snapshots] == [0, 1, 2]

    def test_hook_computes_deltas(self):
        """3 snapshots → exactly 2 consecutive deltas."""
        from semantic_grounding.prism_integration.hooks import RecursiveTrainerPrismHook

        trainer_mock = MagicMock()
        eval_data = [{"prompt": "Q1"}]

        with patch("semantic_grounding.prism_integration.hooks.take_sgt_snapshot",
                   side_effect=lambda **kw: _make_snapshot(gen_idx=kw["generation_idx"])):
            hook = RecursiveTrainerPrismHook(trainer_mock, eval_data, regime="R1", n_samples=1)
            for i in range(3):
                hook.on_generation_end(i)

        assert len(hook.deltas) == 2

    def test_hook_on_generation_end_returns_snapshot(self):
        """on_generation_end() returns an EntropySnapshot instance."""
        from semantic_grounding.prism_integration.hooks import RecursiveTrainerPrismHook

        trainer_mock = MagicMock()
        eval_data = [{"prompt": "Q1"}]

        with patch("semantic_grounding.prism_integration.hooks.take_sgt_snapshot",
                   return_value=_make_snapshot(gen_idx=0)):
            hook = RecursiveTrainerPrismHook(trainer_mock, eval_data, regime="R1", n_samples=1)
            result = hook.on_generation_end(0)

        assert isinstance(result, EntropySnapshot)

    def test_hook_trajectory_report_keys(self):
        """get_trajectory_report() has 'snapshots', 'deltas', 'trajectory', 'errors' keys."""
        from semantic_grounding.prism_integration.hooks import RecursiveTrainerPrismHook

        trainer_mock = MagicMock()
        eval_data = [{"prompt": "Q1"}]

        with patch("semantic_grounding.prism_integration.hooks.take_sgt_snapshot",
                   side_effect=lambda **kw: _make_snapshot(gen_idx=kw["generation_idx"])):
            hook = RecursiveTrainerPrismHook(trainer_mock, eval_data, regime="R1", n_samples=1)
            for i in range(2):
                hook.on_generation_end(i)

        report = hook.get_trajectory_report()
        assert "snapshots" in report
        assert "deltas" in report
        assert "trajectory" in report
        assert "errors" in report
        assert len(report["trajectory"]) == 2
        assert all("spectral_entropy" in row for row in report["trajectory"])

    def test_hook_returns_none_on_prism_unavailable(self):
        """Hook returns None gracefully if PRISM is not installed."""
        from semantic_grounding.prism_integration.hooks import RecursiveTrainerPrismHook
        from prism_adapter import PrismUnavailableError

        trainer_mock = MagicMock()
        eval_data = [{"prompt": "Q1"}]

        with patch("semantic_grounding.prism_integration.hooks.take_sgt_snapshot",
                   side_effect=PrismUnavailableError("not installed")):
            hook = RecursiveTrainerPrismHook(trainer_mock, eval_data, regime="R1", n_samples=1)
            result = hook.on_generation_end(0)

        assert result is None
        assert len(hook.errors) == 1
        assert hook.errors[0][0] == 0  # gen_idx recorded

    def test_hook_as_metrics_rows_format(self):
        """as_metrics_rows() returns list of dicts with expected keys."""
        from semantic_grounding.prism_integration.hooks import RecursiveTrainerPrismHook

        trainer_mock = MagicMock()
        eval_data = [{"prompt": "Q1"}]

        with patch("semantic_grounding.prism_integration.hooks.take_sgt_snapshot",
                   side_effect=lambda **kw: _make_snapshot(gen_idx=kw["generation_idx"],
                                                            spectral_entropy=2.0 + kw["generation_idx"] * 0.1)):
            hook = RecursiveTrainerPrismHook(trainer_mock, eval_data, regime="R2", n_samples=1)
            for i in range(3):
                hook.on_generation_end(i)

        rows = hook.as_metrics_rows()
        assert len(rows) == 3
        for row in rows:
            assert "spectral_entropy" in row
            assert "effective_dimension" in row
            assert "viability_score" in row


# ---------------------------------------------------------------------------
# TestEarlyWarningDetectorGeometric — drift_metrics geometric extension
# ---------------------------------------------------------------------------

class TestEarlyWarningDetectorGeometric:
    """Tests for _analyze_geometric_trajectory() added to EarlyWarningDetector."""

    def _make_trajectory(self, spectral_entropies, effective_dims, base_acc=0.9):
        """Build a metrics history list with both behavioral and PRISM keys."""
        results = []
        for i, (se, ed) in enumerate(zip(spectral_entropies, effective_dims)):
            results.append({
                "generation": i,
                "accuracy": base_acc - i * 0.02,   # gentle behavioral decay
                "perplexity": 10.0 + i * 0.1,
                "spectral_entropy": se,
                "effective_dimension": ed,
            })
        return results

    def test_geometric_drift_detected_rising_entropy(self):
        """Rising spectral entropy beyond threshold triggers geometric_drift_detected=True."""
        detector = EarlyWarningDetector(accuracy_threshold=0.10, perplexity_threshold=0.10)
        # Entropy rises 50% by gen 2 (> 10% threshold)
        trajectory = self._make_trajectory(
            spectral_entropies=[2.0, 2.5, 3.0],
            effective_dims=[200.0, 190.0, 180.0],
        )
        result = detector.analyze_trajectory(trajectory)
        assert "geometric_drift_detected" in result
        assert result["geometric_drift_detected"] is True
        assert result["geometric_failure_generation"] != -1

    def test_geometric_drift_absent_stable_entropy(self):
        """Stable entropy trajectory → geometric_drift_detected=False."""
        detector = EarlyWarningDetector(accuracy_threshold=0.10, perplexity_threshold=0.10)
        trajectory = self._make_trajectory(
            spectral_entropies=[2.0, 2.01, 2.02],  # < 10% rise
            effective_dims=[200.0, 199.5, 199.0],
        )
        result = detector.analyze_trajectory(trajectory)
        assert "geometric_drift_detected" in result
        assert result["geometric_drift_detected"] is False
        assert result["geometric_failure_generation"] == -1

    def test_geometric_keys_absent_when_no_prism_data(self):
        """When spectral_entropy is absent, geometric keys are NOT added to result."""
        detector = EarlyWarningDetector()
        standard_trajectory = [
            {"generation": 0, "accuracy": 0.9, "perplexity": 10.0},
            {"generation": 1, "accuracy": 0.85, "perplexity": 10.5},
        ]
        result = detector.analyze_trajectory(standard_trajectory)
        assert "geometric_drift_detected" not in result
        assert "spectral_entropy_trajectory" not in result

    def test_behavioral_and_geometric_both_reported(self):
        """When PRISM data is present, both behavioral and geometric keys appear."""
        detector = EarlyWarningDetector(accuracy_threshold=0.10, perplexity_threshold=0.10)
        trajectory = self._make_trajectory(
            spectral_entropies=[2.0, 3.5, 5.0],   # strong entropy rise
            effective_dims=[200.0, 150.0, 100.0],  # strong dim drop
            base_acc=0.9,
        )
        result = detector.analyze_trajectory(trajectory)
        # Behavioral keys
        assert "signature_detected" in result
        assert "failure_ordering" in result
        # Geometric keys
        assert "geometric_drift_detected" in result
        assert "spectral_entropy_trajectory" in result
        assert "effective_dimension_trajectory" in result
        assert len(result["spectral_entropy_trajectory"]) == 3

    def test_geometric_trajectory_includes_all_gens(self):
        """spectral_entropy_trajectory has one entry per generation."""
        detector = EarlyWarningDetector()
        trajectory = self._make_trajectory([1.0, 1.1, 1.2, 1.3, 1.4], [100.0] * 5)
        result = detector.analyze_trajectory(trajectory)
        assert len(result["spectral_entropy_trajectory"]) == 5
        assert len(result["effective_dimension_trajectory"]) == 5


# ---------------------------------------------------------------------------
# TestSettlementGate — entropy_measurement_consistency_gate in engines.py
# ---------------------------------------------------------------------------

class TestSettlementGate:
    """Tests for SettlementViabilityEngine._entropy_measurement_consistency_gate().

    Uses mocked prism_adapter so no actual PRISM installation is required.
    """

    def _engine(self, extra_policy: dict = None):
        policy = {"settlement_gates": {"entropy_epsilon": 0.01,
                                        "entropy_consistency_tolerance": 0.25}}
        if extra_policy:
            policy["settlement_gates"].update(extra_policy)
        return SettlementViabilityEngine(policy)

    def _settlement_data(self, claimed=0.10, snap_before=None, snap_after=None):
        data = {"claimed_entropy_reduction": claimed,
                "entropy_snapshot_before": snap_before,
                "entropy_snapshot_after": snap_after}
        return data

    def test_gate_backward_compatible_no_snapshots(self):
        """No snapshots → UNVERIFIED (does not block settlement)."""
        engine = self._engine()
        data = self._settlement_data(snap_before=None, snap_after=None)
        result = engine._entropy_measurement_consistency_gate(data)
        assert result["gate"] == "entropy_measurement_consistency"
        assert result["status"] == "UNVERIFIED"

    def _prism_sys_modules(self, proof, before, after):
        """Return a context manager that injects mock prism_adapter and prism_schemas
        into sys.modules, so the lazy ``from prism_adapter import ...`` inside the gate
        function resolves to our mocks instead of the real (potentially absent) PRISM."""
        mock_adapter = MagicMock()
        mock_adapter.compute_entropy_delta = MagicMock(return_value=proof)

        mock_schemas = MagicMock()
        mock_schemas.EntropySnapshot.from_dict.side_effect = (
            lambda d: before if d.get("generation_idx") == 0 else after
        )

        return patch.dict(
            "sys.modules",
            {"prism_adapter": mock_adapter, "prism_schemas": mock_schemas},
        )

    def test_gate_passes_with_verified_reduction(self):
        """ΔS < −ε, consistent with claimed → PASS."""
        before = _make_snapshot(gen_idx=0, spectral_entropy=3.0)
        after = _make_snapshot(gen_idx=1, spectral_entropy=2.5)   # ΔS = −0.5
        proof = _make_delta(delta_se=-0.5)

        engine = self._engine()
        with self._prism_sys_modules(proof, before, after):
            data = self._settlement_data(
                claimed=0.50,
                snap_before=before.to_dict(),
                snap_after=after.to_dict(),
            )
            result = engine._entropy_measurement_consistency_gate(data)

        assert result["status"] == "PASS"
        assert result["value"] is not None

    def test_gate_fails_if_no_reduction(self):
        """ΔS ≥ −ε → FAIL."""
        before = _make_snapshot(gen_idx=0, spectral_entropy=3.0)
        after = _make_snapshot(gen_idx=1, spectral_entropy=3.2)   # entropy INCREASED
        proof = _make_delta(delta_se=0.2, epsilon=0.01)   # not verified

        engine = self._engine()
        with self._prism_sys_modules(proof, before, after):
            data = self._settlement_data(
                claimed=0.10,
                snap_before=before.to_dict(),
                snap_after=after.to_dict(),
            )
            result = engine._entropy_measurement_consistency_gate(data)

        assert result["status"] == "FAIL"

    def test_gate_review_if_claimed_vs_measured_diverges(self):
        """Verified ΔS but >25% discrepancy with claimed → REVIEW_REQUIRED."""
        before = _make_snapshot(gen_idx=0, spectral_entropy=3.0)
        after = _make_snapshot(gen_idx=1, spectral_entropy=2.8)
        proof = _make_delta(delta_se=-0.2, epsilon=0.01)  # measured = 0.2

        engine = self._engine()
        with self._prism_sys_modules(proof, before, after):
            data = self._settlement_data(
                claimed=0.90,   # claimed 0.90 vs measured 0.20 = 78% discrepancy
                snap_before=before.to_dict(),
                snap_after=after.to_dict(),
            )
            result = engine._entropy_measurement_consistency_gate(data)

        assert result["status"] == "REVIEW_REQUIRED"

    def test_full_evaluate_with_snapshots_passes(self):
        """SettlementViabilityEngine.evaluate() adds the PRISM gate to results."""
        engine = self._engine()
        proof = _make_delta(delta_se=-0.5)
        before = _make_snapshot(gen_idx=0, spectral_entropy=3.0)
        after = _make_snapshot(gen_idx=1, spectral_entropy=2.5)

        with self._prism_sys_modules(proof, before, after):
            settlement_data = {
                "claimed_entropy_reduction": 0.50,
                "payout_amount": 50.0,
                "effective_hourly_estimate": 50.0,
                "duration_minutes": 60,
                "usage_rights": [],
                "allow_model_training": True,
                "entropy_snapshot_before": before.to_dict(),
                "entropy_snapshot_after": after.to_dict(),
            }
            result = engine.evaluate("req-123", settlement_data)

        gate_names = [g["gate"] for g in result["gates"]]
        assert "entropy_measurement_consistency" in gate_names

    def test_full_evaluate_without_snapshots_is_unverified_not_blocked(self):
        """No snapshots → overall_viability is NOT blocked by PRISM gate."""
        engine = self._engine()
        settlement_data = {
            "claimed_entropy_reduction": 0.10,
            "payout_amount": 50.0,
            "effective_hourly_estimate": 50.0,
            "duration_minutes": 60,
            "usage_rights": [],
            "allow_model_training": True,
            "entropy_snapshot_before": None,
            "entropy_snapshot_after": None,
        }
        result = engine.evaluate("req-456", settlement_data)
        prism_gate = next(g for g in result["gates"] if g["gate"] == "entropy_measurement_consistency")
        assert prism_gate["status"] == "UNVERIFIED"
        # Settlement should not be blocked solely by the PRISM gate being UNVERIFIED
        assert result["overall_viability"] is True
