"""Shared data contracts for PRISM entropy and geometric analysis.

These schemas are imported by both the SGT integration layer (sgt/src/...)
and the Maestro settlement engine (legacy_mvp/app/services/viability/).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LayerEntropyProfile:
    """Per-layer spectral entropy and outlier-geometry measurements."""

    layer_idx: int
    spectral_entropy: float
    effective_dimension: float
    viability_score: float
    fisher_curvature: float
    # Outlier geometry (TurboQuant-inspired massive-activation diagnostics)
    outlier_ratio: float = 1.0         # max dim magnitude / mean dim magnitude
    activation_kurtosis: float = 0.0   # excess kurtosis of per-dim magnitudes
    cardinal_proximity: float = 0.0    # mean max-abs component of unit vectors
    quantization_hostility: float = 0.0  # composite [0,1]; >0.7 = hostile

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "spectral_entropy": self.spectral_entropy,
            "effective_dimension": self.effective_dimension,
            "viability_score": self.viability_score,
            "fisher_curvature": self.fisher_curvature,
            "outlier_ratio": self.outlier_ratio,
            "activation_kurtosis": self.activation_kurtosis,
            "cardinal_proximity": self.cardinal_proximity,
            "quantization_hostility": self.quantization_hostility,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LayerEntropyProfile":
        return cls(
            layer_idx=int(d["layer_idx"]),
            spectral_entropy=float(d["spectral_entropy"]),
            effective_dimension=float(d["effective_dimension"]),
            viability_score=float(d["viability_score"]),
            fisher_curvature=float(d["fisher_curvature"]),
            outlier_ratio=float(d.get("outlier_ratio", 1.0)),
            activation_kurtosis=float(d.get("activation_kurtosis", 0.0)),
            cardinal_proximity=float(d.get("cardinal_proximity", 0.0)),
            quantization_hostility=float(d.get("quantization_hostility", 0.0)),
        )


@dataclass
class EntropySnapshot:
    """A single PRISM measurement of a model's internal geometric state.

    Captures the spectral structure of activations at selected layers,
    averaged over a set of eval prompts. Used as the thermodynamic proof
    artifact for Maestro settlement verification.
    """

    timestamp: float
    model_name: str
    generation_idx: int
    regime: str
    mean_spectral_entropy: float
    mean_effective_dimension: float
    mean_viability_score: float
    mean_fisher_curvature: float
    layer_profiles: List[LayerEntropyProfile]
    mean_phase_coherence: float
    noise_sensitivity: float
    n_samples: int
    # Outlier geometry aggregates
    mean_outlier_ratio: float = 1.0
    mean_cardinal_proximity: float = 0.0
    mean_quantization_hostility: float = 0.0
    worst_layer_idx: int = -1  # layer with highest quantization_hostility

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "generation_idx": self.generation_idx,
            "regime": self.regime,
            "mean_spectral_entropy": self.mean_spectral_entropy,
            "mean_effective_dimension": self.mean_effective_dimension,
            "mean_viability_score": self.mean_viability_score,
            "mean_fisher_curvature": self.mean_fisher_curvature,
            "layer_profiles": [lp.to_dict() for lp in self.layer_profiles],
            "mean_phase_coherence": self.mean_phase_coherence,
            "noise_sensitivity": self.noise_sensitivity,
            "n_samples": self.n_samples,
            "mean_outlier_ratio": self.mean_outlier_ratio,
            "mean_cardinal_proximity": self.mean_cardinal_proximity,
            "mean_quantization_hostility": self.mean_quantization_hostility,
            "worst_layer_idx": self.worst_layer_idx,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntropySnapshot":
        return cls(
            timestamp=float(d["timestamp"]),
            model_name=str(d["model_name"]),
            generation_idx=int(d["generation_idx"]),
            regime=str(d["regime"]),
            mean_spectral_entropy=float(d["mean_spectral_entropy"]),
            mean_effective_dimension=float(d["mean_effective_dimension"]),
            mean_viability_score=float(d["mean_viability_score"]),
            mean_fisher_curvature=float(d["mean_fisher_curvature"]),
            layer_profiles=[
                LayerEntropyProfile.from_dict(lp)
                for lp in d.get("layer_profiles", [])
            ],
            mean_phase_coherence=float(d["mean_phase_coherence"]),
            noise_sensitivity=float(d["noise_sensitivity"]),
            n_samples=int(d["n_samples"]),
            mean_outlier_ratio=float(d.get("mean_outlier_ratio", 1.0)),
            mean_cardinal_proximity=float(d.get("mean_cardinal_proximity", 0.0)),
            mean_quantization_hostility=float(d.get("mean_quantization_hostility", 0.0)),
            worst_layer_idx=int(d.get("worst_layer_idx", -1)),
        )


@dataclass
class EntropyDeltaProof:
    """Cryptographically comparable delta between two PRISM snapshots.

    reduction_verified is True iff delta_spectral_entropy < -epsilon_threshold,
    meaning the model's internal activation entropy decreased after the
    human contribution was incorporated (thermodynamic grounding).
    """

    snapshot_before: EntropySnapshot
    snapshot_after: EntropySnapshot
    delta_spectral_entropy: float
    delta_effective_dimension: float
    delta_viability_score: float
    delta_phase_coherence: float
    epsilon_threshold: float
    reduction_verified: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_before": self.snapshot_before.to_dict(),
            "snapshot_after": self.snapshot_after.to_dict(),
            "delta_spectral_entropy": self.delta_spectral_entropy,
            "delta_effective_dimension": self.delta_effective_dimension,
            "delta_viability_score": self.delta_viability_score,
            "delta_phase_coherence": self.delta_phase_coherence,
            "epsilon_threshold": self.epsilon_threshold,
            "reduction_verified": self.reduction_verified,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntropyDeltaProof":
        return cls(
            snapshot_before=EntropySnapshot.from_dict(d["snapshot_before"]),
            snapshot_after=EntropySnapshot.from_dict(d["snapshot_after"]),
            delta_spectral_entropy=float(d["delta_spectral_entropy"]),
            delta_effective_dimension=float(d["delta_effective_dimension"]),
            delta_viability_score=float(d["delta_viability_score"]),
            delta_phase_coherence=float(d["delta_phase_coherence"]),
            epsilon_threshold=float(d["epsilon_threshold"]),
            reduction_verified=bool(d["reduction_verified"]),
        )


@dataclass
class GeometricHealthScore:
    """Composite health score derived from spectral geometric analysis.

    composite_score is always clamped to [0, 1].
    Higher scores indicate healthier activation geometry.
    """

    viability_score: float
    spectral_entropy: float
    noise_sensitivity: float
    composite_score: float

    def __post_init__(self) -> None:
        self.composite_score = max(0.0, min(1.0, self.composite_score))
