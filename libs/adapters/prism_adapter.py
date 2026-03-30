"""PRISM adapter: spectral entropy analysis from HF model hidden states.

Provides the core computation functions consumed by the SGT integration
layer and the Maestro settlement engine.

Public API:
    take_snapshot(model, tokenizer, eval_prompts, ...) -> EntropySnapshot
    compute_entropy_delta(before, after, epsilon)     -> EntropyDeltaProof
    PrismUnavailableError

Computation overview:
    For each eval prompt, a forward pass with output_hidden_states=True is run.
    At each profiled layer, the activation matrix H (seq, dim) is analysed:

    - Spectral entropy:       SVD on H; Shannon entropy of normalised singular values.
    - Effective dimension:    Participation ratio (sum_sigma)^2 / sum_sigma^2.
    - Viability score:        Effective dimension / rank  (fraction of active directions).
    - Fisher curvature:       Mean squared activation value (gradient-free proxy).
    - Phase coherence:        Mean cosine similarity between adjacent-token representations.
    - Noise sensitivity:      1 - (min_sigma / max_sigma)  (condition number proxy).

    Outlier geometry (TurboQuant-inspired massive-activation diagnostics):
    - Outlier ratio:          max per-dim magnitude / mean per-dim magnitude.
                              >10 indicates a dominant "massive activation" dimension.
    - Activation kurtosis:    Excess kurtosis of per-dim magnitude distribution.
                              Positive = heavy-tailed; outlier dims pull the tail up.
    - Cardinal proximity:     Mean max-abs component of each token's unit vector.
                              Near 1.0 = vectors nearly aligned to basis axes, meaning
                              quantization would snap them to cardinal directions.
    - Quantization hostility: Composite [0, 1] from the three metrics above.
                              >0.7 signals this layer is unfriendly to low-bit quant.

    All metrics are averaged across prompts and then across profiled layers.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Resolve prism_schemas from the sibling schemas directory
_SCHEMAS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../schemas"))
if os.path.isdir(_SCHEMAS_DIR) and _SCHEMAS_DIR not in sys.path:
    sys.path.insert(0, _SCHEMAS_DIR)

from prism_schemas import (  # noqa: E402
    EntropySnapshot,
    EntropyDeltaProof,
    GeometricHealthScore,
    LayerEntropyProfile,
)


class PrismUnavailableError(RuntimeError):
    """Raised when PRISM cannot perform a snapshot (e.g. torch not installed)."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_layers(n_layers: int) -> List[int]:
    """Select up to 8 evenly-spaced layers to profile."""
    candidates = list(range(0, n_layers, 3))
    return candidates[:8]


def _outlier_geometry(H_raw: "torch.Tensor") -> Tuple[float, float, float, float]:
    """Compute TurboQuant-inspired massive-activation geometry metrics.

    Args:
        H_raw: Raw (not mean-centred) float tensor of shape (seq_len, hidden_dim).

    Returns:
        (outlier_ratio, activation_kurtosis, cardinal_proximity,
         quantization_hostility)
    """
    import torch
    import torch.nn.functional as F

    seq, dim = H_raw.shape

    # Per-dimension mean absolute magnitude across tokens
    dim_mag = H_raw.abs().mean(dim=0)  # (dim,)

    # 1. Outlier ratio: how dominant is the largest dimension?
    mean_mag = dim_mag.mean()
    max_mag = dim_mag.max()
    outlier_ratio = float((max_mag / (mean_mag + 1e-12)).item())

    # 2. Activation kurtosis: excess kurtosis of per-dim magnitudes
    #    Positive = heavy-tailed distribution (outlier dims pull the tail).
    mu = dim_mag.mean()
    sigma = dim_mag.std(unbiased=False)
    if sigma < 1e-12:
        activation_kurtosis = 0.0
    else:
        activation_kurtosis = float(
            (((dim_mag - mu) ** 4).mean() / (sigma ** 4 + 1e-12) - 3.0).item()
        )

    # 3. Cardinal proximity: mean max-abs component of each token's unit vector.
    #    Uses raw H so directions are not distorted by mean-centring.
    h_unit = F.normalize(H_raw, dim=-1)  # (seq, dim)
    cardinal_proximity = float(h_unit.abs().max(dim=-1).values.mean().item())

    # 4. Composite quantization hostility score in [0, 1].
    #    Each sub-metric is normalised independently before averaging:
    #      outlier_ratio:        log-scale, ceiling at 50x  → [0, 1]
    #      activation_kurtosis:  clamp excess kurtosis to [0, 20] → [0, 1]
    #      cardinal_proximity:   already in (~0, 1)              → [0, 1]
    or_norm = min(math.log(max(outlier_ratio, 1.0)) / math.log(50.0), 1.0)
    ak_norm = min(max(activation_kurtosis, 0.0) / 20.0, 1.0)
    cp_norm = float(cardinal_proximity)
    quantization_hostility = (or_norm + ak_norm + cp_norm) / 3.0

    return outlier_ratio, activation_kurtosis, cardinal_proximity, quantization_hostility


def _compute_layer_metrics(hidden: "torch.Tensor") -> Dict[str, float]:
    """Compute spectral and outlier-geometry metrics from a layer's hidden states.

    Args:
        hidden: Float tensor of shape (seq_len, hidden_dim).

    Returns:
        Dict with keys: spectral_entropy, effective_dimension, viability_score,
        fisher_curvature, phase_coherence, noise_sensitivity,
        outlier_ratio, activation_kurtosis, cardinal_proximity,
        quantization_hostility.
    """
    import torch
    import torch.nn.functional as F

    H = hidden.float()
    seq, dim = H.shape

    _degenerate = {
        "spectral_entropy": 0.0,
        "effective_dimension": 1.0,
        "viability_score": 0.0,
        "fisher_curvature": 0.0,
        "phase_coherence": 1.0,
        "noise_sensitivity": 1.0,
        "outlier_ratio": 1.0,
        "activation_kurtosis": 0.0,
        "cardinal_proximity": 0.0,
        "quantization_hostility": 0.0,
    }
    if seq < 2 or dim < 2:
        return _degenerate

    # Outlier geometry on raw activations (before mean-centring distorts directions)
    outlier_ratio, activation_kurtosis, cardinal_proximity, quantization_hostility = (
        _outlier_geometry(H)
    )

    # Mean-centre activations before SVD
    H = H - H.mean(dim=0, keepdim=True)

    try:
        S = torch.linalg.svdvals(H)
    except Exception:
        return _degenerate

    S = S.clamp(min=0.0)
    rank = len(S)

    # 1. Spectral entropy
    s_sum = S.sum()
    if s_sum < 1e-12:
        spectral_entropy = 0.0
    else:
        p = S / s_sum
        spectral_entropy = float(-(p * (p + 1e-12).log()).sum().item())

    # 2. Effective dimension (participation ratio)
    s2_sum = (S ** 2).sum()
    effective_dimension = float((s_sum ** 2 / (s2_sum + 1e-12)).item())

    # 3. Viability score: normalised effective dimension
    viability_score = effective_dimension / rank if rank > 0 else 0.0

    # 4. Fisher curvature approximation: mean squared activation
    fisher_curvature = float((H ** 2).mean().item())

    # 5. Phase coherence: mean cosine similarity of adjacent token vectors
    h_norm = F.normalize(H, dim=-1)  # (seq, dim)
    cos_sims = (h_norm[:-1] * h_norm[1:]).sum(dim=-1)  # (seq-1,)
    phase_coherence = float(cos_sims.mean().item())

    # 6. Noise sensitivity: 1 - ratio of smallest to largest singular value
    s_max = float(S[0].item()) if len(S) > 0 else 1.0
    nonzero = S[S > 1e-6]
    s_min = float(nonzero.min().item()) if len(nonzero) > 0 else 0.0
    noise_sensitivity = 1.0 - (s_min / (s_max + 1e-12))

    return {
        "spectral_entropy": spectral_entropy,
        "effective_dimension": effective_dimension,
        "viability_score": viability_score,
        "fisher_curvature": fisher_curvature,
        "phase_coherence": phase_coherence,
        "noise_sensitivity": noise_sensitivity,
        "outlier_ratio": outlier_ratio,
        "activation_kurtosis": activation_kurtosis,
        "cardinal_proximity": cardinal_proximity,
        "quantization_hostility": quantization_hostility,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def take_snapshot(
    model: Any,
    tokenizer: Any,
    eval_prompts: List[str],
    generation_idx: int,
    regime: str,
    layers_to_profile: Optional[List[int]] = None,
    n_samples: int = 10,
    device: Optional[str] = None,
) -> EntropySnapshot:
    """Compute an EntropySnapshot from a HF model over eval prompts.

    Args:
        model:            HuggingFace model with output_hidden_states support.
        tokenizer:        HuggingFace tokenizer.
        eval_prompts:     List of plain-text prompts to profile.
        generation_idx:   SGT generation index (for snapshot labelling).
        regime:           SGT regime label (e.g. "R1", "R2").
        layers_to_profile: Layer indices to profile. None = every 3rd, max 8.
        n_samples:        Maximum number of prompts to use.
        device:           Override device string (e.g. "cuda:0"). If None,
                          inferred from model parameters.

    Returns:
        EntropySnapshot with metrics averaged across prompts and layers.

    Raises:
        PrismUnavailableError: If torch is not installed or the model cannot
                               produce hidden states.
        ValueError:            If eval_prompts is empty.
    """
    try:
        import torch
    except ImportError:
        raise PrismUnavailableError(
            "torch is not installed. Install it with: pip install torch"
        )

    if not hasattr(model, "forward"):
        raise PrismUnavailableError("model does not expose a forward() method.")

    if not eval_prompts:
        raise ValueError("eval_prompts must not be empty.")

    prompts = eval_prompts[:n_samples]

    # Resolve device
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    # Best-effort model name
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "name_or_path"):
        model_name = str(cfg.name_or_path)
    elif cfg is not None and hasattr(cfg, "_name_or_path"):
        model_name = str(cfg._name_or_path)
    else:
        model_name = type(model).__name__

    # Collect hidden states: {layer_idx -> [metric_dict, ...]}
    all_layer_metrics: Dict[int, List[Dict[str, float]]] = {}

    for prompt in prompts:
        try:
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=False,
            )
            enc = {k: v.to(model_device) for k, v in enc.items()}

            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)

            hidden_states = out.hidden_states  # tuple of (1, seq, dim)
            n_layers = len(hidden_states)

            selected = (
                _default_layers(n_layers)
                if layers_to_profile is None
                else [l for l in layers_to_profile if 0 <= l < n_layers]
            )

            for layer_idx in selected:
                h = hidden_states[layer_idx][0]  # (seq, dim)
                metrics = _compute_layer_metrics(h)
                all_layer_metrics.setdefault(layer_idx, []).append(metrics)

        except Exception as exc:
            logger.debug("prism_adapter: skipped prompt due to error: %s", exc)
            continue

    if not all_layer_metrics:
        raise PrismUnavailableError(
            "No hidden states could be computed. "
            "Ensure output_hidden_states=True is supported by the model."
        )

    # Average per-layer metrics across prompts
    layer_profiles: List[LayerEntropyProfile] = []
    all_phase_coherences: List[float] = []
    all_noise_sensitivities: List[float] = []

    for layer_idx in sorted(all_layer_metrics.keys()):
        pm = all_layer_metrics[layer_idx]

        def _avg(key: str) -> float:
            return sum(m[key] for m in pm) / len(pm)

        layer_profiles.append(LayerEntropyProfile(
            layer_idx=layer_idx,
            spectral_entropy=_avg("spectral_entropy"),
            effective_dimension=_avg("effective_dimension"),
            viability_score=_avg("viability_score"),
            fisher_curvature=_avg("fisher_curvature"),
            outlier_ratio=_avg("outlier_ratio"),
            activation_kurtosis=_avg("activation_kurtosis"),
            cardinal_proximity=_avg("cardinal_proximity"),
            quantization_hostility=_avg("quantization_hostility"),
        ))
        all_phase_coherences.extend(m["phase_coherence"] for m in pm)
        all_noise_sensitivities.extend(m["noise_sensitivity"] for m in pm)

    def _mean_profile_field(field: str) -> float:
        vals = [getattr(lp, field) for lp in layer_profiles]
        return sum(vals) / len(vals) if vals else 0.0

    mean_phase_coherence = (
        sum(all_phase_coherences) / len(all_phase_coherences)
        if all_phase_coherences else 0.0
    )
    noise_sensitivity = (
        sum(all_noise_sensitivities) / len(all_noise_sensitivities)
        if all_noise_sensitivities else 0.0
    )

    # Identify worst layer by quantization_hostility for targeted diagnostics
    worst_layer_idx = (
        max(layer_profiles, key=lambda lp: lp.quantization_hostility).layer_idx
        if layer_profiles else -1
    )

    return EntropySnapshot(
        timestamp=time.time(),
        model_name=model_name,
        generation_idx=generation_idx,
        regime=regime,
        mean_spectral_entropy=_mean_profile_field("spectral_entropy"),
        mean_effective_dimension=_mean_profile_field("effective_dimension"),
        mean_viability_score=_mean_profile_field("viability_score"),
        mean_fisher_curvature=_mean_profile_field("fisher_curvature"),
        layer_profiles=layer_profiles,
        mean_phase_coherence=mean_phase_coherence,
        noise_sensitivity=noise_sensitivity,
        n_samples=len(prompts),
        mean_outlier_ratio=_mean_profile_field("outlier_ratio"),
        mean_cardinal_proximity=_mean_profile_field("cardinal_proximity"),
        mean_quantization_hostility=_mean_profile_field("quantization_hostility"),
        worst_layer_idx=worst_layer_idx,
    )


def compute_entropy_delta(
    before: EntropySnapshot,
    after: EntropySnapshot,
    epsilon: float = 0.01,
) -> EntropyDeltaProof:
    """Compute the delta proof between two EntropySnapshots.

    reduction_verified is True iff:
        delta_spectral_entropy < -epsilon
    meaning entropy decreased by more than epsilon (grounding was effective).

    Args:
        before:  Snapshot taken before the human contribution.
        after:   Snapshot taken after the human contribution.
        epsilon: Minimum entropy reduction required for verification.

    Returns:
        EntropyDeltaProof with deltas and verification flag.
    """
    delta_se = after.mean_spectral_entropy - before.mean_spectral_entropy
    delta_ed = after.mean_effective_dimension - before.mean_effective_dimension
    delta_vs = after.mean_viability_score - before.mean_viability_score
    delta_pc = after.mean_phase_coherence - before.mean_phase_coherence

    return EntropyDeltaProof(
        snapshot_before=before,
        snapshot_after=after,
        delta_spectral_entropy=delta_se,
        delta_effective_dimension=delta_ed,
        delta_viability_score=delta_vs,
        delta_phase_coherence=delta_pc,
        epsilon_threshold=epsilon,
        reduction_verified=delta_se < -epsilon,
    )
