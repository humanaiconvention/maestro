"""Non-invasive PRISM measurement hook for RecursiveTrainer.

RecursiveTrainerPrismHook attaches to the SGT training loop without
modifying pipeline.py. Call on_generation_end() after each training
generation completes, then use get_trajectory_report() to retrieve
the full per-generation PRISM trajectory.

Usage in a benchmark script:
    hook = RecursiveTrainerPrismHook(trainer, eval_data, regime="R1")

    for gen_idx in range(n_generations):
        trainer.train_generation(...)
        trainer._load_model(...)           # or however model is refreshed
        snapshot = hook.on_generation_end(gen_idx)
        print(f"Gen {gen_idx}: spectral_entropy={snapshot.mean_spectral_entropy:.3f}")

    report = hook.get_trajectory_report()
    # report["trajectory"] → list of per-gen metrics dicts
    # report["snapshots"]  → list of EntropySnapshot.to_dict()
    # report["deltas"]     → list of EntropyDeltaProof.to_dict()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .snapshot import take_sgt_snapshot, compute_entropy_delta, PrismUnavailableError
from prism_schemas import EntropySnapshot, EntropyDeltaProof

logger = logging.getLogger(__name__)


class RecursiveTrainerPrismHook:
    """Observer that records PRISM snapshots across SGT training generations.

    This class does not modify RecursiveTrainer. It is instantiated alongside
    the trainer and called explicitly at the end of each generation.

    Attributes:
        snapshots: List of EntropySnapshot objects, one per generation.
        deltas:    List of EntropyDeltaProof objects, one per consecutive pair.
        errors:    List of (gen_idx, error_message) for any failed snapshots.
    """

    def __init__(
        self,
        trainer: Any,
        eval_data: List[Dict[str, Any]],
        regime: str,
        n_samples: int = 10,
        layers_to_profile: Optional[List[int]] = None,
        epsilon: float = 0.01,
    ) -> None:
        """
        Args:
            trainer: RecursiveTrainer instance (must have .model, .tokenizer, .device).
            eval_data: SGT eval dataset — list of dicts with "prompt" key.
            regime: SGT regime label ("R1", "R2", "R3", "R4", "R1_accum").
            n_samples: Prompts to use per snapshot (trade-off: accuracy vs speed).
            layers_to_profile: Layer indices to measure. None = PRISM default (every 3rd, max 8).
            epsilon: Threshold for EntropyDeltaProof.reduction_verified check.
        """
        self.trainer = trainer
        self.eval_data = eval_data
        self.regime = regime
        self.n_samples = n_samples
        self.layers_to_profile = layers_to_profile
        self.epsilon = epsilon

        self.snapshots: List[EntropySnapshot] = []
        self.deltas: List[EntropyDeltaProof] = []
        self.errors: List[tuple] = []

    def on_generation_end(self, gen_idx: int) -> Optional[EntropySnapshot]:
        """Capture a PRISM snapshot after a training generation completes.

        Call this after trainer.train_generation() and after the model has been
        reloaded / refreshed (so the snapshot reflects the post-training state).

        Args:
            gen_idx: The generation index that just completed (0-indexed).

        Returns:
            EntropySnapshot if successful, None if PRISM is unavailable or failed.
            Errors are logged and appended to self.errors — the training loop
            continues regardless.
        """
        try:
            snapshot = take_sgt_snapshot(
                trainer=self.trainer,
                eval_data=self.eval_data,
                generation_idx=gen_idx,
                regime=self.regime,
                n_samples=self.n_samples,
                layers_to_profile=self.layers_to_profile,
            )
            self.snapshots.append(snapshot)

            # Compute delta if we have at least two snapshots
            if len(self.snapshots) >= 2:
                delta = compute_entropy_delta(
                    self.snapshots[-2],
                    self.snapshots[-1],
                    epsilon=self.epsilon,
                )
                self.deltas.append(delta)
                logger.info(
                    "prism_hook gen=%d: ΔS=%.4f eff_dim_delta=%.2f viability_delta=%.4f verified=%s",
                    gen_idx,
                    delta.delta_spectral_entropy,
                    delta.delta_effective_dimension,
                    delta.delta_viability_score,
                    delta.reduction_verified,
                )

            logger.info(
                "prism_hook gen=%d: entropy=%.4f eff_dim=%.2f viability=%.4f coherence=%.4f",
                gen_idx,
                snapshot.mean_spectral_entropy,
                snapshot.mean_effective_dimension,
                snapshot.mean_viability_score,
                snapshot.mean_phase_coherence,
            )
            return snapshot

        except PrismUnavailableError:
            msg = "PRISM not installed — snapshot skipped."
            logger.warning("prism_hook gen=%d: %s", gen_idx, msg)
            self.errors.append((gen_idx, msg))
            return None
        except Exception as exc:
            msg = str(exc)
            logger.error("prism_hook gen=%d: snapshot failed: %s", gen_idx, msg, exc_info=True)
            self.errors.append((gen_idx, msg))
            return None

    def get_trajectory_report(self) -> Dict[str, Any]:
        """Return a serializable trajectory report for inclusion in benchmark output.

        The "trajectory" list is the most useful for report generation — it provides
        a flat per-generation dict that can be directly added to the metrics CSV
        or rendered as a Markdown table in make_report.py.

        Returns:
            Dict with keys:
                "trajectory": List[Dict] — flat per-gen metrics (regime, spectral_entropy, etc.)
                "snapshots":  List[Dict] — full EntropySnapshot serializations
                "deltas":     List[Dict] — full EntropyDeltaProof serializations
                "errors":     List of (gen_idx, message) tuples
        """
        trajectory = [
            {
                "generation": s.generation_idx,
                "regime": s.regime,
                "spectral_entropy": s.mean_spectral_entropy,
                "effective_dimension": s.mean_effective_dimension,
                "viability_score": s.mean_viability_score,
                "phase_coherence": s.mean_phase_coherence,
                "noise_sensitivity": s.noise_sensitivity,
                "n_samples": s.n_samples,
            }
            for s in self.snapshots
        ]

        return {
            "regime": self.regime,
            "n_generations": len(self.snapshots),
            "trajectory": trajectory,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "deltas": [d.to_dict() for d in self.deltas],
            "errors": self.errors,
        }

    def as_metrics_rows(self) -> List[Dict[str, Any]]:
        """Return trajectory as a list of per-gen metric dicts suitable for EarlyWarningDetector.

        The returned dicts contain "spectral_entropy" and "effective_dimension" keys
        that EarlyWarningDetector._analyze_geometric_trajectory() uses.
        Merge these with the standard accuracy/perplexity metrics dicts:

            gen_metrics = standard_metrics_dict.copy()
            gen_metrics.update(prism_hook.as_metrics_rows()[gen_idx])
        """
        return [
            {
                "generation": s.generation_idx,
                "spectral_entropy": s.mean_spectral_entropy,
                "effective_dimension": s.mean_effective_dimension,
                "viability_score": s.mean_viability_score,
                "phase_coherence": s.mean_phase_coherence,
            }
            for s in self.snapshots
        ]
