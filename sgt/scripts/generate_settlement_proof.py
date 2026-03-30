"""Bridge script: SGT checkpoint pair → Maestro SettlementAttempt JSON.

Takes a model before and after incorporating human correction data,
measures the PRISM entropy snapshot at each state, computes ΔS, and
outputs a SettlementAttempt-compatible JSON that Maestro can use for
thermodynamic settlement verification.

Usage:
    python scripts/generate_settlement_proof.py \\
        --model Qwen/Qwen2.5-0.5B \\
        --checkpoint-before results/multi_task/R3/generation_2 \\
        --checkpoint-after  results/multi_task/R3/generation_3 \\
        --regime R3 \\
        --claimed-reduction 0.15 \\
        --payout 50.0 \\
        --output /tmp/settlement_proof.json

If --eval-data is omitted, the script generates synthetic eval prompts
via ColorBindingEnvironment (no dataset download required).
"""

import argparse
import json
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Resolve imports from the SGT src package
_SGT_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "../src"))
if _SGT_SRC not in sys.path:
    sys.path.insert(0, _SGT_SRC)

# Resolve imports from maestro libs
_MAESTRO_ADAPTERS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../libs/adapters")
)
_MAESTRO_SCHEMAS = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../libs/schemas")
)
for _p in [_MAESTRO_ADAPTERS, _MAESTRO_SCHEMAS]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def load_eval_data(eval_data_path: str, n_samples: int) -> list:
    """Load eval prompts from a JSON file or generate synthetic ones."""
    if eval_data_path and os.path.isfile(eval_data_path):
        with open(eval_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[:n_samples]
        raise ValueError(f"eval_data must be a JSON list, got {type(data)}")

    # Fallback: synthetic prompts from ColorBindingEnvironment
    logger.info("No --eval-data provided. Generating %d synthetic prompts.", n_samples)
    from semantic_grounding.datasets.synthetic_env import ColorBindingEnvironment
    env = ColorBindingEnvironment(seed=99)
    dataset = env.generate_eval_set()
    return dataset[:n_samples]


def load_trainer_at_checkpoint(model_name: str, checkpoint_path: str, dtype: str, config: dict):
    """Load a RecursiveTrainer and restore the LoRA adapter from checkpoint_path."""
    from semantic_grounding.training.pipeline import RecursiveTrainer

    trainer = RecursiveTrainer(
        model_name=model_name,
        config=config,
        dtype=dtype,
    )
    if checkpoint_path and os.path.isdir(checkpoint_path):
        logger.info("Loading checkpoint from %s", checkpoint_path)
        trainer._load_model(checkpoint_path)
    else:
        logger.warning("Checkpoint path %r not found — using base model weights.", checkpoint_path)
    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Maestro-compatible SettlementAttempt JSON from two SGT checkpoints."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace model name or local path.")
    parser.add_argument("--checkpoint-before", type=str, required=True,
                        help="Path to the SGT adapter checkpoint BEFORE the human contribution.")
    parser.add_argument("--checkpoint-after", type=str, required=True,
                        help="Path to the SGT adapter checkpoint AFTER incorporating the contribution.")
    parser.add_argument("--eval-data", type=str, default=None,
                        help="Path to a JSON eval dataset (list of dicts with 'prompt' key). "
                             "If omitted, uses synthetic ColorBindingEnvironment prompts.")
    parser.add_argument("--regime", type=str, default="",
                        help="SGT regime label (R1, R2, R3, R4, R1_accum).")
    parser.add_argument("--generation-before", type=int, default=0,
                        help="SGT generation index for the 'before' state.")
    parser.add_argument("--generation-after", type=int, default=1,
                        help="SGT generation index for the 'after' state.")
    parser.add_argument("--claimed-reduction", type=float, default=0.10,
                        help="Claimed entropy reduction (positive float). Default: 0.10.")
    parser.add_argument("--payout", type=float, default=50.0,
                        help="Settlement payout amount in USD. Default: 50.0.")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of eval prompts for PRISM snapshots. Default: 10.")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Minimum entropy reduction threshold (epsilon) for settlement. Default: 0.01.")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype. Default: float16.")
    parser.add_argument("--output", type=str, default="settlement_proof.json",
                        help="Output path for the SettlementAttempt JSON.")
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # 1. Load PRISM adapter
    # ---------------------------------------------------------------------------
    try:
        from prism_adapter import take_snapshot, compute_entropy_delta, PrismUnavailableError
        from prism_schemas import EntropySnapshot
    except ImportError as e:
        logger.error(
            "PRISM adapter not importable: %s\n"
            "Install PRISM: pip install -e %s",
            e,
            os.path.normpath(os.path.join(os.path.dirname(__file__), "../../prism"))
        )
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # 2. Load eval data
    # ---------------------------------------------------------------------------
    eval_data = load_eval_data(args.eval_data, args.n_samples)
    prompts = [item.get("prompt") or item.get("input") or item.get("question", "")
               for item in eval_data if item]
    prompts = [p for p in prompts if p][:args.n_samples]
    if not prompts:
        logger.error("Could not extract any prompts from eval data. Exiting.")
        sys.exit(1)
    logger.info("Using %d eval prompts for PRISM snapshots.", len(prompts))

    # ---------------------------------------------------------------------------
    # 3. Snapshot BEFORE checkpoint
    # ---------------------------------------------------------------------------
    model_config = {"base_model": args.model, "dtype": args.dtype}
    logger.info("=== Snapshot BEFORE (checkpoint: %s) ===", args.checkpoint_before)
    trainer_before = load_trainer_at_checkpoint(args.model, args.checkpoint_before,
                                                args.dtype, model_config)

    t0 = time.time()
    try:
        snapshot_before = take_snapshot(
            model=trainer_before.model,
            tokenizer=trainer_before.tokenizer,
            eval_prompts=prompts,
            generation_idx=args.generation_before,
            regime=args.regime,
            n_samples=len(prompts),
        )
    except PrismUnavailableError as e:
        logger.error("PRISM not available: %s", e)
        sys.exit(1)

    logger.info(
        "Snapshot BEFORE: entropy=%.4f, eff_dim=%.2f, viability=%.4f, coherence=%.4f (%.1fs)",
        snapshot_before.mean_spectral_entropy,
        snapshot_before.mean_effective_dimension,
        snapshot_before.mean_viability_score,
        snapshot_before.mean_phase_coherence,
        time.time() - t0,
    )

    # Free memory before loading next checkpoint
    del trainer_before

    # ---------------------------------------------------------------------------
    # 4. Snapshot AFTER checkpoint
    # ---------------------------------------------------------------------------
    logger.info("=== Snapshot AFTER (checkpoint: %s) ===", args.checkpoint_after)
    trainer_after = load_trainer_at_checkpoint(args.model, args.checkpoint_after,
                                               args.dtype, model_config)

    t0 = time.time()
    snapshot_after = take_snapshot(
        model=trainer_after.model,
        tokenizer=trainer_after.tokenizer,
        eval_prompts=prompts,
        generation_idx=args.generation_after,
        regime=args.regime,
        n_samples=len(prompts),
    )
    logger.info(
        "Snapshot AFTER:  entropy=%.4f, eff_dim=%.2f, viability=%.4f, coherence=%.4f (%.1fs)",
        snapshot_after.mean_spectral_entropy,
        snapshot_after.mean_effective_dimension,
        snapshot_after.mean_viability_score,
        snapshot_after.mean_phase_coherence,
        time.time() - t0,
    )
    del trainer_after

    # ---------------------------------------------------------------------------
    # 5. Compute ΔS proof
    # ---------------------------------------------------------------------------
    proof = compute_entropy_delta(snapshot_before, snapshot_after, epsilon=args.epsilon)

    print("\n" + "=" * 60)
    print("PRISM Settlement Proof Summary")
    print("=" * 60)
    print(f"  ΔS (spectral entropy):       {proof.delta_spectral_entropy:+.4f}")
    print(f"  Δ Effective dimension:        {proof.delta_effective_dimension:+.2f}")
    print(f"  Δ Viability score:            {proof.delta_viability_score:+.4f}")
    print(f"  Δ Phase coherence:            {proof.delta_phase_coherence:+.4f}")
    print(f"  Epsilon threshold:            {proof.epsilon_threshold:.4f}")
    print(f"  Reduction verified:           {'YES ✓' if proof.reduction_verified else 'NO ✗'}")
    print(f"  Claimed reduction:            {args.claimed_reduction:.4f}")
    if abs(args.claimed_reduction) > 1e-6:
        measured = -proof.delta_spectral_entropy
        disc = abs(measured - args.claimed_reduction) / (abs(args.claimed_reduction) + 1e-9)
        print(f"  Claimed vs measured:          {disc:.1%} discrepancy")
    print("=" * 60 + "\n")

    # ---------------------------------------------------------------------------
    # 6. Build SettlementAttempt JSON
    # ---------------------------------------------------------------------------
    settlement_attempt = {
        "prior_distribution": {
            "source": "sgt",
            "regime": args.regime,
            "generation": args.generation_before,
            "model": args.model,
        },
        "posterior_distribution": {
            "source": "sgt",
            "regime": args.regime,
            "generation": args.generation_after,
            "model": args.model,
        },
        "observation_count": len(prompts),
        "claimed_entropy_reduction": args.claimed_reduction,
        "total_payout": args.payout,
        "entropy_snapshot_before": snapshot_before.to_dict(),
        "entropy_snapshot_after": snapshot_after.to_dict(),
        "entropy_delta_proof": proof.to_dict(),
        # Metadata
        "_generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "_script": "generate_settlement_proof.py",
    }

    # ---------------------------------------------------------------------------
    # 7. Write output
    # ---------------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(settlement_attempt, f, indent=2)

    logger.info("SettlementAttempt JSON written to: %s", args.output)
    print(f"Output: {args.output}")
    print(f"reduction_verified: {proof.reduction_verified}")
    return 0 if proof.reduction_verified else 1


if __name__ == "__main__":
    sys.exit(main())
