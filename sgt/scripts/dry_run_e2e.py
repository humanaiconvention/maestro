"""End-to-end dry-run: exercises the real ML pipeline with 1 generation, 10 samples.

Validates that model loading → LoRA init → training → generation → evaluation →
report generation all work together on real hardware (float16 or bfloat16).

Usage (GPU, float16):
    python scripts/dry_run_e2e.py

Usage (L4 with bfloat16):
    python scripts/dry_run_e2e.py --dtype bfloat16

Usage (CPU-only, slow but works anywhere):
    python scripts/dry_run_e2e.py --dtype float32

The script runs R1 (pure synthetic) for 1 generation with 10 training samples
and produces a mini report in results/dry_run/.  Total time: ~2-5 min on GPU.
"""

import argparse
import json
import os
import sys
import time
import traceback

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "dry_run")


def main():
    parser = argparse.ArgumentParser(description="End-to-end dry-run of the ML pipeline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace model name (default: Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Torch dtype: float16 (GPU), bfloat16 (newer GPU), float32 (CPU)")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of training samples per generation (default: 10)")
    parser.add_argument("--generations", type=int, default=1,
                        help="Number of recursive generations (default: 1)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "dry_run.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        # Windows cp1252 console chokes on Unicode; replace non-ASCII with ASCII fallbacks
        safe_line = line.encode("ascii", errors="replace").decode("ascii")
        print(safe_line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Clear previous log
    with open(log_path, "w") as f:
        f.write(f"# Dry-run started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Model: {args.model}, dtype: {args.dtype}, samples: {args.samples}\n\n")

    steps = [
        "Import modules",
        "Create synthetic environment",
        "Initialize RecursiveTrainer (model + LoRA)",
        "Train generation 0",
        "Generate synthetic completions",
        "Evaluate (perplexity)",
        "Mix dataset for next generation",
        "Generate report",
    ]

    results = {}
    t_start = time.time()

    try:
        # ── Step 1: Imports ──────────────────────────────────────────────
        log("Step 1/8: Importing modules...")
        from semantic_grounding.training.pipeline import RecursiveTrainer
        from semantic_grounding.datasets.mixer import RegimeMixer
        from semantic_grounding.datasets.synthetic_env import ColorBindingEnvironment
        from semantic_grounding.reporting.drift_metrics import EarlyWarningDetector
        log("  ✓ All imports successful")

        # ── Step 2: Synthetic environment ────────────────────────────────
        log("Step 2/8: Creating synthetic environment...")
        env = ColorBindingEnvironment(seed=42)
        train_data = env.generate_human_dataset(args.samples)
        eval_data = env.generate_eval_set()
        correction_pool = env.generate_human_dataset(args.samples * 2)
        log(f"  ✓ Generated {len(train_data)} train, {len(eval_data)} eval, {len(correction_pool)} correction samples")
        log(f"  Ground truth: {env.ground_truth}")

        # ── Step 3: Initialize trainer ───────────────────────────────────
        log(f"Step 3/8: Loading model {args.model} (dtype={args.dtype})...")
        t_load = time.time()
        trainer_config = {
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 32,
            "batch_size": 2,
            "learning_rate": 2e-4,
            "epochs_per_generation": 1,
        }
        trainer = RecursiveTrainer(
            model_name=args.model,
            config=trainer_config,
            dtype=args.dtype,
        )
        log(f"  ✓ Model loaded in {time.time() - t_load:.1f}s on {trainer.device}")

        all_gen_metrics = []

        for gen in range(args.generations + 1):
            gen_dir = os.path.join(OUTPUT_DIR, f"gen_{gen}")
            os.makedirs(gen_dir, exist_ok=True)

            # ── Step 4: Train ────────────────────────────────────────────
            log(f"Step 4/8: Training generation {gen} ({len(train_data)} samples)...")
            t_train = time.time()
            trainer.train_generation(train_data, gen_dir)
            train_time = time.time() - t_train
            log(f"  ✓ Training complete in {train_time:.1f}s")

            # Reload trained adapter for evaluation
            trainer._load_model(gen_dir)

            # ── Step 5: Generate ─────────────────────────────────────────
            log(f"Step 5/8: Generating {len(eval_data)} synthetic completions...")
            t_gen = time.time()
            prompts = [d["prompt"] for d in eval_data]
            predictions = trainer.generate_synthetic_data(prompts, max_new_tokens=10)
            gen_time = time.time() - t_gen
            log(f"  ✓ Generation complete in {gen_time:.1f}s")

            # Check accuracy against ground truth
            correct = 0
            for pred, expected in zip(predictions, eval_data):
                if expected["expected"].lower() in pred["completion"].lower():
                    correct += 1
            accuracy = correct / len(eval_data) if eval_data else 0.0
            log(f"  Semantic binding accuracy: {accuracy:.0%} ({correct}/{len(eval_data)})")

            # ── Step 6: Evaluate perplexity ──────────────────────────────
            log(f"Step 6/8: Calculating perplexity...")
            t_eval = time.time()
            ppl = trainer.calculate_perplexity(eval_data)
            eval_time = time.time() - t_eval
            log(f"  ✓ Perplexity = {ppl:.2f} (computed in {eval_time:.1f}s)")

            gen_metrics = {
                "generation": gen,
                "accuracy": accuracy,
                "perplexity": ppl,
                "train_time_s": train_time,
                "gen_time_s": gen_time,
                "eval_time_s": eval_time,
                "n_samples": len(train_data),
            }
            all_gen_metrics.append(gen_metrics)

            # ── Step 7: Mix for next generation ──────────────────────────
            if gen < args.generations:
                log(f"Step 7/8: Mixing dataset for generation {gen + 1}...")
                mixer = RegimeMixer({"data_policy": "replace"})
                # R1: pure synthetic (correction_fraction=0.0)
                synthetic_gen = trainer.generate_synthetic_data(
                    [d["prompt"] for d in train_data], max_new_tokens=10
                )
                train_data = mixer.mix_dataset(
                    synthetic_data=synthetic_gen,
                    correction_pool=correction_pool,
                    target_size=args.samples,
                    correction_fraction=0.0,
                )
                log(f"  ✓ Mixed {len(train_data)} samples (R1 = 100% synthetic)")

        # ── Step 8: Generate report ──────────────────────────────────
        log("Step 8/8: Generating dry-run report...")
        metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_gen_metrics, f, indent=2)

        # Run EarlyWarningDetector
        detector = EarlyWarningDetector(accuracy_threshold=0.10, perplexity_threshold=0.05)
        analysis = detector.analyze_trajectory(all_gen_metrics)

        report_lines = [
            "# Dry-Run Report",
            "",
            f"**Model**: {args.model}",
            f"**dtype**: {args.dtype}",
            f"**Device**: {trainer.device}",
            f"**Samples**: {args.samples}",
            f"**Generations**: {args.generations}",
            f"**Total time**: {time.time() - t_start:.1f}s",
            "",
            "## Per-Generation Metrics",
            "",
            "| Gen | Accuracy | Perplexity | Train (s) | Gen (s) | Eval (s) |",
            "|-----|----------|------------|-----------|---------|----------|",
        ]
        for m in all_gen_metrics:
            report_lines.append(
                f"| {m['generation']} | {m['accuracy']:.0%} | {m['perplexity']:.2f} "
                f"| {m['train_time_s']:.1f} | {m['gen_time_s']:.1f} | {m['eval_time_s']:.1f} |"
            )
        report_lines.extend([
            "",
            "## Early Warning Analysis",
            "",
            f"- **Signature detected**: {analysis.get('signature_detected', 'N/A')}",
            f"- **Failure ordering**: {analysis.get('failure_ordering', 'N/A')}",
            f"- **Accuracy failure gen**: {analysis.get('accuracy_failure_generation', 'N/A')}",
            f"- **Perplexity failure gen**: {analysis.get('perplexity_failure_generation', 'N/A')}",
            "",
            "## Verdict",
            "",
        ])

        if len(all_gen_metrics) < 2:
            report_lines.append("Single generation — no drift measurable. Pipeline validated ✓")
        elif analysis.get("signature_detected"):
            report_lines.append("⚠️ Silent semantic drift detected within dry-run range.")
        else:
            report_lines.append("✓ No semantic drift in dry-run range (expected for 1 generation).")

        report_path = os.path.join(OUTPUT_DIR, "dry_run_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")

        log(f"  ✓ Report written to {report_path}")

        # ── Summary ──────────────────────────────────────────────────
        total_time = time.time() - t_start
        log("")
        log("=" * 60)
        log("DRY-RUN COMPLETE")
        log(f"  Total time: {total_time:.1f}s")
        log(f"  Final accuracy: {all_gen_metrics[-1]['accuracy']:.0%}")
        log(f"  Final perplexity: {all_gen_metrics[-1]['perplexity']:.2f}")
        log(f"  Outputs: {OUTPUT_DIR}/")
        log("=" * 60)

        results["success"] = True
        results["total_time"] = total_time

    except Exception as e:
        log(f"\n❌ FAILED at step: {e}")
        traceback.print_exc()
        with open(log_path, "a") as f:
            traceback.print_exc(file=f)
        results["success"] = False
        results["error"] = str(e)

    # Write machine-readable result
    with open(os.path.join(OUTPUT_DIR, "dry_run_result.json"), "w") as f:
        json.dump(results, f, indent=2)

    sys.exit(0 if results.get("success") else 1)


if __name__ == "__main__":
    main()
