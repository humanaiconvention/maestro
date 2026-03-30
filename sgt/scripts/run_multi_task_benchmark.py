"""Extended script to run the multi-task recursive regime benchmark."""

import argparse
import yaml
import os
import json
import torch
from datasets import disable_progress_bar

from semantic_grounding.training.pipeline import RecursiveTrainer
from semantic_grounding.datasets.synthetic_env import ColorBindingEnvironment
from semantic_grounding.datasets.mixer import RegimeMixer
from semantic_grounding.datasets.natural.loaders import NaturalDatasetLoader
from semantic_grounding.reporting.calibration import CalibrationMetrics, DiversityMetrics
from scripts.path_utils import project_results_path

disable_progress_bar()

def load_task_families(config):
    """Loads train and multi-eval datasets."""
    print("Loading Task Families...")
    eval_samples = config["datasets"].get("eval_samples", 50)
    train_samples = config["recursion"]["synthetic_samples_per_generation"]
    
    # 1. Train Family (Human Data G0)
    train_family = config["datasets"].get("train_family", "synthetic")
    
    if train_family == "synthetic":
        env = ColorBindingEnvironment(seed=42)
        human_train = env.generate_human_dataset(train_samples)
        eval_syn = env.generate_eval_set()
        eval_arc = NaturalDatasetLoader.load_arc_easy(num_train=0, num_eval=eval_samples)[1]
    elif train_family == "arc":
        human_train, eval_arc = NaturalDatasetLoader.load_arc_easy(num_train=train_samples, num_eval=eval_samples)
        env = ColorBindingEnvironment(seed=42)
        eval_syn = env.generate_eval_set()
    else:
        raise ValueError(f"Unknown train_family: {train_family}")

    # 2. OOD / Robustness Family
    eval_hellaswag = NaturalDatasetLoader.load_hellaswag(num_eval=eval_samples)
    
    # 3. Fluency / Calibration Family
    eval_wikitext = NaturalDatasetLoader.load_wikitext(num_samples=eval_samples)
    
    return {
        "train": human_train,
        "eval_families": {
            "synthetic_binding": eval_syn,
            "grounded_arc": eval_arc,
            "ood_hellaswag": eval_hellaswag,
            "fluency_wiki": eval_wikitext
        }
    }

def evaluate_all_families(trainer, eval_families) -> dict:
    """Evaluates all families to capture the 'Early Warning Signature'."""
    print("Evaluating Task Families...")
    metrics = {}
    
    for family_name, eval_set in eval_families.items():
        if not eval_set:
            continue
            
        # 1. Fluency Family just needs Perplexity
        if "fluency" in family_name:
            ppl = trainer.calculate_perplexity(eval_set)
            metrics[f"{family_name}_perplexity"] = ppl
            print(f"  {family_name} Perplexity: {ppl:.2f}")
            continue
            
        # 2. Grounded / OOD Families need Accuracy, Perplexity, Calibration, and Diversity
        prompts = [d["prompt"] for d in eval_set]
        predictions = trainer.generate_synthetic_data(prompts, max_new_tokens=10)
        completions = [p["completion"] for p in predictions]
        
        correct = 0
        labels = []
        for pred, expected in zip(predictions, eval_set):
            is_correct = pred["completion"].strip().lower().startswith(expected["expected"].strip().lower())
            if is_correct:
                correct += 1
            labels.append(1 if is_correct else 0)
                
        acc = correct / len(eval_set)
        ppl = trainer.calculate_perplexity(eval_set)
        
        # Calibration (requires logits)
        max_probs, _ = trainer.get_logits_for_eval(eval_set)
        ece = CalibrationMetrics.calculate_ece(max_probs, labels)
        
        # Diversity
        dist2 = DiversityMetrics.calculate_ngram_diversity(completions, n=2)
        
        metrics[f"{family_name}_accuracy"] = acc
        metrics[f"{family_name}_perplexity"] = ppl
        metrics[f"{family_name}_ece"] = ece
        metrics[f"{family_name}_dist2"] = dist2
        
        print(f"  {family_name} -> Acc: {acc:.2%}, PPL: {ppl:.2f}, ECE: {ece:.4f}, Dist2: {dist2:.4f}")
        
    return metrics

def run_regime(config_path: str, regime: str, prism: bool = False, prism_samples: int = 10):
    """Programmatic entry point to run a specific recursive regime."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    regime_config = config["regimes"].get(regime)
    if not regime_config:
        raise ValueError(f"Regime {regime} not found in {config_path}.")

    print(f"=== Starting Multi-Task Benchmark for Regime: {regime_config['name']} ===")

    data = load_task_families(config)
    frozen_human_data = data["train"]
    eval_families = data["eval_families"]
    
    mixer = RegimeMixer(regime_config)
    trainer = RecursiveTrainer(
        model_name=config["model"]["base_model"],
        config=config["model"],
        dtype=config["model"].get("dtype", "float16")
    )

    results = []
    current_adapter_path = None

    # Optionally initialise PRISM hook
    prism_hook = None
    if prism:
        try:
            import sys, os as _os
            _pi_dir = _os.path.join(_os.path.dirname(__file__),
                                    "../src/semantic_grounding/prism_integration")
            if _os.path.isdir(_pi_dir) and _pi_dir not in sys.path:
                sys.path.insert(0, _pi_dir)
            from semantic_grounding.prism_integration.hooks import RecursiveTrainerPrismHook
            # Use the grounded_arc eval family as the PRISM eval set (small, stable)
            prism_eval_data = data["eval_families"].get("grounded_arc") or data["train"]
            prism_hook = RecursiveTrainerPrismHook(
                trainer=trainer,
                eval_data=prism_eval_data,
                regime=regime,
                n_samples=prism_samples,
            )
            print(f"[PRISM] Hook initialised. Snapshots at each generation end (n_samples={prism_samples}).")
        except Exception as _e:
            print(f"[PRISM] Could not initialise hook (PRISM may not be installed): {_e}")
            prism_hook = None

    # G0: Base Human Training
    print("\n--- GENERATION 0 (Initial Fine-Tuning) ---")
    g0_dir = project_results_path("multi_task", regime, "generation_0")
    trainer.train_generation(frozen_human_data, g0_dir)
    current_adapter_path = g0_dir

    metrics = evaluate_all_families(trainer, eval_families)

    # PRISM snapshot after G0
    if prism_hook is not None:
        snapshot = prism_hook.on_generation_end(gen_idx=0)
        if snapshot is not None:
            metrics.update({
                "spectral_entropy": snapshot.mean_spectral_entropy,
                "effective_dimension": snapshot.mean_effective_dimension,
                "viability_score": snapshot.mean_viability_score,
                "phase_coherence": snapshot.mean_phase_coherence,
            })

    results.append({"generation": 0, **metrics})

    # Recursive Loop
    for g in range(1, config["recursion"]["max_generations"] + 1):
        print(f"\n--- GENERATION {g} ---")
        trainer._load_model(current_adapter_path)

        prompts = [d["prompt"] for d in frozen_human_data]
        synthetic_data = trainer.generate_synthetic_data(prompts)

        # Derive unified correction fraction from whichever regime fields are set
        correction_fraction = (
            regime_config.get("corrected_fraction", 0.0) +
            regime_config.get("frozen_real_fraction", 0.0) +
            regime_config.get("fresh_real_fraction", 0.0)
        )
        mixed_dataset = mixer.mix_dataset(
            synthetic_data=synthetic_data,
            correction_pool=frozen_human_data,
            target_size=len(frozen_human_data),
            correction_fraction=correction_fraction
        )

        out_dir = project_results_path("multi_task", regime, f"generation_{g}")
        trainer.train_generation(mixed_dataset, out_dir)
        current_adapter_path = out_dir

        metrics = evaluate_all_families(trainer, eval_families)

        # PRISM snapshot after this generation
        if prism_hook is not None:
            snapshot = prism_hook.on_generation_end(gen_idx=g)
            if snapshot is not None:
                metrics.update({
                    "spectral_entropy": snapshot.mean_spectral_entropy,
                    "effective_dimension": snapshot.mean_effective_dimension,
                    "viability_score": snapshot.mean_viability_score,
                    "phase_coherence": snapshot.mean_phase_coherence,
                })

        results.append({"generation": g, **metrics})

        regime_dir = project_results_path("multi_task", regime)
        os.makedirs(regime_dir, exist_ok=True)
        with open(os.path.join(regime_dir, "metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Save PRISM trajectory alongside metrics
    if prism_hook is not None:
        prism_report = prism_hook.get_trajectory_report()
        prism_path = project_results_path("multi_task", regime, "prism_trajectory.json")
        with open(prism_path, "w") as f:
            json.dump(prism_report, f, indent=2)
        print(f"[PRISM] Trajectory saved to {prism_path}")

    print(f"\n=== Finished Benchmark for {regime} ===")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/multi_task_config.yaml")
    parser.add_argument("--regime", type=str, required=True)
    parser.add_argument(
        "--prism",
        action="store_true",
        default=False,
        help="Enable PRISM spectral measurement at each generation end. "
             "Requires spectral-microscope installed (pip install -e ../prism). "
             "Outputs prism_trajectory.json alongside metrics.json.",
    )
    parser.add_argument(
        "--prism-samples",
        type=int,
        default=10,
        help="Number of eval prompts to use per PRISM snapshot (default: 10).",
    )
    args = parser.parse_args()

    run_regime(config_path=args.config, regime=args.regime,
               prism=args.prism, prism_samples=args.prism_samples)

if __name__ == "__main__":
    main()
