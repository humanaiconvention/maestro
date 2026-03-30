"""Script to run the full recursive regime benchmark."""

import argparse
import yaml
import os
import json
import torch
import numpy as np

from semantic_grounding.training.pipeline import RecursiveTrainer
from semantic_grounding.datasets.synthetic_env import ColorBindingEnvironment
from semantic_grounding.datasets.mixer import RegimeMixer
from scripts.path_utils import project_results_path

def main():
    parser = argparse.ArgumentParser(description="Run Recursive Grounding Benchmark")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml", help="Path to config file")
    parser.add_argument("--regime", type=str, required=True, help="Which regime to run (e.g., R1, R2)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    regime_config = config["regimes"].get(args.regime)
    if not regime_config:
        raise ValueError(f"Regime {args.regime} not found in config.")

    print(f"=== Starting Benchmark for Regime: {regime_config['name']} ===")

    # 1. Initialize Environment
    env = ColorBindingEnvironment(seed=42)
    eval_set = env.generate_eval_set()
    
    # 2. Generate Initial Datasets
    n_samples = config["recursion"]["synthetic_samples_per_generation"]
    frozen_human_data = env.generate_human_dataset(n_samples)
    
    mixer = RegimeMixer(regime_config)
    
    # 3. Initialize Trainer (Load base model)
    trainer = RecursiveTrainer(model_name=config["model"]["base_model"], config=config["model"])

    # Track results
    results = []

    # 4. The Recursive Loop
    current_adapter_path = None
    
    # G0 is a special case: Train the base model on the initial human data
    print("\n--- GENERATION 0 (Initial Fine-Tuning) ---")
    g0_dir = project_results_path(args.regime, "generation_0")
    trainer.train_generation(frozen_human_data, g0_dir)
    current_adapter_path = g0_dir
    
    # Eval G0
    g0_acc, g0_ppl = evaluate_model(trainer, eval_set)
    results.append({"generation": 0, "accuracy": g0_acc, "perplexity": g0_ppl})
    
    # Loop over subsequent generations
    for g in range(1, config["recursion"]["max_generations"] + 1):
        print(f"\n--- GENERATION {g} ---")
        
        # 4a. Reload model with current adapter
        trainer._load_model(current_adapter_path)
        
        # 4b. Generate Synthetic Data
        prompts = [d["prompt"] for d in frozen_human_data] # Use the same question structure
        synthetic_data = trainer.generate_synthetic_data(prompts)
        
        # 4c. Generate Fresh Real Data (if needed)
        fresh_human_data = []
        if regime_config["fresh_real_fraction"] > 0:
            # Re-seed to get 'fresh' examples of the same distribution
            fresh_env = ColorBindingEnvironment(seed=42+g) 
            fresh_human_data = fresh_env.generate_human_dataset(n_samples)
            
        # 4d. Mix Dataset according to Regime policy
        mixed_dataset = mixer.mix_dataset(
            synthetic_data=synthetic_data,
            frozen_human_data=frozen_human_data,
            fresh_human_data=fresh_human_data,
            target_size=n_samples
        )
        
        # 4e. Train new generation
        out_dir = project_results_path(args.regime, f"generation_{g}")
        trainer.train_generation(mixed_dataset, out_dir)
        current_adapter_path = out_dir
        
        # 4f. Evaluate
        acc, ppl = evaluate_model(trainer, eval_set)
        results.append({"generation": g, "accuracy": acc, "perplexity": ppl})
        
        # Save intermediate results
        regime_dir = project_results_path(args.regime)
        os.makedirs(regime_dir, exist_ok=True)
        with open(os.path.join(regime_dir, "metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n=== Finished Benchmark for {args.regime} ===")
    print(json.dumps(results, indent=2))

def evaluate_model(trainer, eval_set) -> tuple[float, float]:
    """Evaluates accuracy and perplexity on the evaluation set."""
    print("Evaluating current generation...")
    prompts = [d["prompt"] for d in eval_set]
    predictions = trainer.generate_synthetic_data(prompts, max_new_tokens=5)
    
    correct = 0
    for pred, expected in zip(predictions, eval_set):
        if expected["expected"].lower() in pred["completion"].lower():
            correct += 1
            
    acc = correct / len(eval_set)
    
    # Calculate perplexity on the eval set (prompt + expected)
    ppl = trainer.calculate_perplexity(eval_set)
    
    print(f"Accuracy: {acc:.2%} | Perplexity: {ppl:.2f}")
    return acc, ppl

if __name__ == "__main__":
    main()
