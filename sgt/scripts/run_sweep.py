"""Main entry point to execute the robust sweep matrix."""

import argparse
import yaml
import os
import json
from datetime import datetime

# Adjust Python path to allow imports if run from root
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from semantic_grounding.evaluator.tasks import BenchmarkRegistry
from semantic_grounding.training.pipeline import RecursiveTrainer
# Assume we have a generic mixer or will build one
from semantic_grounding.datasets.mixer import RegimeMixer
from semantic_grounding.datasets.synthetic_env import ColorBindingEnvironment

def run_single_condition(config: dict, model_name: str, seed: int, correction_frac: float, output_dir: str):
    print(f"\n{'='*60}")
    print(f"RUNNING: Model={model_name} | Seed={seed} | C_t={correction_frac}")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, "metrics.jsonl")
    
    # 1. Setup Environment and Evaluator
    env = ColorBindingEnvironment(seed=seed)
    registry = BenchmarkRegistry(eval_samples=config["datasets"].get("eval_samples", 100), seed=seed)
    mixer = RegimeMixer(config["recursion"])
    
    # 2. Setup Trainer
    trainer_config = config["training"].copy()
    trainer = RecursiveTrainer(
        model_name=model_name,
        config=trainer_config,
        dtype=config["model"].get("dtype", "float16")
    )
    
    # 3. Generate Initial Human Dataset (Gen 0)
    train_samples = config["recursion"]["synthetic_samples_per_generation"]
    gen0_data = env.generate_human_dataset(train_samples)
    
    # Correction pool (independent ground truth data)
    correction_pool = env.generate_human_dataset(train_samples * 2) 
    
    # Training Loop
    current_adapter = None
    train_data = gen0_data
    
    for gen in range(config["recursion"]["max_generations"] + 1):
        print(f"\n--- Generation {gen} ---")
        
        # Load adapter
        trainer._load_model(current_adapter)
        
        gen_dir = os.path.join(output_dir, f"gen_{gen}")
        
        # Train
        trainer.train_generation(train_data, gen_dir)
        current_adapter = gen_dir
        
        # Reload to ensure we are evaluating the trained model
        trainer._load_model(current_adapter)
        
        # Rigorous Evaluation
        metrics = registry.evaluate_model(trainer, generation_idx=gen)
        metrics["correction_fraction"] = correction_frac
        metrics["seed"] = seed
        metrics["model"] = model_name
        
        # Append to JSONL
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
            
        # Prepare data for NEXT generation
        if gen < config["recursion"]["max_generations"]:
            print(f"Preparing data for Generation {gen+1}...")
            
            # Generate synthetic data from current model
            # We use the same prompts as the original environment for consistency
            prompts = [d["prompt"] for d in env.generate_human_dataset(train_samples)]
            synthetic_gen = trainer.generate_synthetic_data(prompts, max_new_tokens=20)
            
            # Mix with correction according to policy
            train_data = mixer.mix_dataset(
                synthetic_data=synthetic_gen,
                correction_pool=correction_pool,
                target_size=train_samples,
                correction_fraction=correction_frac
            )
            
    print(f"Finished condition. Metrics saved to {metrics_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sweeps/temporal_falsification.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = f"{config['output_dir']}_{timestamp}"
    
    for model in config["models"]:
        for c_frac in config["sweeps"]["correction_fractions"]:
            for seed in config["sweeps"]["seeds"]:
                cond_dir = os.path.join(base_out_dir, model.replace("/", "_"), f"c_{c_frac}", f"seed_{seed}")
                run_single_condition(config, model, seed, c_frac, cond_dir)

if __name__ == "__main__":
    main()
