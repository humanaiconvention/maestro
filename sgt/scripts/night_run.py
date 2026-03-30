"""Overnight autonomous sweep: iterates R4 correction fractions and generates summary.

Runs the multi-task benchmark with correction fractions [10%, 30%, 50%, 80%],
mutating configs/multi_task_config.yaml's R4 block for each step.
Results are appended to results/night_summary.md as a markdown table.

Usage:
    python scripts/night_run.py
"""

import os
import subprocess
import yaml
import json
from scripts.path_utils import project_results_path

def update_config(fraction):
    config_path = "configs/multi_task_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # We will overwrite the R4 configuration for this test
    config["regimes"]["R4"]["corrected_fraction"] = fraction
    config["regimes"]["R4"]["synthetic_fraction"] = 1.0 - fraction
    config["regimes"]["R4"]["name"] = f"synthetic_plus_{int(fraction*100)}pct_correction"
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    
    print(f"Updated config: R4 is now using {fraction*100}% correction.")
    return config["regimes"]["R4"]["name"]

def analyze_and_record(fraction, regime_name):
    # After a run, we pull the metrics and append to the night log
    metrics_path = project_results_path("multi_task", "R4", "metrics.json")
    if not os.path.exists(metrics_path):
        return "Run failed or metrics not found."
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    summary = f"\n### Run with {fraction*100}% Correction ({regime_name})\n"
    summary += "| Generation | ARC Acc | ARC PPL | Syn Acc | Syn PPL |\n"
    summary += "|---|---|---|---|---|\n"
    
    for m in metrics:
        gen = m["generation"]
        arc_acc = m.get("grounded_arc_accuracy", 0.0)
        arc_ppl = m.get("grounded_arc_perplexity", 0.0)
        syn_acc = m.get("synthetic_binding_accuracy", 0.0)
        syn_ppl = m.get("synthetic_binding_perplexity", 0.0)
        
        summary += f"| {gen} | {arc_acc:.2%} | {arc_ppl:.2f} | {syn_acc:.2%} | {syn_ppl:.2f} |\n"
        
    with open(project_results_path("night_summary.md"), "a") as f:
        f.write(summary)
        
    return summary

def main():
    print("Starting Overnight Autonomous Test Cycle...")
    
    # Ensure baseline report file exists
    night_summary = project_results_path("night_summary.md")
    if not os.path.exists(night_summary):
        os.makedirs(os.path.dirname(night_summary), exist_ok=True)
        with open(night_summary, "w") as f:
            f.write("# Overnight Run Analysis: Searching for the Viability Threshold ($C_t$)\n")
            f.write("Testing different correction bandwidths to find where $E_t \le C_t$ holds.\n\n")
    else:
        with open(night_summary, "a") as f:
            f.write(f"\n\n--- RESTARTING SESSION {os.getpid()} ---\n\n")

    test_fractions = [0.1, 0.3, 0.5, 0.8]
    
    for fraction in test_fractions:
        print(f"\n{'='*50}\nDesigning Test: {fraction*100}% Exogenous Correction\n{'='*50}")
        
        regime_name = update_config(fraction)
        
        # CLEAR PREVIOUS R4 RESULTS to ensure we don't read stale data
        r4_path = project_results_path("multi_task", "R4")
        if os.path.exists(r4_path):
            print(f"Cleaning previous results in {r4_path}...")
            import shutil
            # We keep the directory but clear metrics.json
            metrics_file = os.path.join(r4_path, "metrics.json")
            if os.path.exists(metrics_file):
                os.remove(metrics_file)
        
        print("Running benchmark... (this will take a while)")
        
        # Run the benchmark
        cmd = [
            "python", "scripts/run_multi_task_benchmark.py", 
            "--regime", "R4", 
            "--config", "configs/multi_task_config.yaml"
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"ERROR: Benchmark failed for {fraction*100}% correction.")
            with open(night_summary, "a") as f:
                f.write(f"\n### Run with {fraction*100}% Correction ({regime_name})\nFAILED with return code {result.returncode}\n")
            continue

        print("Analyzing results...")
        summary = analyze_and_record(fraction, regime_name)
        print(summary)
        
        # Regenerate global report just in case
        subprocess.run([
            "python",
            "scripts/make_report.py",
            "--results-dir",
            project_results_path("multi_task"),
            "--output",
            project_results_path("multi_task", "final_report.md"),
        ])

    print("Overnight cycle complete.")

if __name__ == "__main__":
    main()
