"""Plotting scripts for the rigorous sweep analysis."""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import json
import glob
from scripts.path_utils import project_results_path

def generate_plots(csv_file: str, base_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    
    if df.empty:
        print("No aggregated data to plot.")
        return
        
    # 1. Delta T Distribution (Boxplot equivalent in Matplotlib)
    if "delta_t" in df.columns and df["delta_t"].notnull().any():
        plt.figure(figsize=(10, 6))
        data_to_plot = []
        labels = []
        for frac, group in df.groupby("correction_fraction"):
            valid_dt = group["delta_t"].dropna()
            if not valid_dt.empty:
                data_to_plot.append(valid_dt.tolist())
                labels.append(f"C_t={frac}")
        
        if data_to_plot:
            plt.boxplot(data_to_plot, labels=labels)
            plt.axhline(0, color='r', linestyle='--', alpha=0.5, label='Synchronized')
            plt.title("Failure Ordering ($\Delta t = T_{PPL} - T_{OOD}$)")
            plt.ylabel("$\Delta t$ (Generations)")
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(os.path.join(output_dir, "delta_t_distribution.png"))
        plt.close()
    
    # 2. Trajectory Plots (Average across seeds)
    # We need to read the raw metrics again to get trajectories
    print("Generating trajectory plots...")
    metric_files = glob.glob(os.path.join(base_dir, "**", "metrics.jsonl"), recursive=True)
    
    traj_data = []
    for f in metric_files:
        with open(f, 'r') as f_in:
            for line in f_in:
                traj_data.append(json.loads(line))
                
    if traj_data:
        tdf = pd.DataFrame(traj_data)
        
        for model in tdf["model"].unique():
            # Accuracy Trajectory
            plt.figure(figsize=(10, 6))
            for frac in sorted(tdf["correction_fraction"].unique()):
                subset = tdf[(tdf["model"] == model) & (tdf["correction_fraction"] == frac)]
                avg_traj = subset.groupby("generation")["arc_easy_accuracy"].mean()
                plt.plot(avg_traj.index, avg_traj.values, marker='o', label=f"C_t={frac}")
            
            plt.title(f"ARC-Easy Accuracy Trajectory ({model})")
            plt.xlabel("Generation")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"accuracy_trajectory_{model.replace('/', '_')}.png"))
            plt.close()
            
            # Perplexity Trajectory
            plt.figure(figsize=(10, 6))
            for frac in sorted(tdf["correction_fraction"].unique()):
                subset = tdf[(tdf["model"] == model) & (tdf["correction_fraction"] == frac)]
                avg_traj = subset.groupby("generation")["val_perplexity"].mean()
                plt.plot(avg_traj.index, avg_traj.values, marker='s', label=f"C_t={frac}")
            
            plt.title(f"Validation Perplexity Trajectory ({model})")
            plt.xlabel("Generation")
            plt.ylabel("Perplexity")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"perplexity_trajectory_{model.replace('/', '_')}.png"))
            plt.close()

    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to aggregated_temporal_stats.csv")
    parser.add_argument("--results-dir", type=str, required=True, help="Base sweep results dir for trajectories")
    parser.add_argument("--output-dir", type=str, default=project_results_path("sweeps", "plots"))
    args = parser.parse_args()
    
    generate_plots(args.csv, args.results_dir, args.output_dir)
