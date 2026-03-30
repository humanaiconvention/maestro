"""Script to generate a final markdown report of all tested regimes."""

import os
import json
import argparse
from semantic_grounding.reporting.drift_metrics import EarlyWarningDetector
from semantic_grounding.reporting.plotting import ResultsPlotter
from scripts.path_utils import project_results_path

def make_report(results_dir: str, output_file: str, plot_dir: str, task_prefix: str = "grounded_arc"):
    print(f"Generating report from {results_dir} for task: {task_prefix}...")
    raw_regime_results = {}    # for plotter
    regime_results = {}        # for EarlyWarningDetector

    # 1. Collect Results
    if os.path.isdir(results_dir):
        # Scan recursively for multi_task or direct results
        for root, dirs, files in os.walk(results_dir):
            if "metrics.json" in files:
                regime_name = os.path.basename(root)
                with open(os.path.join(root, "metrics.json"), "r") as f:
                    raw_metrics = json.load(f)

                raw_regime_results[regime_name] = raw_metrics

                normalized_metrics = []
                for m in raw_metrics:
                    acc_key = f"{task_prefix}_accuracy"
                    ppl_key = f"{task_prefix}_perplexity"
                    normalized_metrics.append({
                        "generation": m["generation"],
                        "accuracy": m.get(acc_key, 0.0),
                        "perplexity": m.get(ppl_key, 0.0)
                    })
                regime_results[regime_name] = normalized_metrics

    if not regime_results:
        print("No metrics found. Please run the regimes first.")
        return

    # 2. Analyze trajectories and Plot
    detector = EarlyWarningDetector(accuracy_threshold=0.10, perplexity_threshold=0.05)
    ResultsPlotter.plot_regime_trajectories(raw_regime_results, plot_dir)

    
    analysis_results = {}
    for regime, metrics in regime_results.items():
        analysis_results[regime] = detector.analyze_trajectory(metrics)

    # 3. Generate Markdown
    md = [
        "# Recursive Grounding Benchmark Report\n",
        "## Overview\n",
        f"This report summarizes the degradation trajectories for **{task_prefix}** under various recursive training regimes.\n",
        "The key hypothesis is that **grounded semantic performance degrades before validation perplexity worsens** in synthetic-heavy regimes.\n\n",
        "## Regime Analysis\n"
    ]
    
    # Table header
    md.append("| Regime | Signature Detected? | Failure Ordering | Acc Drop Gen | PPL Rise Gen |")
    md.append("|---|---|---|---|---|")
    
    for regime, analysis in analysis_results.items():
        sig_detected = "✅ YES" if analysis["signature_detected"] else "❌ NO"
        ordering = analysis["failure_ordering"]
        acc_gen = analysis["accuracy_failure_generation"] if analysis["accuracy_failure_generation"] != -1 else "N/A"
        ppl_gen = analysis["perplexity_failure_generation"] if analysis["perplexity_failure_generation"] != -1 else "N/A"
        
        md.append(f"| {regime} | {sig_detected} | {ordering} | Gen {acc_gen} | Gen {ppl_gen} |")
        
    md.append("\n## Detailed Trajectories\n")

    for regime, metrics in regime_results.items():
        md.append(f"### {regime}\n")
        md.append("| Generation | Accuracy | Perplexity |")
        md.append("|---|---|---|")
        for m in metrics:
            md.append(f"| {m['generation']} | {m['accuracy']:.2%} | {m['perplexity']:.2f} |")
        md.append("\n")

    # PRISM geometric drift section — only rendered if prism_trajectory.json files exist
    prism_sections = []
    for root, dirs, files in os.walk(results_dir):
        if "prism_trajectory.json" in files:
            regime_name = os.path.basename(root)
            prism_path = os.path.join(root, "prism_trajectory.json")
            try:
                with open(prism_path, "r") as f:
                    prism_data = json.load(f)
                trajectory = prism_data.get("trajectory", [])
                if trajectory:
                    prism_sections.append((regime_name, trajectory, prism_data))
            except Exception:
                pass

    if prism_sections:
        md.append("\n---\n")
        md.append("## Geometric Drift Trajectory (PRISM)\n")
        md.append(
            "> Spectral entropy, effective dimension, and phase coherence measured by PRISM "
            "at each generation end. Rising spectral entropy or falling effective dimension "
            "before accuracy drops is the mechanistic precursor to silent semantic drift.\n"
        )

        for regime_name, trajectory, prism_data in prism_sections:
            md.append(f"\n### {regime_name}\n")
            md.append("| Gen | Spectral Entropy | Effective Dim | Viability Score | Phase Coherence |")
            md.append("|-----|-----------------|---------------|-----------------|-----------------|")
            for row in trajectory:
                md.append(
                    f"| {row['generation']} "
                    f"| {row.get('spectral_entropy', 0.0):.4f} "
                    f"| {row.get('effective_dimension', 0.0):.2f} "
                    f"| {row.get('viability_score', 0.0):.4f} "
                    f"| {row.get('phase_coherence', 0.0):.4f} |"
                )

            # Summarise deltas if available
            deltas = prism_data.get("deltas", [])
            if deltas:
                md.append("\n**Delta summary (generation-over-generation):**\n")
                md.append("| Gen→Gen | ΔSpectral Entropy | ΔEffective Dim | Verified? |")
                md.append("|---------|-------------------|----------------|-----------|")
                for d in deltas:
                    snap_b = d.get("snapshot_before", {})
                    snap_a = d.get("snapshot_after", {})
                    g_from = snap_b.get("generation_idx", "?")
                    g_to = snap_a.get("generation_idx", "?")
                    verified = "YES" if d.get("reduction_verified") else "NO"
                    md.append(
                        f"| {g_from}→{g_to} "
                        f"| {d.get('delta_spectral_entropy', 0.0):+.4f} "
                        f"| {d.get('delta_effective_dimension', 0.0):+.2f} "
                        f"| {verified} |"
                    )
            md.append("\n")

    # 4. Write output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Report generated at {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=project_results_path())
    parser.add_argument("--output", type=str, default=project_results_path("final_report.md"))
    parser.add_argument("--plot-dir", type=str, default=project_results_path("plots"))
    parser.add_argument("--task-prefix", type=str, default="grounded_arc")
    args = parser.parse_args()
    
    make_report(args.results_dir, args.output, args.plot_dir, args.task_prefix)
