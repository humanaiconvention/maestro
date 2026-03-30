"""
Statistical aggregation of multi-seed sweep data.

This script targets the output of the sweep pipeline (run_sweep.py) and expects 
JSONL format at paths matching:
results/sweeps/<experiment>/<model>/c_<fraction>/seed_<seed>/metrics.jsonl

It does NOT support the single-regime multi-task pipeline output (metrics.json).
"""

import os
import json
import pandas as pd
import argparse
import logging
from typing import List, Dict, Any

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from semantic_grounding.statistics.temporal import TemporalAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_metrics_files(base_dir: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(base_dir):
        for f in filenames:
            if f == "metrics.jsonl":
                files.append(os.path.join(root, f))
    return files

def parse_run_metadata(fpath: str, first_line: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts model, correction_fraction, and seed from either the path or the JSON content.
    Path expected: .../<model>/c_<frac>/seed_<seed>/metrics.jsonl
    """
    parts = fpath.replace("\\", "/").split("/")
    
    model = "unknown"
    c_frac_str = "0.0"
    seed_str = "-1"
    use_fallback = False

    # Attempt path parsing
    if len(parts) >= 4:
        model = parts[-4]
        c_frac_str = parts[-3].replace("c_", "")
        seed_str = parts[-2].replace("seed_", "")
        
        # Validate numeric types
        try:
            float(c_frac_str)
            int(seed_str)
        except ValueError:
            use_fallback = True
    else:
        use_fallback = True

    if use_fallback:
        logger.warning(f"Path format unexpected for {fpath}. Falling back to internal metadata.")
        model = first_line.get("model", "unknown")
        c_frac_str = str(first_line.get("correction_fraction", 0.0))
        seed_str = str(first_line.get("seed", -1))
        
    return {
        "model": model,
        "correction_fraction": float(c_frac_str),
        "seed": int(seed_str),
        "path": fpath
    }

def aggregate_and_analyze(base_dir: str, output_csv: str):
    metric_files = find_metrics_files(base_dir)
    print(f"Found {len(metric_files)} run files to analyze.")
    
    analyzer = TemporalAnalyzer(drop_threshold=0.05, rise_threshold=0.05)
    all_temporal_results = []
    
    for fpath in metric_files:
        with open(fpath, 'r') as f:
            lines = [json.loads(l) for l in f]
            
        if not lines:
            continue
            
        meta = parse_run_metadata(fpath, lines[0])
        
        # Analyze temporal signature for ARC-Easy
        temporal_arc = analyzer.analyze_run(lines, ood_key="arc_easy_accuracy", ppl_key="val_perplexity")
        
        # Merge metadata and analysis
        row = {**meta, **temporal_arc}
        all_temporal_results.append(row)
        
    if not all_temporal_results:
        print("No results found.")
        return

    df = pd.DataFrame(all_temporal_results)
    df.to_csv(output_csv, index=False)
    print(f"Aggregated statistical results written to {output_csv}")
    
    # Statistical Summary by (Model, Correction Fraction)
    print("\n" + "="*50)
    print("STATISTICAL SUMMARY BY CONDITION")
    print("="*50)
    
    summary = df.groupby(["model", "correction_fraction"]).agg({
        "delta_t": ["mean", "std", "count"],
        "regime_classification": lambda x: x.mode()[0] if not x.mode().empty else "unknown"
    }).reset_index()
    
    # Flatten columns
    summary.columns = ["model", "correction_fraction", "delta_t_mean", "delta_t_std", "n_seeds", "dominant_regime"]
    print(summary.to_string(index=False))
    
    # Determine overall verdict on temporal ordering
    print("\n" + "="*50)
    print("TEMPORAL SIGNATURE VERDICT")
    print("="*50)
    
    for (model, c_frac), group in df.groupby(["model", "correction_fraction"]):
        if group["delta_t"].notnull().any():
            mean_dt = group["delta_t"].mean()
            if mean_dt < 0:
                verdict = "FALSIFIED (Syntax First)"
            elif mean_dt > 0:
                verdict = "SUPPORTED (Semantic First)"
            else:
                verdict = "SYNCHRONIZED"
            print(f"Model: {model} | C_t: {c_frac} -> {verdict} (mean delta_t={mean_dt:.2f})")
        else:
            print(f"Model: {model} | C_t: {c_frac} -> INCONCLUSIVE (No collapse detected in window)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True, help="Path to the sweep output directory")
    parser.add_argument("--output-csv", type=str, default="aggregated_temporal_stats.csv")
    args = parser.parse_args()
    
    aggregate_and_analyze(args.results_dir, args.output_csv)
