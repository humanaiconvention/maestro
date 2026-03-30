import argparse
import yaml
import sys
import os
import traceback

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import scripts.run_multi_task_benchmark as benchmark
import scripts.make_report as reporter
from scripts.path_utils import project_results_path

def main():
    parser = argparse.ArgumentParser(description="Run all recursive training regimes in sequence.")
    parser.add_argument("--config", type=str, default="configs/multi_task_config.yaml", help="Path to config file.")
    parser.add_argument("--regimes", type=str, help="Comma-separated list of regimes to run (default: all in config).")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    all_regimes = list(config.get("regimes", {}).keys())
    if not all_regimes:
        print("Error: No regimes defined in config.")
        sys.exit(1)

    if args.regimes:
        regimes_to_run = [r.strip() for r in args.regimes.split(",")]
        # Validate regimes
        invalid = [r for r in regimes_to_run if r not in all_regimes]
        if invalid:
            print(f"Error: Invalid regimes specified: {invalid}")
            print(f"Available regimes: {all_regimes}")
            sys.exit(1)
    else:
        regimes_to_run = all_regimes

    print(f"Starting benchmark for regimes: {regimes_to_run}")
    results = {}

    for regime in regimes_to_run:
        print(f"\n{'='*60}")
        print(f"RUNNING REGIME: {regime}")
        print(f"{'='*60}\n")
        
        try:
            benchmark.run_regime(config_path=args.config, regime=regime)
            results[regime] = "COMPLETED"
        except Exception as e:
            print(f"\nERROR running regime {regime}:")
            traceback.print_exc()
            results[regime] = "FAILED"

    print(f"\n{'='*60}")
    print("GENERATING FINAL REPORT")
    print(f"{'='*60}\n")

    try:
        reporter.make_report(
            results_dir=project_results_path("multi_task"),
            output_file=project_results_path("multi_task", "final_report.md"),
            plot_dir=project_results_path("multi_task", "plots"),
            task_prefix="grounded_arc"
        )
        report_status = "SUCCESS"
    except Exception as e:
        print("\nERROR generating report:")
        traceback.print_exc()
        report_status = "FAILED"

    # Print summary table
    print(f"\n{'='*40}")
    print(f"{'Regime':<20} | {'Status':<15}")
    print("-" * 40)
    for regime in regimes_to_run:
        status = results.get(regime, "SKIPPED")
        print(f"{regime:<20} | {status:<15}")
    print("-" * 40)
    print(f"{'Final Report':<20} | {report_status:<15}")
    print(f"{'='*40}\n")

    if "FAILED" in results.values() or report_status == "FAILED":
        sys.exit(1)
    else:
        print("All tasks completed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main()
