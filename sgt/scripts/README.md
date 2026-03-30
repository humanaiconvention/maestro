# Execution Scripts

This directory contains the entry points for running benchmarks, processing metrics, and generating reports.

| Script | Purpose | Example Usage |
| :--- | :--- | :--- |
| `run_recursive_regime.py` | Runs a single-task recursive grounding loop. | `python scripts/run_recursive_regime.py --regime R1` |
| `run_multi_task_benchmark.py` | Runs a multi-task recursive benchmark for one regime. | `python scripts/run_multi_task_benchmark.py --regime R4` |
| `run_all_regimes.py` | Orchestrates running all defined regimes sequentially. | `python scripts/run_all_regimes.py` |
| `make_report.py` | Generates a Markdown report and plots from benchmark results. | `python scripts/make_report.py --results-dir results/multi_task` |
| `smoke_test.py` | Fast, GPU-free verification of the mixing logic and pipeline. | `python scripts/smoke_test.py` |
| `aggregate_metrics.py` | Statistical aggregation of multi-seed sweep data. | `python scripts/aggregate_metrics.py --results-dir results/sweeps/X` |
| `generate_falsification_report.py` | Generates a summary report for the falsification sweep. | `python scripts/generate_falsification_report.py --csv stats.csv` |
| `night_run.py` | Automated search for the viability threshold bandwidth. | `python scripts/night_run.py` |
| `plot_results.py` | Generates trajectory and distribution plots for sweeps. | `python scripts/plot_results.py --csv stats.csv --results-dir X` |
| `run_sweep.py` | Matrix runner for large multi-seed experimental sweeps. | `python scripts/run_sweep.py --config configs/sweeps/example_sweep.yaml` |

All scripts should be executed from the project root with `PYTHONPATH=src`.
