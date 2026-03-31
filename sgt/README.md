# Semantic Grounding & Information Preservation Benchmark

> Orientation note: this file is the detailed benchmark/manual surface.  
> For subsystem navigation and current HAIC-facing status, start with [README-new.md](./README-new.md), [STATUS.md](./STATUS.md), and [HANDOFF.md](./HANDOFF.md).

**Operationalizing the Semantic Viability Condition ($E_t \le C_t$) for Recursive Learning Systems.**

---

## 1. Overview

The **Semantic Grounding Test Module** is a comprehensive experimental framework designed to test the preservation of information in recursive generative models. As AI systems increasingly train on their own synthetic outputs, they face the risk of **Model Collapse** -- a state where internal coherence is maintained but semantic grounding to reality decays.

This framework allows researchers to:
1.  **Simulate** recursive training across multiple generations ($G_0 \to G_n$).
2.  **Compare** different data-mixing regimes (Regimes R1 through R4).
3.  **Detect** the "Silent Semantic Drift" signature where grounded accuracy fails before stylistic fluency.
4.  **Quantify** the effect of exogenous corrective bandwidth on system stability.
5.  **Diagnose** loss landscape pathologies via Hessian spectral analysis.

## 2. Experimental Thesis

This module is built to test the following core thesis:
> "Semantic grounding is required for the preservation of information in recursive learning systems. Under recursive learning, inherited or indirect grounding is not sufficient for long-run semantic preservation unless an exogenous corrective channel ($C_t$) remains above a viability threshold ($E_t$). The predicted failure signature is that grounded / OOD / reference-sensitive performance degrades before validation perplexity clearly worsens."

Detailed hypotheses can be found in [docs/paper_hypotheses.md](docs/paper_hypotheses.md).

## 3. Regime Definitions

The framework evaluates the following recursive training regimes:

*   **R1 (Synthetic-Only / Replace)**: Pure closed-loop recursion. 100% synthetic data, replaced each generation. Tests the base model collapse rate.
*   **R1_accum (Synthetic-Only / Accumulate)**: Same as R1 but synthetic data is accumulated into a growing history. Tests whether "memory" of previous outputs slows collapse.
*   **R2 (Frozen Real Data Anchor)**: Mix of synthetic + original $G_0$ human seed data (e.g., 50/50). Tests the "Static Grounding" hypothesis.
*   **R3 (Fresh Real Data Replacement)**: Mix of synthetic + newly acquired human data each generation. Tests "Continuous Grounding" stability.
*   **R4 (Exogenous Correction)**: 80% oracle-corrected synthetic data. Tests the Viability Condition ($E_t \le C_t$).

### 3.1 Evaluation Families
-   **Fluency**: Wikitext-2 Perplexity (stylistic coherence)
-   **Grounded Reference**: ARC-Easy accuracy (factual mapping to science)
-   **OOD Robustness**: HellaSwag (common-sense situational grounding)
-   **Controlled Stress Test**: `ColorBindingEnvironment` (symbol-reference binding drift)
-   **Metrics**: Accuracy, Perplexity, ECE (Calibration), Dist-2 (Diversity), Shannon Entropy

### 3.2 Early-Warning Signature Detection
The `EarlyWarningDetector` identifies **Silent Semantic Drift**: cases where grounded accuracy degrades *without* a prior perplexity warning. `signature_detected=True` means accuracy failed first or alone -- the model's semantic grounding collapsed silently.

## 4. Getting Started

### 4.1 Prerequisites
-   Python 3.10+
-   CUDA-compatible GPU (8GB+ VRAM recommended for Qwen-0.5B)
-   `pip install -e .`

### 4.2 Quick Start: Dry Run
Verify the full ML pipeline (model load, LoRA, train, generate, evaluate, report) end-to-end:
```bash
# GPU (float16):
python scripts/dry_run_e2e.py

# CPU-only (slow):
python scripts/dry_run_e2e.py --dtype float32
```
Outputs a report to `results/dry_run/dry_run_report.md`.

### 4.3 Smoke Test
Verify config + mixer integration without loading any model:
```bash
python scripts/smoke_test.py
```

### 4.4 Running Tests
```bash
# 163 tests, ~14s
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=semantic_grounding --cov-report=term-missing
```

### 4.5 Running a Full Benchmark
Run a specific regime:
```bash
python scripts/run_multi_task_benchmark.py --regime R1 --config configs/multi_task_config.yaml
```
Run all regimes sequentially with a summary report:
```bash
python scripts/run_all_regimes.py --config configs/multi_task_config.yaml
```

### 4.6 Generating the Report
```bash
python scripts/make_report.py --results-dir results/multi_task --output results/multi_task/final_report.md
```

### 4.7 Parameter Sweeps
Run a sweep matrix across models, correction fractions, and seeds:
```bash
python scripts/run_sweep.py --config configs/sweeps/example_sweep.yaml
python scripts/aggregate_metrics.py --results-dir results/sweeps
python scripts/generate_falsification_report.py --csv results/sweeps/aggregated_temporal_stats.csv
```

## 5. Architecture

```text
src/semantic_grounding/
    training/
        pipeline.py          # RecursiveTrainer: LoRA fine-tuning, generation, perplexity
    datasets/
        mixer.py             # RegimeMixer: replace/accumulate data policies
        synthetic_env.py     # ColorBindingEnvironment: controlled stress test
        natural/
            loaders.py       # NaturalDatasetLoader: ARC, HellaSwag, Wikitext
    evaluator/
        tasks.py             # BenchmarkRegistry: immutable hold-out evaluation
    evaluator_core.py        # SemanticGroundingEvaluator + EvaluationResult
    hessian_diagnostics.py   # HessianCompute, FailureModeClassifier, LandscapeMetrics
    metrics.py               # GroundingMetrics: exact + embedding-based matching
    reporting/
        drift_metrics.py     # EarlyWarningDetector: silent semantic drift signature
        plotting.py          # ResultsPlotter: trajectory + early-warning gap plots
        calibration.py       # CalibrationMetrics (ECE) + DiversityMetrics (Dist-n)
    analysis/
        early_warning.py     # EarlyWarningAnalyzer: T_acc vs T_ppl detection
    statistics/
        temporal.py          # TemporalAnalyzer: T_OOD, T_PPL, delta_t, regime class

scripts/
    run_multi_task_benchmark.py  # Single-regime benchmark runner
    run_all_regimes.py           # Multi-regime orchestrator + report
    run_sweep.py                 # Parameter sweep matrix
    aggregate_metrics.py         # Sweep JSONL aggregation + temporal stats
    make_report.py               # Markdown report generator
    generate_falsification_report.py  # Falsification verdict from sweep stats
    plot_results.py              # Sweep trajectory + delta_t plots
    night_run.py                 # Overnight autonomous correction-fraction sweep
    smoke_test.py                # Quick mixer + config validation
    dry_run_e2e.py               # Full ML pipeline validation (1 gen, 10 samples)

configs/
    multi_task_config.yaml       # Main experiment config (model, regimes, training)
    sweeps/
        example_sweep.yaml       # Template for parameter sweep runs
```

## 6. Test Coverage

| Module | Tests | Notes |
|--------|-------|-------|
| GroundingMetrics (exact + embedding) | 19 | Includes sentence-transformers integration |
| SemanticGroundingEvaluator | 23 | LoRA, landscape metrics, batch eval |
| HessianCompute / FailureModeClassifier | 28 | Spectral analysis, power law, stability |
| RegimeMixer | 6 | All regimes, accumulate/replace policies |
| EarlyWarningDetector | 4 | Silent drift signature detection |
| EarlyWarningAnalyzer | 4 | T_acc vs T_ppl temporal detection |
| TemporalAnalyzer | 6 | Collapse classification |
| BenchmarkRegistry + NaturalDatasetLoader | 11 | Mocked HF datasets, error handling |
| CalibrationMetrics + DiversityMetrics | 7 | ECE, n-gram diversity, self-BLEU |
| Orchestration (make_report, run_all) | 6 | Report generation, CLI validation |
| Pipeline integration | 3 | End-to-end report content verification |
| Config + mixer smoke tests | 13 | Parametrized over all real regimes |
| run_regime config parsing | 4 | R1/R4 config read, missing key, output |
| **Total** | **163** | **0 skipped, 0 failures** |

## 7. Implementation Plan
The detailed implementation roadmap is documented in [docs/experiment_plan.md](docs/experiment_plan.md).

---
**HumanAI Convention Research**
*Part of the Semantic Grounding and Information Preservation Series.*
