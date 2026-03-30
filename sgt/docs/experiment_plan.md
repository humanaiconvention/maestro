# Semantic Grounding Experiment Plan

This document outlines the roadmap for measuring semantic drift and model collapse in recursive learning systems.

## 1. Experimental Goals
*   **Quantify Semantic Decay**: Measure the rate of factual/grounding loss across multiple generations ($G_0 \to G_n$).
*   **Identify the Early Warning Gap**: Detect the temporal lag between the degradation of grounded accuracy and the collapse of stylistic fluency.
*   **Map the Viability Threshold**: Determine the minimum exogenous corrective bandwidth ($C_t$) required to satisfy the condition $E_t \le C_t$.
*   **Evaluate Data Policies**: Compare 'replace' vs 'accumulate' strategies for preserving long-run information.

## 2. Phase 1: Regime Comparison
Run the primary recursive loops for five generations across all defined regimes:
*   **R1 (Pure Synthetic)**: Baseline for maximum collapse rate.
*   **R2 (Frozen Seed)**: Test if a static grounding anchor prevents drift.
*   **R3 (Fresh Real)**: Measure stability under continuous real-world data injection.
*   **R4 (Exogenous Correction)**: Measure the impact of correcting synthetic errors before re-training.

**Deliverables**: `metrics.json` for each regime and initial trajectory plots.

## 3. Phase 2: Early Warning Signature Detection
Utilize the `scripts/make_report.py` and `EarlyWarningDetector` to find the gap between:
*   $T_{grounding}$: The generation where ARC-Easy or HellaSwag accuracy drops by >10%.
*   $T_{fluency}$: The generation where Wikitext-2 perplexity increases by >5%.

**Hypothesis**: $T_{grounding} < T_{fluency}$ in most uncorrected regimes (Silent Semantic Drift).

## 4. Phase 3: Ablation Studies
Systematically vary hyperparameters to test the robustness of the findings:
*   **LoRA Rank (`lora_r`)**: Test if higher capacity models delay or accelerate collapse (ranks 4, 8, 16, 32).
*   **Training Intensity (`epochs_per_generation`)**: Measure if more epochs per generation increase the "poisoning" effect of synthetic data.
*   **Sample Size (`target_size`)**: Evaluate the impact of data volume on the stability of the recursive loop.

## 5. Expected Outcomes

| Regime | Predicted Failure Gen | Signature? |
|---|---|---|
| **R1 (Replace)** | Gen 2-3 | Yes (Grounding fails first) |
| **R1_accum** | Gen 4-5 | Yes (Delayed collapse, growing pool) |
| **R2 (Frozen Seed)** | None / Stable | No |
| **R3 (Fresh Real)** | None / Stable | No |
| **R4 (80% Correction)** | None / Stable | No (oracle correction stabilises) |

## 6. Component Status

| Component | Status | Test Coverage | Notes |
| :--- | :--- | :--- | :--- |
| **RegimeMixer** | ✅ Complete | 6 tests | Accumulate + replace policies |
| **EarlyWarningDetector** | ✅ Complete | 4 tests | grounding_failed_first/only = signature |
| **EarlyWarningAnalyzer** | ✅ Complete | 4 tests | Silent Semantic Drift detection |
| **TemporalAnalyzer** | ✅ Complete | 6 tests | T_OOD, T_PPL, delta_t, regime_classification |
| **HessianCompute / FailureModeClassifier** | ✅ Complete | 28 tests | Loss landscape diagnostics |
| **SemanticGroundingEvaluator** | ✅ Complete | 23 tests | LandscapeMetrics integration |
| **make_report / ResultsPlotter** | ✅ Complete | 9 tests | Multi-regime markdown reports |
| **RecursiveTrainer** | 🔲 Stub | 0 unit tests | Requires ML stack (LoRA + PEFT) |

> [!IMPORTANT]
> **Key Semantic Note**: `signature_detected=True` specifically identifies the "Silent Semantic Drift" failure mode. This means that semantic grounding accuracy degraded significantly *without* a preceding rise in validation perplexity. In this state, the model remains stylistically fluent but has become factually untethered from its original training reality.
