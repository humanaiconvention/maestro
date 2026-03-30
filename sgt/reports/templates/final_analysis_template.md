# Falsification Report: Temporal Signatures in Recursive Grounding

## 1. Executive Summary
This report presents the findings of a rigorous, multi-seed sweep designed to test the temporal signature claim regarding semantic vs. structural collapse in recursive models. 

**Core Claim Tested:** Does Out-of-Distribution (OOD) accuracy degrade *before* validation perplexity rises? ($\Delta t > 0$)

### Top-Level Verdict
*(To be populated by automated script)*

## 2. Methodology
- **Immutable Benchmarks:** ARC-Easy, GSM8K, and Wikitext-2 were held strictly outside the training loop.
- **Falsification Thresholds:**
  - $T_{OOD}$: First generation with $>5\%$ relative drop in OOD accuracy.
  - $T_{PPL}$: First generation with $>5\%$ relative rise in validation perplexity.
  - $\Delta t = T_{PPL} - T_{OOD}$

## 3. Statistical Findings ($\Delta t$ Analysis)

<!-- STATS_TABLE_START -->
| Correction Fraction | N (Seeds) | Mean $\Delta t$ | Dominant Regime |
|---|---|---|---|
<!-- STATS_TABLE_END -->

## 4. Viability Threshold Analysis
*(To be populated with Gen 0 vs Gen N endpoint analysis to test the $E_t \le C_t$ claim)*

## 5. What this does and does not falsify
*   **Does this falsify the temporal ordering claim?** [Insert derived verdict: Yes/No/Mixed depending on $\Delta t < 0$ frequency].
*   **Does this falsify the broader corrective-bandwidth viability idea?** [Insert derived verdict: No, if endpoint viability scales with $C_t$].

## 6. Threats to Validity
*   Benchmark Sensitivity: Is ARC-Easy too coarse?
*   Model Scale: Does this hold for >7B parameter models?
*   Prompting Artifacts: Did strict match grading penalize format changes rather than semantic loss?
