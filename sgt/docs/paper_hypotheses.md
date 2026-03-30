# Paper Hypotheses: Semantic Viability & Information Preservation

This document formalizes the hypotheses tested by the Semantic Grounding Test Module.

## Hypothesis 1 (The Decoupling Effect)
**Recursive synthetic-data training can preserve stylistic fluency (low perplexity) while degrading externally grounded semantic performance (low accuracy).**
- *Metric*: Wikitext-2 Perplexity vs. ARC-Easy/HellaSwag Accuracy.
- *Success Criterion*: Stable or improving PPL alongside declining Acc.

## Hypothesis 2 (The Early-Warning Signature)
**Grounded and OOD performance degrades before validation perplexity clearly worsens.**
- *Metric*: Per-generation slope of Acc vs. per-generation slope of PPL.
- *Success Criterion*: `Acc_Failure_Gen < PPL_Failure_Gen` in R1 (Synthetic-Only).

## Hypothesis 3 (Corrective Bandwidth)
**Semantic preservation improves as the ratio of fresh human data or exogenous correction ($C_t$) increases.**
- *Metric*: Viability Ratio ($C_t / E_t$) vs. Information Preservation Rate (IPR).
- *Success Criterion*: Positive correlation between $C_t$ fraction and final generation IPR.

## Hypothesis 4 (Informational Closure)
**If a model remains coherent internally (low self-entropy) while drifting on reference-sensitive tasks, it has entered a state of informational closure.**
- *Metric*: Shannon Entropy ($H_t$) vs. Grounding Score.
- *Success Criterion*: Decreasing $H_t$ (distributional narrowing) coinciding with loss of referential accuracy.

## Hypothesis 5 (Viability Threshold)
**There exists a non-zero threshold of exogenous grounding ($C_t > \epsilon$) below which model collapse is inevitable regardless of the initial dataset size.**
- *Metric*: Comparison between R2 (Frozen Seed) and R3 (Fresh Real).
- *Success Criterion*: R3 significantly outperforms R2 in long-run ($G_{10}+$) stability.
