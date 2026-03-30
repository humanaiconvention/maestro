#!/usr/bin/env python
"""
Example: Hessian-based Diagnostic Workflow

This example demonstrates how to use the Hessian diagnostic system to detect
failure modes in semantic grounding models:

1. Hallucinations (confident interpolation to nowhere) - high condition number κ
2. Distribution Shift Failures (memorized patterns don't transfer) - high spectral sharpness ε
3. Adversarial Brittleness (small input changes cause large output swings) - negative eigenvalues δ

This is a standalone example that doesn't require actual model weights.
It uses mock eigenvalues to demonstrate the diagnostic workflow.
"""

import numpy as np
from semantic_grounding.hessian_diagnostics import (
    compute_landscape_metrics,
    FailureModeClassifier,
    RiskLevel,
)


def example_well_conditioned_model():
    """Example 1: Well-conditioned model (SAFE)."""
    print("\n" + "=" * 80)
    print("Example 1: Well-Conditioned Model (SAFE)")
    print("=" * 80)

    # Simulate eigenvalues from a well-trained, robust model
    # - Small condition number (50)
    # - Low spectral sharpness (0.4)
    # - No negative eigenvalues
    eigenvalues = np.array([0.4, 0.32, 0.25, 0.18, 0.12, 0.08, 0.05, 0.02, 0.01, 0.008])

    print(f"\nEigenvalue spectrum: {eigenvalues[:5]}... (showing first 5)")
    print(f"Condition number: {eigenvalues[0] / eigenvalues[-1]:.1f}")

    # Compute landscape metrics
    metrics = compute_landscape_metrics(eigenvalues)

    print(f"\nLandscape Metrics:")
    print(f"  Condition Number (κ):        {metrics.condition_number:.1f}")
    print(f"  Spectral Sharpness (λ_max):  {metrics.spectral_sharpness:.2f}")
    print(f"  Min Eigenvalue:              {metrics.min_eigenvalue:.4f}")
    print(f"  Has Negative Eigenvalues:    {metrics.has_negative_eigenvalues}")
    print(f"  Stability Score:             {metrics.stability_score:.2f}")
    print(f"  Power-law Compliance:        {metrics.power_law_compliance:.2f}")

    # Classify failure modes
    classifier = FailureModeClassifier()
    report = classifier.generate_diagnostic_report(metrics)

    print(f"\nDiagnostic Report:")
    print(f"  Overall Risk: {report['overall_risk']}")
    print(f"  Stability Score: {report['stability_score']:.2f}")

    print(f"\nFailure Mode Analysis:")
    for mode, analysis in report["failure_modes"].items():
        print(f"  {mode.replace('_', ' ').title()}:")
        print(f"    Risk Level: {analysis['risk_level']}")
        print(f"    Confidence: {analysis['confidence']:.2f}")
        print(f"    Metric: {analysis['metric']} = {analysis['value']}")

    print(f"\nRecommendations:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")


def example_hallucination_prone_model():
    """Example 2: Hallucination-prone model (high condition number)."""
    print("\n" + "=" * 80)
    print("Example 2: Hallucination-Prone Model (High Condition Number)")
    print("=" * 80)

    # Simulate eigenvalues from a model with ill-conditioned landscape
    # - Very high condition number (5000)
    # - Large spread in eigenvalues
    # - Indicates flat, featureless regions where model may hallucinate
    eigenvalues = np.array([5.0, 3.5, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001])

    print(f"\nEigenvalue spectrum: {eigenvalues[:5]}... (showing first 5)")
    print(f"Condition number: {eigenvalues[0] / eigenvalues[-1]:.1f}")

    metrics = compute_landscape_metrics(eigenvalues)

    print(f"\nLandscape Metrics:")
    print(f"  Condition Number (κ):        {metrics.condition_number:.1f} ⚠️ HIGH")
    print(f"  Spectral Sharpness (λ_max):  {metrics.spectral_sharpness:.2f}")
    print(f"  Stability Score:             {metrics.stability_score:.2f}")

    classifier = FailureModeClassifier()
    report = classifier.generate_diagnostic_report(metrics)

    print(f"\nDiagnostic Report:")
    print(f"  Overall Risk: {report['overall_risk']} ⚠️")

    print(f"\nHallucination Risk Analysis:")
    hall = report["failure_modes"]["hallucination"]
    print(f"  Risk Level: {hall['risk_level']}")
    print(f"  Confidence: {hall['confidence']:.2f}")

    print(f"\nKey Recommendation:")
    # Find hallucination-related recommendation
    hall_recs = [r for r in report["recommendations"] if "CONDITION NUMBER" in r]
    if hall_recs:
        print(f"  {hall_recs[0]}")


def example_distribution_shift_vulnerable():
    """Example 3: Distribution shift vulnerable model (high sharpness)."""
    print("\n" + "=" * 80)
    print("Example 3: Distribution Shift Vulnerable Model (Sharp Minima)")
    print("=" * 80)

    # Simulate eigenvalues from a model in sharp minima
    # - High spectral sharpness (3.0)
    # - May fail under distribution shift
    # - Memorized training patterns don't transfer well
    eigenvalues = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.5, 0.3, 0.1, 0.05])

    print(f"\nEigenvalue spectrum: {eigenvalues[:5]}... (showing first 5)")
    print(f"Maximum eigenvalue (sharpness): {eigenvalues[0]:.2f}")

    metrics = compute_landscape_metrics(eigenvalues)

    print(f"\nLandscape Metrics:")
    print(f"  Spectral Sharpness (λ_max):  {metrics.spectral_sharpness:.2f} ⚠️ HIGH")
    print(f"  Condition Number (κ):        {metrics.condition_number:.1f}")
    print(f"  Stability Score:             {metrics.stability_score:.2f}")

    classifier = FailureModeClassifier()
    report = classifier.generate_diagnostic_report(metrics)

    print(f"\nDiagnostic Report:")
    print(f"  Overall Risk: {report['overall_risk']} ⚠️")

    print(f"\nDistribution Shift Risk Analysis:")
    dist = report["failure_modes"]["distribution_shift"]
    print(f"  Risk Level: {dist['risk_level']}")
    print(f"  Confidence: {dist['confidence']:.2f}")

    print(f"\nKey Recommendation:")
    dist_recs = [r for r in report["recommendations"] if "SPECTRAL SHARPNESS" in r]
    if dist_recs:
        print(f"  {dist_recs[0]}")


def example_adversarial_vulnerable():
    """Example 4: Adversarially vulnerable model (negative eigenvalues)."""
    print("\n" + "=" * 80)
    print("Example 4: Adversarially Vulnerable Model (Saddle Points)")
    print("=" * 80)

    # Simulate eigenvalues from a model at saddle point
    # - Contains negative eigenvalues
    # - Indicates unstable equilibrium
    # - Small perturbations can cause large output changes
    eigenvalues = np.array([2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, -0.02, -0.08, -0.15])

    print(f"\nEigenvalue spectrum: {eigenvalues}")
    print(f"Negative eigenvalues: {eigenvalues[eigenvalues < 0]}")

    metrics = compute_landscape_metrics(eigenvalues)

    print(f"\nLandscape Metrics:")
    print(f"  Min Eigenvalue:              {metrics.min_eigenvalue:.4f} ⚠️ NEGATIVE")
    print(f"  Has Negative Eigenvalues:    {metrics.has_negative_eigenvalues} ⚠️")
    print(f"  Stability Score:             {metrics.stability_score:.2f}")

    classifier = FailureModeClassifier()
    report = classifier.generate_diagnostic_report(metrics)

    print(f"\nDiagnostic Report:")
    print(f"  Overall Risk: {report['overall_risk']} ⚠️")

    print(f"\nAdversarial Brittleness Analysis:")
    adv = report["failure_modes"]["adversarial_brittleness"]
    print(f"  Risk Level: {adv['risk_level']}")
    print(f"  Confidence: {adv['confidence']:.2f}")

    print(f"\nKey Recommendation:")
    adv_recs = [r for r in report["recommendations"] if "NEGATIVE EIGENVALUE" in r]
    if adv_recs:
        print(f"  {adv_recs[0]}")


def example_pathological_landscape():
    """Example 5: Pathological landscape (all failure modes)."""
    print("\n" + "=" * 80)
    print("Example 5: Pathological Landscape (Multiple Failure Modes)")
    print("=" * 80)

    # Simulate worst-case scenario
    # - High condition number
    # - High spectral sharpness
    # - Negative eigenvalues
    eigenvalues = np.array([10.0, 8.0, 5.0, 2.0, 0.5, 0.1, 0.01, -0.05, -0.2, -0.5])

    print(f"\nEigenvalue spectrum: {eigenvalues}")
    print(f"⚠️ WARNING: Multiple risk factors detected!")

    metrics = compute_landscape_metrics(eigenvalues)

    print(f"\nLandscape Metrics:")
    print(f"  Condition Number (κ):        {metrics.condition_number:.1f} ⚠️")
    print(f"  Spectral Sharpness (λ_max):  {metrics.spectral_sharpness:.2f} ⚠️")
    print(f"  Has Negative Eigenvalues:    {metrics.has_negative_eigenvalues} ⚠️")
    print(f"  Stability Score:             {metrics.stability_score:.2f} ⚠️ LOW")

    classifier = FailureModeClassifier()
    report = classifier.generate_diagnostic_report(metrics)

    print(f"\nDiagnostic Report:")
    print(f"  Overall Risk: {report['overall_risk']} 🚨 CRITICAL")
    print(f"  Stability Score: {report['stability_score']:.2f}/1.00")

    print(f"\nAll Failure Modes:")
    for mode, analysis in report["failure_modes"].items():
        risk_emoji = "🚨" if analysis["risk_level"] == "RISK" else "⚠️"
        print(f"  {mode.replace('_', ' ').title()}: {analysis['risk_level']} {risk_emoji}")

    print(f"\nCritical Recommendations:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")

    print(f"\nResearch Caveats:")
    for i, caveat in enumerate(report["research_caveats"][:3], 1):
        print(f"  {i}. {caveat}")


def example_custom_thresholds():
    """Example 6: Using custom thresholds."""
    print("\n" + "=" * 80)
    print("Example 6: Custom Thresholds for Specific Model Family")
    print("=" * 80)

    # Suppose you have empirical data for your specific model family
    # You can customize thresholds based on your findings
    custom_thresholds = {
        "condition_number_safe": 200.0,  # More lenient for transformers
        "condition_number_caution": 2000.0,
        "spectral_sharpness_safe": 1.0,  # Different for your architecture
        "spectral_sharpness_caution": 3.0,
    }

    classifier = FailureModeClassifier(custom_thresholds=custom_thresholds)

    # Same eigenvalues as Example 2
    eigenvalues = np.array([5.0, 3.5, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001])
    metrics = compute_landscape_metrics(eigenvalues)

    print(f"\nUsing custom thresholds for transformer models:")
    print(f"  Condition Number Safe:    < {custom_thresholds['condition_number_safe']:.0f}")
    print(f"  Condition Number Caution: < {custom_thresholds['condition_number_caution']:.0f}")

    report = classifier.generate_diagnostic_report(metrics)

    print(f"\nWith Default Thresholds:")
    default_classifier = FailureModeClassifier()
    default_report = default_classifier.generate_diagnostic_report(metrics)
    print(f"  Hallucination Risk: {default_report['failure_modes']['hallucination']['risk_level']}")

    print(f"\nWith Custom Thresholds:")
    print(f"  Hallucination Risk: {report['failure_modes']['hallucination']['risk_level']}")

    print(
        f"\n💡 Tip: Calibrate thresholds based on empirical data from your model family!"
    )


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("HESSIAN-BASED DIAGNOSTIC SYSTEM - EXAMPLE WORKFLOW")
    print("=" * 80)
    print("\nThis example demonstrates how to use Hessian diagnostics to detect")
    print("failure modes in semantic grounding models.")
    print(
        "\nNote: These examples use mock eigenvalues for demonstration."
    )
    print(
        "In production, eigenvalues would be computed from actual model Hessians."
    )

    # Run all examples
    example_well_conditioned_model()
    example_hallucination_prone_model()
    example_distribution_shift_vulnerable()
    example_adversarial_vulnerable()
    example_pathological_landscape()
    example_custom_thresholds()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Low condition number (κ < 100) → Safe from hallucinations")
    print("  2. Low spectral sharpness (λ_max < 0.5) → Robust to distribution shift")
    print("  3. No negative eigenvalues → Stable, not adversarially brittle")
    print("  4. High stability score (> 0.7) → Overall robust model")
    print("\nWarnings:")
    print("  • Thresholds are empirical baselines, not universal constants")
    print("  • Limited research on transformers (QWEN/LLAMA)")
    print("  • Calibrate thresholds for your specific model family")
    print(
        "  • Correlation ≠ causation (flatness doesn't guarantee robustness)"
    )
    print("\nFor production use:")
    print("  • Compute actual Hessian eigenvalues from model loss")
    print("  • Use HessianCompute class with real models")
    print("  • Calibrate thresholds on validation data")
    print("  • Integrate into continuous evaluation pipeline")
    print("\nSee docs/HESSIAN_RESEARCH_PROMPT.md for research directions.")
    print("=" * 80)


if __name__ == "__main__":
    main()
