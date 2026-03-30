import pytest
from semantic_grounding.reporting.drift_metrics import EarlyWarningDetector

def test_grounding_fails_first():
    detector = EarlyWarningDetector(accuracy_threshold=0.10, perplexity_threshold=0.10)
    
    results = [
        {"generation": 0, "accuracy": 1.0, "perplexity": 10.0},
        {"generation": 1, "accuracy": 0.8, "perplexity": 10.1}, # Acc drops 20%, PPL rises 1%
        {"generation": 2, "accuracy": 0.7, "perplexity": 12.0}  # PPL rises 20%
    ]
    
    analysis = detector.analyze_trajectory(results)
    
    assert analysis["signature_detected"] is True
    assert analysis["failure_ordering"] == "grounding_failed_first"
    assert analysis["accuracy_failure_generation"] == 1
    assert analysis["perplexity_failure_generation"] == 2

def test_perplexity_fails_first():
    detector = EarlyWarningDetector(accuracy_threshold=0.10, perplexity_threshold=0.10)
    
    results = [
        {"generation": 0, "accuracy": 1.0, "perplexity": 10.0},
        {"generation": 1, "accuracy": 0.95, "perplexity": 15.0}, # Acc drops 5%, PPL rises 50%
        {"generation": 2, "accuracy": 0.8, "perplexity": 20.0}   # Acc drops 20%
    ]
    
    analysis = detector.analyze_trajectory(results)
    
    assert analysis["signature_detected"] is False
    assert analysis["failure_ordering"] == "perplexity_failed_first"
    assert analysis["accuracy_failure_generation"] == 2
    assert analysis["perplexity_failure_generation"] == 1

def test_grounding_fails_only():
    detector = EarlyWarningDetector(accuracy_threshold=0.10, perplexity_threshold=0.10)
    
    results = [
        {"generation": 0, "accuracy": 1.0, "perplexity": 10.0},
        {"generation": 1, "accuracy": 0.8, "perplexity": 10.0},
        {"generation": 2, "accuracy": 0.5, "perplexity": 10.5}
    ]
    
    analysis = detector.analyze_trajectory(results)
    
    assert analysis["signature_detected"] is True
    assert analysis["failure_ordering"] == "grounding_failed_only"
    assert analysis["accuracy_failure_generation"] == 1
    assert analysis["perplexity_failure_generation"] == -1
    
def test_stable_trajectory():
    detector = EarlyWarningDetector()
    
    results = [
        {"generation": 0, "accuracy": 1.0, "perplexity": 10.0},
        {"generation": 1, "accuracy": 0.95, "perplexity": 10.1},
        {"generation": 2, "accuracy": 0.95, "perplexity": 10.2}
    ]
    
    analysis = detector.analyze_trajectory(results)
    
    assert analysis["signature_detected"] is False
    assert analysis["failure_ordering"] == "stable"
    assert analysis["accuracy_failure_generation"] == -1
    assert analysis["perplexity_failure_generation"] == -1
