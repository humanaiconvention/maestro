import pytest
from semantic_grounding.statistics.temporal import TemporalAnalyzer

@pytest.fixture
def analyzer():
    return TemporalAnalyzer(drop_threshold=0.05, rise_threshold=0.05)

def test_accuracy_first_collapse(analyzer):
    """Test scenario where OOD accuracy drops BEFORE perplexity rises."""
    # Gen 0: Base
    # Gen 1: Acc drops by 10% (0.8 -> 0.72)
    # Gen 2: PPL rises by 10% (10 -> 11)
    results = [
        {"generation": 0, "arc_easy_accuracy": 0.8, "val_perplexity": 10.0},
        {"generation": 1, "arc_easy_accuracy": 0.7, "val_perplexity": 10.2},
        {"generation": 2, "arc_easy_accuracy": 0.6, "val_perplexity": 11.5}
    ]
    analysis = analyzer.analyze_run(results)
    
    assert analysis["T_OOD"] == 1
    assert analysis["T_PPL"] == 2
    assert analysis["delta_t"] == 1 # 2 - 1
    assert analysis["regime_classification"] == "accuracy_first"

def test_perplexity_first_collapse(analyzer):
    """Test scenario where perplexity rises BEFORE OOD accuracy drops."""
    # Gen 0: Base
    # Gen 1: PPL rises by 10% (10 -> 11)
    # Gen 2: Acc drops by 10% (0.8 -> 0.72)
    results = [
        {"generation": 0, "arc_easy_accuracy": 0.8, "val_perplexity": 10.0},
        {"generation": 1, "arc_easy_accuracy": 0.79, "val_perplexity": 11.5},
        {"generation": 2, "arc_easy_accuracy": 0.6, "val_perplexity": 12.0}
    ]
    analysis = analyzer.analyze_run(results)
    
    assert analysis["T_OOD"] == 2
    assert analysis["T_PPL"] == 1
    assert analysis["delta_t"] == -1 # 1 - 2
    assert analysis["regime_classification"] == "perplexity_first"

def test_synchronized_collapse(analyzer):
    """Test scenario where both fail in the same generation."""
    results = [
        {"generation": 0, "arc_easy_accuracy": 0.8, "val_perplexity": 10.0},
        {"generation": 1, "arc_easy_accuracy": 0.6, "val_perplexity": 12.0}
    ]
    analysis = analyzer.analyze_run(results)
    
    assert analysis["T_OOD"] == 1
    assert analysis["T_PPL"] == 1
    assert analysis["delta_t"] == 0
    assert analysis["regime_classification"] == "synchronized"

def test_no_collapse(analyzer):
    """Test scenario where no threshold is crossed."""
    results = [
        {"generation": 0, "arc_easy_accuracy": 0.8, "val_perplexity": 10.0},
        {"generation": 1, "arc_easy_accuracy": 0.79, "val_perplexity": 10.1},
        {"generation": 2, "arc_easy_accuracy": 0.78, "val_perplexity": 10.2}
    ]
    analysis = analyzer.analyze_run(results)
    
    assert analysis["T_OOD"] == -1
    assert analysis["T_PPL"] == -1
    assert analysis["delta_t"] is None
    assert analysis["regime_classification"] == "no_collapse"

def test_empty_run(analyzer):
    """Test graceful handling of empty data."""
    analysis = analyzer.analyze_run([])
    assert analysis == {}

def test_missing_metrics(analyzer):
    """Test handling of results with missing keys."""
    results = [{"generation": 0, "other": 1.0}]
    analysis = analyzer.analyze_run(results)
    assert analysis["status"] == "missing_metrics"
