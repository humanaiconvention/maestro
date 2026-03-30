import pytest
import json
import os
import tempfile
import shutil
from unittest.mock import patch
from scripts.make_report import make_report


def _make_tmp():
    """Create a temp dir inside the project tmp/ folder to avoid Windows system-temp ACL issues."""
    base = os.path.join(os.path.dirname(__file__), "..", "tmp", "test-pipeline")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(dir=base)


def test_full_pipeline_dry_run():
    """
    Validate the full reporting pipeline independently of the ML stack.
    Simulates: metrics written -> make_report reads them -> report generated.
    """
    tmp = _make_tmp()
    try:
        # 1. Create fake metrics for regime R1
        regime_dir = os.path.join(tmp, "R1")
        os.makedirs(regime_dir)
        metrics_file = os.path.join(regime_dir, "metrics.json")

        # Plausible metric values:
        # Accuracy drops from 0.8 to 0.5 (meets 0.10 threshold)
        # Perplexity rises from 10.0 to 12.0 (meets 0.05 threshold)
        fake_metrics = [
            {
                "generation": 0,
                "grounded_arc_accuracy": 0.8,
                "grounded_arc_perplexity": 10.0,
                "fluency_wiki_perplexity": 12.0
            },
            {
                "generation": 1,
                "grounded_arc_accuracy": 0.75,
                "grounded_arc_perplexity": 10.5,
                "fluency_wiki_perplexity": 12.5
            },
            {
                "generation": 2,
                "grounded_arc_accuracy": 0.5,   # Significant drop
                "grounded_arc_perplexity": 12.0, # Significant rise
                "fluency_wiki_perplexity": 14.0
            }
        ]

        with open(metrics_file, "w") as f:
            json.dump(fake_metrics, f)

        output_report = os.path.join(tmp, "final_report.md")
        plot_dir = os.path.join(tmp, "plots")

        # 2. Patch ResultsPlotter to be a no-op
        with patch("scripts.make_report.ResultsPlotter.plot_regime_trajectories") as mock_plot:
            # 3. Call make_report
            make_report(
                results_dir=tmp,
                output_file=output_report,
                plot_dir=plot_dir,
                task_prefix="grounded_arc"
            )

            # Verify plotter was called once
            mock_plot.assert_called_once()

        # 4. Assertions on the output markdown
        assert os.path.exists(output_report)

        with open(output_report, "r", encoding="utf-8") as f:
            content = f.read()

            # Structure assertions
            assert "Recursive Grounding Benchmark Report" in content
            assert "R1" in content
            # The table uses raw generation numbers like | 0 |
            assert "| 0 |" in content

            # 5. Parse regime analysis section
            # Look for the table row for R1
            lines = content.splitlines()
            r1_line = [l for l in lines if "| R1 |" in l]
            assert len(r1_line) == 1

            # Row format: | Regime | Signature Detected? | Failure Ordering | Acc Drop Gen | PPL Rise Gen |
            columns = [c.strip() for c in r1_line[0].split("|")]

            # Filter out empty strings from the ends
            columns = [c for c in columns if c]

            # columns: ['R1', '❌ NO', 'perplexity_failed_first', 'Gen 2', 'Gen 1']
            assert len(columns) >= 3

            # Verify that failure ordering is populated (not "N/A" or empty)
            ordering = columns[2]
            assert ordering != "N/A"
            assert len(ordering) > 0

            # Verify signature detected is not just N/A placeholder
            sig = columns[1]
            assert "YES" in sig or "NO" in sig

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_report_signature_detected_yes():
    """(1) Test regime R2 where accuracy drops before perplexity rises (silent semantic drift)."""
    tmp = _make_tmp()
    try:
        regime_dir = os.path.join(tmp, "R2")
        os.makedirs(regime_dir)
        metrics_file = os.path.join(regime_dir, "metrics.json")

        # Signature Detected (YES): Grounding/accuracy fails BEFORE perplexity rises.
        # gen0: base (acc=0.8, ppl=10.0)
        # gen1: acc=0.6 (25% drop > 10% threshold), ppl=10.4 (4% rise < 5% threshold) → acc fails first
        # gen2: acc=0.5, ppl=11.5 (15% rise > 5% threshold) → ppl fails later
        fake_metrics = [
            {"generation": 0, "grounded_arc_accuracy": 0.8, "grounded_arc_perplexity": 10.0, "fluency_wiki_perplexity": 12.0},
            {"generation": 1, "grounded_arc_accuracy": 0.6, "grounded_arc_perplexity": 10.4, "fluency_wiki_perplexity": 12.5},
            {"generation": 2, "grounded_arc_accuracy": 0.5, "grounded_arc_perplexity": 11.5, "fluency_wiki_perplexity": 14.0}
        ]
        with open(metrics_file, "w") as f:
            json.dump(fake_metrics, f)

        output_report = os.path.join(tmp, "final_report.md")
        with patch("scripts.make_report.ResultsPlotter"):
            make_report(results_dir=tmp, output_file=output_report, plot_dir=os.path.join(tmp, "plots"))

        with open(output_report, "r", encoding="utf-8") as f:
            content = f.read()
            # Find the R2 row in the Regime Analysis table
            r2_line = [l for l in content.splitlines() if "| R2 |" in l][0]
            assert "YES" in r2_line
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_report_multi_regime():
    """(2) Test report with two different regimes."""
    tmp = _make_tmp()
    try:
        # R1: Acc fails first, PPL barely moves → grounding_failed_only → signature YES
        # gen0: acc=0.8, ppl=10.0; gen1: acc drops 37.5% (>10% threshold), ppl rises 1% (<5% threshold)
        r1_dir = os.path.join(tmp, "R1")
        os.makedirs(r1_dir)
        with open(os.path.join(r1_dir, "metrics.json"), "w") as f:
            json.dump([
                {"generation": 0, "grounded_arc_accuracy": 0.8, "grounded_arc_perplexity": 10.0},
                {"generation": 1, "grounded_arc_accuracy": 0.5, "grounded_arc_perplexity": 10.1}
            ], f)

        # R2: Only PPL rises, acc stays stable → perplexity_failed_only → signature NO
        # gen0: acc=0.8, ppl=10.0; gen1: ppl rises 20% (>5% threshold), acc drops 1.25% (<10% threshold)
        r2_dir = os.path.join(tmp, "R2")
        os.makedirs(r2_dir)
        with open(os.path.join(r2_dir, "metrics.json"), "w") as f:
            json.dump([
                {"generation": 0, "grounded_arc_accuracy": 0.8, "grounded_arc_perplexity": 10.0},
                {"generation": 1, "grounded_arc_accuracy": 0.79, "grounded_arc_perplexity": 12.0}
            ], f)

        output_report = os.path.join(tmp, "final_report.md")
        with patch("scripts.make_report.ResultsPlotter"):
            make_report(results_dir=tmp, output_file=output_report, plot_dir=os.path.join(tmp, "plots"))

        with open(output_report, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.splitlines()
            r1_line = [l for l in lines if "| R1 |" in l][0]
            r2_line = [l for l in lines if "| R2 |" in l][0]

            # R1: grounding_failed_only → ✅ YES (silent semantic drift detected)
            assert "| R1 |" in r1_line and "YES" in r1_line
            # R2: perplexity_failed_only → ❌ NO (perplexity warned first, not the silent signature)
            assert "| R2 |" in r2_line and "NO" in r2_line
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
