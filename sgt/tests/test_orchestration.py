import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from scripts.make_report import make_report
from scripts.run_all_regimes import main as run_all_main


def _make_tmp():
    """Create a temp dir inside the project tmp/ folder to avoid Windows system-temp ACL issues."""
    base = os.path.join(os.path.dirname(__file__), "..", "tmp", "test-orch")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(dir=base)


def test_make_report_no_results_dir(capsys):
    """Test make_report with non-existent directory."""
    tmp = _make_tmp()
    try:
        non_existent = os.path.join(tmp, "ghost")
        make_report(non_existent, os.path.join(tmp, "report.md"), os.path.join(tmp, "plots"))
        captured = capsys.readouterr()
        assert "No metrics found" in captured.out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_make_report_with_synthetic_metrics():
    """Test make_report with fake metrics.json."""
    tmp = _make_tmp()
    try:
        regime_dir = os.path.join(tmp, "R1")
        os.makedirs(regime_dir)
        metrics_file = os.path.join(regime_dir, "metrics.json")

        fake_metrics = [
            {
                "generation": i,
                "grounded_arc_accuracy": 0.8 - (i * 0.1),
                "grounded_arc_perplexity": 10.0 + i,
                "fluency_wiki_perplexity": 12.0 + (i * 0.5),
            }
            for i in range(3)
        ]
        with open(metrics_file, "w") as f:
            json.dump(fake_metrics, f)

        report_file = os.path.join(tmp, "final_report.md")
        plot_dir = os.path.join(tmp, "plots")

        with patch("scripts.make_report.ResultsPlotter"):
            make_report(tmp, report_file, plot_dir)

        assert os.path.exists(report_file)
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Recursive Grounding Benchmark Report" in content
        assert "R1" in content
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_make_report_plotter_receives_raw_metrics():
    """Verify that the plotter receives raw metrics (not normalized)."""
    tmp = _make_tmp()
    try:
        regime_dir = os.path.join(tmp, "R1")
        os.makedirs(regime_dir)
        metrics_file = os.path.join(regime_dir, "metrics.json")

        raw_key = "grounded_arc_accuracy"
        fake_metrics = [{"generation": 0, raw_key: 0.8}]
        with open(metrics_file, "w") as f:
            json.dump(fake_metrics, f)

        with patch("scripts.make_report.ResultsPlotter.plot_regime_trajectories") as mock_plot:
            make_report(tmp, os.path.join(tmp, "report.md"), os.path.join(tmp, "plots"))
            called_args = mock_plot.call_args[0][0]
            assert "R1" in called_args
            assert raw_key in called_args["R1"][0]
            assert "accuracy" not in called_args["R1"][0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_all_regimes_invalid_regime(capsys):
    """Test run_all_regimes with an invalid regime name."""
    tmp = _make_tmp()
    try:
        config_file = os.path.join(tmp, "config.yaml")
        with open(config_file, "w") as f:
            f.write("regimes:\n  R1:\n    name: test")

        with patch("sys.argv", ["run_all_regimes.py", "--config", config_file, "--regimes", "INVALID"]):
            with pytest.raises(SystemExit) as excinfo:
                run_all_main()
            assert excinfo.value.code == 1

        captured = capsys.readouterr()
        assert "Error: Invalid regimes specified" in captured.out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_all_regimes_missing_config():
    """Test run_all_regimes with a non-existent config file."""
    with patch("sys.argv", ["run_all_regimes.py", "--config", "nonexistent.yaml"]):
        with pytest.raises(SystemExit) as excinfo:
            run_all_main()
        assert excinfo.value.code == 1


def test_run_all_regimes_dry_run():
    """Test run_all_regimes with a minimal config and mocked components."""
    tmp = _make_tmp()
    try:
        config_file = os.path.join(tmp, "config.yaml")
        with open(config_file, "w") as f:
            f.write("regimes:\n  R1:\n    synthetic_fraction: 1.0\n    corrected_fraction: 0.0\n    data_policy: replace")

        with patch("sys.argv", ["run_all_regimes.py", "--config", config_file]):
            with patch("scripts.run_all_regimes.benchmark.run_regime") as mock_run:
                with patch("scripts.run_all_regimes.reporter.make_report") as mock_report:
                    with pytest.raises(SystemExit) as excinfo:
                        run_all_main()
                    assert excinfo.value.code == 0
                    mock_run.assert_called_once()
                    mock_report.assert_called_once()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
