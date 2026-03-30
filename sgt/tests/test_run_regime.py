import pytest
import os
import json
import tempfile
import shutil
import yaml
from unittest.mock import patch, MagicMock, mock_open
from scripts.run_multi_task_benchmark import run_regime
from scripts.path_utils import project_results_path

def _make_tmp():
    """Create a temp dir inside the project tmp/ folder to avoid Windows system-temp ACL issues."""
    base = os.path.join(os.path.dirname(__file__), "..", "tmp", "test-run-regime")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(dir=base)

@pytest.fixture
def mock_config_path():
    tmp = _make_tmp()
    config_path = os.path.join(tmp, "multi_task_config.yaml")
    config_data = {
        "model": {"base_model": "test-model"},
        "recursion": {"max_generations": 1},
        "datasets": {"train_family": "synthetic"},
        "regimes": {
            "R1": {
                "name": "synthetic_only",
                "synthetic_fraction": 1.0,
                "corrected_fraction": 0.0,
                "data_policy": "replace"
            },
            "R4": {
                "name": "synthetic_plus_80pct_correction",
                "synthetic_fraction": 0.2,
                "corrected_fraction": 0.8,
                "data_policy": "replace"
            }
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    yield config_path
    shutil.rmtree(tmp, ignore_errors=True)

def test_run_regime_missing_key(mock_config_path):
    """(1) Missing regime key raises ValueError."""
    with pytest.raises(ValueError, match="Regime NONEXISTENT not found"):
        run_regime(mock_config_path, "NONEXISTENT")

@patch("scripts.run_multi_task_benchmark.RegimeMixer")
@patch("scripts.run_multi_task_benchmark.RecursiveTrainer")
@patch("scripts.run_multi_task_benchmark.load_task_families")
@patch("scripts.run_multi_task_benchmark.evaluate_all_families")
def test_run_regime_r1_config(mock_eval, mock_load, mock_trainer_cls, mock_mixer_cls, mock_config_path):
    """(2) R1 config read -> mix_dataset called with correction_fraction=0.0."""
    mock_load.return_value = {
        "train": [{"prompt": "p1", "completion": "c1"}],
        "eval_families": {}
    }
    mock_trainer = mock_trainer_cls.return_value
    mock_trainer.generate_synthetic_data.return_value = [{"prompt": "p1", "completion": "s1"}]
    
    mock_mixer = mock_mixer_cls.return_value
    
    run_regime(mock_config_path, "R1")
    
    # Verify mix_dataset was called with 0.0 correction
    # Generation 0 doesn't call mixer, Generation 1 does.
    mock_mixer.mix_dataset.assert_called_once()
    args, kwargs = mock_mixer.mix_dataset.call_args
    assert kwargs["correction_fraction"] == 0.0

@patch("scripts.run_multi_task_benchmark.RegimeMixer")
@patch("scripts.run_multi_task_benchmark.RecursiveTrainer")
@patch("scripts.run_multi_task_benchmark.load_task_families")
@patch("scripts.run_multi_task_benchmark.evaluate_all_families")
def test_run_regime_r4_config(mock_eval, mock_load, mock_trainer_cls, mock_mixer_cls, mock_config_path):
    """(3) R4 config read -> mix_dataset called with correction_fraction=0.8."""
    mock_load.return_value = {
        "train": [{"prompt": "p1", "completion": "c1"}],
        "eval_families": {}
    }
    mock_trainer = mock_trainer_cls.return_value
    mock_trainer.generate_synthetic_data.return_value = [{"prompt": "p1", "completion": "s1"}]
    
    mock_mixer = mock_mixer_cls.return_value
    
    run_regime(mock_config_path, "R4")
    
    mock_mixer.mix_dataset.assert_called_once()
    args, kwargs = mock_mixer.mix_dataset.call_args
    assert kwargs["correction_fraction"] == 0.8

@patch("scripts.run_multi_task_benchmark.RecursiveTrainer")
@patch("scripts.run_multi_task_benchmark.load_task_families")
@patch("scripts.run_multi_task_benchmark.evaluate_all_families")
@patch("scripts.run_multi_task_benchmark.RegimeMixer")
@patch("os.makedirs")
@patch("builtins.open")
def test_run_regime_output_created(mock_open_func, mock_dirs, mock_mixer, mock_eval, mock_load, mock_trainer, mock_config_path):
    """(4) Output directory created and metrics.json written."""
    mock_load.return_value = {
        "train": [{"prompt": "p1", "completion": "c1"}],
        "eval_families": {}
    }
    
    # We need to handle both reading the config and writing the metrics
    # Create mock handles for read and write
    config_content = """
model: {base_model: test-model}
recursion: {max_generations: 1}
datasets: {train_family: synthetic}
regimes:
  R1: {name: synthetic_only, synthetic_fraction: 1.0, corrected_fraction: 0.0, data_policy: replace}
"""
    mock_read_handle = mock_open(read_data=config_content).return_value
    mock_write_handle = mock_open().return_value
    
    # Define a side effect for open() to return different handles based on mode
    def open_side_effect(path, mode="r", *args, **kwargs):
        if "r" in mode:
            return mock_read_handle
        else:
            return mock_write_handle
            
    mock_open_func.side_effect = open_side_effect
    
    run_regime(mock_config_path, "R1")
    
    # Check if metrics.json was written to the expected path
    expected_path = project_results_path("multi_task", "R1", "metrics.json").replace("\\", "/")
    
    # Verify open was called with the expected path for writing
    any_metrics_write = any(
        call[0][0].replace("\\", "/") == expected_path and "w" in call[0][1]
        for call in mock_open_func.call_args_list if len(call[0]) > 1
    )
    assert any_metrics_write
