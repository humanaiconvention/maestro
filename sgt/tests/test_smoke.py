"""Smoke tests: exercise mixer + real config integration as a pytest suite.

These tests load the actual configs/multi_task_config.yaml and verify that
RegimeMixer produces the correct output size and correction ratio for every
defined regime.  They catch config/code mismatches that unit tests with
synthetic configs would miss.
"""

import os
import yaml
import pytest
from semantic_grounding.datasets.mixer import RegimeMixer

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "multi_task_config.yaml"
)


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def dummy_data():
    synthetic = [{"prompt": f"syn_{i}", "completion": f"val_{i}"} for i in range(20)]
    correction = [{"prompt": f"corr_{i}", "completion": f"val_{i}"} for i in range(20)]
    return synthetic, correction


def _corr_frac(regime_cfg):
    """Unified correction fraction matching run_multi_task_benchmark.py formula."""
    return (
        regime_cfg.get("corrected_fraction", 0.0)
        + regime_cfg.get("frozen_real_fraction", 0.0)
        + regime_cfg.get("fresh_real_fraction", 0.0)
    )


# ---- Parametrized over every regime in the real config ----


def _regime_ids(config_path):
    """Read regime IDs from config at import time for parametrize."""
    if not os.path.exists(config_path):
        return []
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get("regimes", {}).keys())


@pytest.mark.parametrize("regime_id", _regime_ids(CONFIG_PATH))
def test_mixer_output_size(config, dummy_data, regime_id):
    """mix_dataset returns exactly target_size items for every regime."""
    syn, corr = dummy_data
    regime_cfg = config["regimes"][regime_id]
    mixer_cfg = {**config.get("recursion", {}), **regime_cfg}
    mixer = RegimeMixer(mixer_cfg)
    target = 10

    result = mixer.mix_dataset(
        synthetic_data=syn,
        correction_pool=corr,
        target_size=target,
        correction_fraction=_corr_frac(regime_cfg),
    )
    assert len(result) == target, f"{regime_id}: expected {target}, got {len(result)}"


@pytest.mark.parametrize("regime_id", _regime_ids(CONFIG_PATH))
def test_mixer_correction_ratio(config, dummy_data, regime_id):
    """Correction count matches expected fraction for every regime."""
    syn, corr = dummy_data
    regime_cfg = config["regimes"][regime_id]
    mixer_cfg = {**config.get("recursion", {}), **regime_cfg}
    mixer = RegimeMixer(mixer_cfg)
    target = 10
    cf = _corr_frac(regime_cfg)

    result = mixer.mix_dataset(
        synthetic_data=syn,
        correction_pool=corr,
        target_size=target,
        correction_fraction=cf,
    )
    n_corr = sum(1 for item in result if item["prompt"].startswith("corr_"))
    expected = int(target * cf)
    assert n_corr == expected, (
        f"{regime_id}: expected {expected} correction samples, got {n_corr}"
    )


def test_config_regime_fractions_sum_to_one(config):
    """Every regime's synthetic + correction fractions sum to 1.0."""
    for regime_id, regime_cfg in config.get("regimes", {}).items():
        syn = regime_cfg.get("synthetic_fraction", 0.0)
        total = syn + _corr_frac(regime_cfg)
        assert abs(total - 1.0) < 1e-6, (
            f"{regime_id}: fractions sum to {total}, expected 1.0"
        )


def test_config_has_required_top_level_keys(config):
    """Config YAML contains all keys the pipeline expects."""
    for key in ("model", "recursion", "training", "datasets", "regimes"):
        assert key in config, f"Missing top-level key: {key}"


def test_config_model_dtype_valid(config):
    """model.dtype is one of the supported values."""
    dtype = config["model"].get("dtype", "float16")
    assert dtype in ("float16", "bfloat16", "float32"), f"Invalid dtype: {dtype}"
