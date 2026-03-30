import pytest
from semantic_grounding.datasets.mixer import RegimeMixer

@pytest.fixture
def synthetic_data():
    return [{"prompt": f"syn_{i}", "completion": f"val_{i}"} for i in range(10)]

@pytest.fixture
def correction_pool():
    return [{"prompt": f"corr_{i}", "completion": f"val_{i}"} for i in range(10)]

def test_r1_pure_synthetic(synthetic_data, correction_pool):
    mixer = RegimeMixer(config={"data_policy": "replace"})
    target_size = 10
    result = mixer.mix_dataset(synthetic_data, correction_pool, target_size, correction_fraction=0.0)
    
    assert len(result) == target_size
    for item in result:
        assert item["prompt"].startswith("syn_")

def test_r4_80pct_correction(synthetic_data, correction_pool):
    mixer = RegimeMixer(config={"data_policy": "replace"})
    target_size = 10
    result = mixer.mix_dataset(synthetic_data, correction_pool, target_size, correction_fraction=0.8)
    
    assert len(result) == target_size
    corr_count = sum(1 for item in result if item["prompt"].startswith("corr_"))
    # n_corr = int(10 * 0.8) = 8
    assert corr_count == 8

def test_r2_50pct_frozen(synthetic_data, correction_pool):
    mixer = RegimeMixer(config={"data_policy": "replace"})
    target_size = 10
    result = mixer.mix_dataset(synthetic_data, correction_pool, target_size, correction_fraction=0.5)
    
    assert len(result) == target_size
    corr_count = sum(1 for item in result if item["prompt"].startswith("corr_"))
    syn_count = sum(1 for item in result if item["prompt"].startswith("syn_"))
    assert corr_count == 5
    assert syn_count == 5

def test_empty_correction_pool(synthetic_data):
    mixer = RegimeMixer(config={"data_policy": "replace"})
    target_size = 10
    # When correction_pool is empty, all target_size slots fall back to synthetic.
    # n_corr_actual=0 (no correction pool), n_syn_required=target_size.
    result = mixer.mix_dataset(synthetic_data, correction_pool=[], target_size=target_size, correction_fraction=0.5)

    assert len(result) == target_size
    for item in result:
        assert item["prompt"].startswith("syn_")

def test_accumulate_policy(correction_pool):
    mixer = RegimeMixer(config={"data_policy": "accumulate"})
    
    syn_batch_1 = [{"prompt": "syn_1", "completion": "v1"}]
    mixer.mix_dataset(syn_batch_1, correction_pool, target_size=1, correction_fraction=0.0)
    assert len(mixer.accumulated_history) == 1
    
    syn_batch_2 = [{"prompt": "syn_2", "completion": "v2"}]
    mixer.mix_dataset(syn_batch_2, correction_pool, target_size=1, correction_fraction=0.0)
    assert len(mixer.accumulated_history) == 2
    
    prompts = [item["prompt"] for item in mixer.accumulated_history]
    assert "syn_1" in prompts
    assert "syn_2" in prompts

def test_target_size_respected(synthetic_data, correction_pool):
    mixer = RegimeMixer(config={"data_policy": "replace"})
    for size in [5, 20, 100]:
        result = mixer.mix_dataset(synthetic_data, correction_pool, target_size=size, correction_fraction=0.3)
        assert len(result) == size
