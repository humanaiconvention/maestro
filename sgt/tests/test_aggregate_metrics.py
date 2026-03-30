import pytest
import os
import shutil
import tempfile
from scripts.aggregate_metrics import find_metrics_files, parse_run_metadata

def _make_tmp():
    """Create a temp dir inside the project tmp/ folder to avoid Windows system-temp ACL issues."""
    base = os.path.join(os.path.dirname(__file__), "..", "tmp", "test-agg")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(dir=base)

def test_find_metrics_files():
    """Test finding metrics.jsonl files in a nested tree."""
    tmp = _make_tmp()
    try:
        # Create nested structure
        # dir1/metrics.jsonl
        # dir1/subdir/metrics.jsonl
        # dir2/metrics.jsonl
        # dir2/not_metrics.txt
        
        d1 = os.path.join(tmp, "dir1")
        d1s = os.path.join(d1, "subdir")
        d2 = os.path.join(tmp, "dir2")
        os.makedirs(d1s)
        os.makedirs(d2)
        
        with open(os.path.join(d1, "metrics.jsonl"), "w") as f: f.write("{}")
        with open(os.path.join(d1s, "metrics.jsonl"), "w") as f: f.write("{}")
        with open(os.path.join(d2, "metrics.jsonl"), "w") as f: f.write("{}")
        with open(os.path.join(d2, "not_metrics.txt"), "w") as f: f.write("{}")
        
        files = find_metrics_files(tmp)
        assert len(files) == 3
        
        # Test empty
        empty_tmp = _make_tmp()
        try:
            assert find_metrics_files(empty_tmp) == []
        finally:
            shutil.rmtree(empty_tmp, ignore_errors=True)
            
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_parse_run_metadata_path_parsing():
    """Test extracting metadata from path string."""
    fpath = "results/sweeps/exp/gpt2/c_0.5/seed_3/metrics.jsonl"
    first_line = {} # Not used if path parsing works
    
    meta = parse_run_metadata(fpath, first_line)
    
    assert meta["model"] == "gpt2"
    assert meta["correction_fraction"] == 0.5
    assert meta["seed"] == 3
    assert meta["path"] == fpath

def test_parse_run_metadata_fallback():
    """Test fallback to internal metadata when path is short or invalid."""
    # Path too short (fewer than 4 parts before metrics.jsonl)
    fpath = "short/metrics.jsonl"
    first_line = {
        "model": "llama",
        "correction_fraction": 0.25,
        "seed": 7
    }
    
    # This should trigger fallback and log a warning
    meta = parse_run_metadata(fpath, first_line)
    
    assert meta["model"] == "llama"
    assert meta["correction_fraction"] == 0.25
    assert meta["seed"] == 7
    
    # Path exists but values are non-numeric
    invalid_path = "results/sweeps/exp/model/c_INVALID/seed_STALE/metrics.jsonl"
    meta_invalid = parse_run_metadata(invalid_path, first_line)
    assert meta_invalid["model"] == "llama" # Fallback triggered
