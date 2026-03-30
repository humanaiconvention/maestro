import yaml
import sys
import os

# Adjust Python path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from semantic_grounding.datasets.mixer import RegimeMixer

def run_smoke_test():
    config_path = "configs/multi_task_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return False

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    regimes = config.get("regimes", {})
    if not regimes:
        print("Error: No regimes found in config.")
        return False

    dummy_synthetic = [{"prompt": f"syn_{i}", "completion": f"val_{i}"} for i in range(10)]
    dummy_correction = [{"prompt": f"corr_{i}", "completion": f"val_{i}"} for i in range(10)]
    target_size = 10

    success = True
    for regime_id, regime_cfg in regimes.items():
        try:
            # RegimeMixer expects the regime config (synthetic_fraction, etc.)
            # and recursion config (data_policy)
            mixer_config = {**config.get("recursion", {}), **regime_cfg}
            mixer = RegimeMixer(mixer_config)
            
            # Derive unified correction fraction from all non-synthetic fields
            # (matches the formula in run_multi_task_benchmark.py)
            corr_frac = (
                regime_cfg.get("corrected_fraction", 0.0) +
                regime_cfg.get("frozen_real_fraction", 0.0) +
                regime_cfg.get("fresh_real_fraction", 0.0)
            )
            
            result = mixer.mix_dataset(
                synthetic_data=dummy_synthetic,
                correction_pool=dummy_correction,
                target_size=target_size,
                correction_fraction=corr_frac
            )
            
            n_corr = sum(1 for item in result if item["prompt"].startswith("corr_"))
            expected_n_corr = int(target_size * corr_frac)
            if len(result) != target_size:
                print(f"FAIL: {regime_id} - Expected size {target_size}, got {len(result)}")
                success = False
            elif n_corr != expected_n_corr:
                print(f"FAIL: {regime_id} - Expected {expected_n_corr} correction samples, got {n_corr}")
                success = False
            else:
                print(f"PASS: {regime_id} (size={len(result)}, corr={n_corr}/{target_size})")
                
        except Exception as e:
            print(f"FAIL: {regime_id} - Exception: {e}")
            success = False

    if success:
        print("\nAll mixer smoke tests passed.")
        return test_make_report_cli()
    else:
        print("\nSome mixer smoke tests failed.")
        return False

def test_make_report_cli():
    """Exercise make_report end-to-end via subprocess."""
    import tempfile
    import shutil
    import subprocess
    import json

    print("\nStarting make_report CLI integration test...")
    
    # Create project-local tmp/smoke/ to avoid Windows ACL issues
    base_tmp = os.path.join(os.path.dirname(__file__), "..", "tmp", "smoke")
    os.makedirs(base_tmp, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=base_tmp)
    
    try:
        # 1. Create fake metrics for R1 and R4
        for regime in ["R1", "R4"]:
            regime_path = os.path.join(tmp_dir, regime)
            os.makedirs(regime_path)
            fake_metrics = [
                {"generation": 0, "grounded_arc_accuracy": 0.8, "grounded_arc_perplexity": 10.0, "fluency_wiki_perplexity": 12.0},
                {"generation": 1, "grounded_arc_accuracy": 0.5, "grounded_arc_perplexity": 12.0, "fluency_wiki_perplexity": 14.0}
            ]
            with open(os.path.join(regime_path, "metrics.json"), "w") as f:
                json.dump(fake_metrics, f)
        
        # 2. Call make_report.py via subprocess
        report_path = os.path.join(tmp_dir, "report.md")
        plot_dir = os.path.join(tmp_dir, "plots")
        
        cmd = [
            sys.executable, 
            os.path.join(os.path.dirname(__file__), "make_report.py"),
            "--results-dir", tmp_dir,
            "--output", report_path,
            "--plot-dir", plot_dir
        ]
        
        # Suppress output to keep smoke test clean, unless it fails
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FAIL: make_report CLI - Exit code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
        # 3. Assertions
        if not os.path.exists(report_path):
            print("FAIL: make_report CLI - report.md not created")
            return False
            
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        checks = [
            ("Recursive Grounding Benchmark Report" in content, "Header missing"),
            ("| R1 |" in content, "R1 row missing"),
            ("| R4 |" in content, "R4 row missing")
        ]
        
        for success, msg in checks:
            if not success:
                print(f"FAIL: make_report CLI - {msg}")
                return False
                
        print("PASS: make_report CLI")
        return True
        
    except Exception as e:
        print(f"FAIL: make_report CLI - Exception: {e}")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    if run_smoke_test():
        sys.exit(0)
    else:
        sys.exit(1)
