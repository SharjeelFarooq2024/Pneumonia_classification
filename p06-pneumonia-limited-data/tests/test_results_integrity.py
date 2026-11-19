"""Integrity tests for saved evaluation outputs, metrics, and figures."""
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

def test_test_sample_counts_and_metrics():
    # Test set must have exactly 624 samples: NORMAL 234, BACTERIA 242, VIRUS 148
    for exp in ["baseline", "proposed"]:
        for seed in [42, 43, 44]:
            metrics_path = ROOT / "results" / exp / f"seed{seed}" / "metrics_test.json"
            assert metrics_path.exists(), f"Missing {metrics_path}"
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            assert m["n_samples"] == 624
            assert m["support_per_class"]["NORMAL"] == 234
            assert m["support_per_class"]["BACTERIA"] == 242
            assert m["support_per_class"]["VIRUS"] == 148

def test_proposed_superiority_across_seeds():
    base_f1s = []
    prop_f1s = []
    for seed in [42, 43, 44]:
        with open(ROOT / "results" / "baseline" / f"seed{seed}" / "metrics_test.json") as f:
            base_f1s.append(json.load(f)["macro_f1"])
        with open(ROOT / "results" / "proposed" / f"seed{seed}" / "metrics_test.json") as f:
            prop_f1s.append(json.load(f)["macro_f1"])
    
    # Proposed must strictly outperform baseline for every seed
    for b, p in zip(base_f1s, prop_f1s):
        assert p > b, f"Proposed ({p}) must beat baseline ({b})"
    
    assert np.mean(prop_f1s) > np.mean(base_f1s) + 0.20, "Mean macro-F1 gain must exceed +0.20"
    assert np.std(prop_f1s) < np.std(base_f1s), "Proposed variance must be lower than baseline"

def test_prediction_probabilities_validity():
    for exp in ["baseline", "proposed"]:
        for seed in [42, 43, 44]:
            csv_path = ROOT / "results" / exp / f"seed{seed}" / "predictions_test.csv"
            df = pd.read_csv(csv_path)
            assert len(df) == 624
            
            # Check probability sum
            probs = df[["prob_NORMAL", "prob_BACTERIA", "prob_VIRUS"]].values
            sums = probs.sum(axis=1)
            assert np.allclose(sums, 1.0, atol=1e-4), "Probabilities must sum to 1.0"
            
            # Check predicted_label matches argmax
            argmax_preds = probs.argmax(axis=1)
            assert np.array_equal(argmax_preds, df["predicted_label"].values)

def test_paired_bootstrap_file():
    bootstrap_file = ROOT / "results" / "bootstrap_comparison.json"
    assert bootstrap_file.exists()
    with open(bootstrap_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["mean_difference_across_seeds"] > 0.24
    for seed_str, s_data in data["per_seed"].items():
        assert s_data["difference_ci95"][0] > 0, f"Lower CI for {seed_str} must be strictly positive"
