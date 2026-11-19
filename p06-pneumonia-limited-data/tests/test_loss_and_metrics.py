"""Unit tests for metric computations, confusion matrices, calibration, and bootstrap."""
import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import compute_metrics
from src.analysis import paired_bootstrap

def test_compute_metrics_perfect_predictions():
    # 3 classes, 30 samples, 10 per class, all perfectly predicted
    y_true = np.array([0]*10 + [1]*10 + [2]*10)
    probs = np.zeros((30, 3))
    for i, t in enumerate(y_true):
        probs[i, t] = 0.99
        # add tiny probabilities for others
        for c in range(3):
            if c != t:
                probs[i, c] = 0.005

    class_names = ["NORMAL", "BACTERIA", "VIRUS"]
    metrics = compute_metrics(y_true, probs, class_names=class_names, bootstrap_n=0)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_auroc"] == 1.0
    assert metrics["ece"] < 0.05

def test_compute_metrics_imbalanced_confusion():
    # Simulate the baseline viral-to-bacterial confusion
    y_true = np.array([0]*10 + [1]*10 + [2]*10)
    # Class 2 (virus) is predicted as 1 (bacteria) 80% of the time
    probs = np.zeros((30, 3))
    for i in range(10): # normal
        probs[i] = [0.8, 0.1, 0.1]
    for i in range(10, 20): # bacteria
        probs[i] = [0.1, 0.8, 0.1]
    for i in range(20, 28): # virus -> bacteria
        probs[i] = [0.1, 0.8, 0.1]
    for i in range(28, 30): # virus -> virus
        probs[i] = [0.1, 0.1, 0.8]

    class_names = ["NORMAL", "BACTERIA", "VIRUS"]
    metrics = compute_metrics(y_true, probs, class_names=class_names, bootstrap_n=0)
    
    assert metrics["accuracy"] == 22 / 30
    assert metrics["per_class"]["VIRUS"]["recall"] == 0.2
    assert metrics["per_class"]["BACTERIA"]["recall"] == 1.0
    assert metrics["macro_f1"] < 0.75

def test_paired_bootstrap_difference():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 3, size=100)
    
    # Baseline probs: poorly predicted
    probs_a = np.zeros((100, 3))
    for i, t in enumerate(y_true):
        if i < 40: # 40 errors
            probs_a[i, (t + 1) % 3] = 0.8
            probs_a[i, t] = 0.2
        else:
            probs_a[i, t] = 0.8
            probs_a[i, (t + 1) % 3] = 0.2
            
    # Proposed probs: superior (only 10 errors)
    probs_b = np.zeros((100, 3))
    for i, t in enumerate(y_true):
        if i < 10:
            probs_b[i, (t + 1) % 3] = 0.8
            probs_b[i, t] = 0.2
        else:
            probs_b[i, t] = 0.8
            probs_b[i, (t + 1) % 3] = 0.2

    stats = paired_bootstrap(y_true, probs_a, probs_b, n_boot=200, seed=42)
    assert stats["difference_mean"] > 0, "Proposed should have higher macro-F1 than baseline"
    assert stats["difference_ci95"][0] > 0, "Lower bound of difference should be positive"
    assert stats["probability_proposed_better"] == 1.0
