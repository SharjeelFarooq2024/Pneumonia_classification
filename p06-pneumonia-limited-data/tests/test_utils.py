"""Tests for shared utilities in src/utils.py."""
import numpy as np
import pytest
import torch
from pathlib import Path

from src.utils import (
    deep_merge,
    set_by_dotted_key,
    expected_calibration_error,
    aggregate_seed_metrics,
    to_jsonable,
    set_seed,
    count_parameters,
    probe_batch_size,
    setup_logger,
)
from src.models.baseline_cnn import BaselineCNN


def test_deep_merge():
    base = {"a": 1, "b": {"x": 10, "y": 20}, "c": [1, 2]}
    override = {"b": {"y": 99, "z": 100}, "d": "new"}
    merged = deep_merge(base, override)

    assert merged["a"] == 1
    assert merged["b"]["x"] == 10
    assert merged["b"]["y"] == 99
    assert merged["b"]["z"] == 100
    assert merged["c"] == [1, 2]
    assert merged["d"] == "new"
    # Ensure original base was not mutated
    assert base["b"]["y"] == 20


def test_set_by_dotted_key():
    cfg = {"model": {"name": "cnn"}}
    set_by_dotted_key(cfg, "model.lr", 0.001)
    set_by_dotted_key(cfg, "training.epochs", 20)

    assert cfg["model"]["lr"] == 0.001
    assert cfg["training"]["epochs"] == 20


def test_expected_calibration_error():
    # Perfectly calibrated case
    # If confidence matches accuracy, ECE should be 0
    y_true = np.array([0, 1, 0, 1])
    # Class 0: [1.0, 0.0], Class 1: [0.0, 1.0]
    y_prob = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    ece, rows = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-5)
    assert len(rows) > 0

    # Overconfident completely wrong case
    y_wrong = np.array([1, 0, 1, 0])
    ece_wrong, _ = expected_calibration_error(y_wrong, y_prob, n_bins=10)
    assert ece_wrong == pytest.approx(1.0, abs=1e-5)


def test_aggregate_seed_metrics():
    seed_records = [
        {"macro_f1": 0.80, "accuracy": 0.85},
        {"macro_f1": 0.82, "accuracy": 0.87},
        {"macro_f1": 0.84, "accuracy": 0.89},
    ]
    agg = aggregate_seed_metrics(seed_records, ["macro_f1", "accuracy"])
    assert agg["macro_f1"]["mean"] == pytest.approx(0.82)
    assert agg["macro_f1"]["n"] == 3
    assert agg["accuracy"]["mean"] == pytest.approx(0.87)


def test_to_jsonable():
    data = {
        "int": np.int64(42),
        "float": np.float32(3.14),
        "bool": np.bool_(True),
        "array": np.array([1, 2, 3]),
        "path": Path("results/summary.json"),
    }
    jsonable = to_jsonable(data)
    assert isinstance(jsonable["int"], int)
    assert isinstance(jsonable["float"], float)
    assert isinstance(jsonable["bool"], bool)
    assert isinstance(jsonable["array"], list)
    assert isinstance(jsonable["path"], str)


def test_set_seed_reproducibility():
    set_seed(12345)
    rand1 = torch.randn(10).numpy()
    np_rand1 = np.random.rand(10)

    set_seed(12345)
    rand2 = torch.randn(10).numpy()
    np_rand2 = np.random.rand(10)

    np.testing.assert_allclose(rand1, rand2)
    np.testing.assert_allclose(np_rand1, np_rand2)


def test_count_parameters():
    model = BaselineCNN(num_classes=3)
    count = count_parameters(model)
    assert count == 93635


def test_probe_batch_size_cpu():
    model = BaselineCNN(num_classes=3)
    logger = setup_logger("test_logger")
    cfg = {"training": {"batch_size": 32, "batch_size_fallbacks": [16, 8]}}
    device = torch.device("cpu")
    bs = probe_batch_size(model, cfg, device, logger)
    assert bs == 32
