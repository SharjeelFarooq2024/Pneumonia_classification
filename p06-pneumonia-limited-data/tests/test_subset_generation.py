"""Tests for subset generation logic in src/data/subset.py."""
import numpy as np
import pytest

from src.data.subset import stratified_positions, _class_summary


def test_stratified_positions_invalid():
    with pytest.raises(ValueError, match="empty split"):
        stratified_positions(np.array([]), 0.25, 42)

    with pytest.raises(ValueError, match="fraction must be in"):
        stratified_positions(np.array([0, 1, 2]), 0.0, 42)

    with pytest.raises(ValueError, match="fraction must be in"):
        stratified_positions(np.array([0, 1, 2]), 1.5, 42)


def test_stratified_positions_fraction_one():
    labels = np.array([0, 1, 2, 0, 1, 2])
    pos = stratified_positions(labels, 1.0, 42)
    np.testing.assert_array_equal(pos, np.arange(len(labels)))


def test_stratified_positions_preserves_distribution():
    # 1000 samples, 3 classes with 20%, 50%, 30% distribution
    n = 1000
    labels = np.array([0] * 200 + [1] * 500 + [2] * 300)
    pos = stratified_positions(labels, 0.25, seed=42)

    assert len(pos) == 250
    sub_labels = labels[pos]

    c0 = np.sum(sub_labels == 0)
    c1 = np.sum(sub_labels == 1)
    c2 = np.sum(sub_labels == 2)

    assert c0 == 50   # 20% of 250
    assert c1 == 125  # 50% of 250
    assert c2 == 75   # 30% of 250


def test_class_summary_utility():
    labels = np.array([0, 0, 1, 2, 2, 2])
    class_names = ["NORMAL", "BACTERIA", "VIRUS"]
    counts, props = _class_summary(labels, class_names)

    assert counts["NORMAL"] == 2
    assert counts["BACTERIA"] == 1
    assert counts["VIRUS"] == 3
    assert props["NORMAL"] == pytest.approx(2 / 6)
    assert props["BACTERIA"] == pytest.approx(1 / 6)
    assert props["VIRUS"] == pytest.approx(3 / 6)
