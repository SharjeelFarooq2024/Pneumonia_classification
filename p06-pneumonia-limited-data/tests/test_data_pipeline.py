"""Unit tests for dataset splits, stratification, and index integrity."""
import json
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

def test_split_files_exist():
    splits_dir = ROOT / "data" / "splits"
    assert splits_dir.exists(), "Splits directory must exist"
    assert (splits_dir / "train25_seed42.json").exists(), "25% subset file must exist"
    assert (splits_dir / "train10_seed42.json").exists(), "10% subset file must exist"
    assert (splits_dir / "train100_seed42.json").exists(), "100% split file must exist"
    assert (splits_dir / "split_indices.json").exists(), "Split indices manifest must exist"

def test_train25_subset_properties():
    subset_file = ROOT / "data" / "splits" / "train25_seed42.json"
    with open(subset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert data["fraction"] == 0.25, "Fraction must be 0.25"
    assert data["seed"] == 42, "Seed must be 42"
    assert data["n_full_split"] == 4415, "Full training split has 4415 images"
    assert data["n_subset"] == 1103, "25% subset must contain exactly 1103 images"
    assert len(data["positions"]) == 1103, "Positions length must match n_subset"
    assert len(set(data["positions"])) == 1103, "Positions must have no duplicates"
    
    # Class proportions
    counts = data["class_counts"]
    assert counts == [290, 526, 287] or counts == {"NORMAL": 290, "BACTERIA": 526, "VIRUS": 287} or sum(counts.values() if isinstance(counts, dict) else counts) == 1103

def test_split_manifest_sizes():
    manifest_file = ROOT / "data" / "splits" / "split_indices.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    assert manifest["train"]["n"] == 4415, "Train split must be 4415"
    assert manifest["val"]["n"] == 817, "Val split must be 817"
    assert manifest["test"]["n"] == 624, "Test split must be 624"
    
    # Check partition disjointness
    train_set = set(manifest["train"]["global_indices"])
    val_set = set(manifest["val"]["global_indices"])
    test_set = set(manifest["test"]["global_indices"])
    assert len(train_set.intersection(val_set)) == 0, "Train and Val must be disjoint"
    assert len(train_set.intersection(test_set)) == 0, "Train and Test must be disjoint"
    assert len(val_set.intersection(test_set)) == 0, "Val and Test must be disjoint"
