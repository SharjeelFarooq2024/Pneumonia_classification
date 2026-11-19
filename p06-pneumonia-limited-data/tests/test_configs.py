"""Unit tests for YAML configuration schemas and inheritance."""
import sys
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config

def test_base_config_loading():
    base_file = ROOT / "configs" / "base.yaml"
    assert base_file.exists()
    cfg = load_config(base_file)
    assert cfg["project_name"] == "p06-pneumonia-limited-data"
    assert cfg["data"]["num_classes"] == 3
    assert cfg["data"]["class_names"] == ["NORMAL", "BACTERIA", "VIRUS"]

def test_baseline_config_loading():
    cfg = load_config(ROOT / "configs" / "baseline.yaml")
    assert cfg["experiment"] == "baseline"
    assert cfg["model"]["name"] == "baseline_cnn"
    assert cfg["model"]["pretrained"] is False
    assert cfg["training"]["epochs"] == 20
    assert cfg["training"]["augment"] is False
    assert cfg["training"]["class_weights_from_subset"] is False

def test_proposed_config_loading():
    cfg = load_config(ROOT / "configs" / "proposed.yaml")
    assert cfg["experiment"] == "proposed"
    assert cfg["model"]["name"] == "resnet18"
    assert cfg["model"]["pretrained"] is True
    assert cfg["training"]["augment"] is True
    assert cfg["training"]["label_smoothing"] == 0.1
    assert cfg["training"]["class_weights_from_subset"] is True
    assert len(cfg["training"]["stages"]) == 3

def test_all_ablation_configs_loadable():
    ablations_dir = ROOT / "configs" / "ablations"
    ablation_files = list(ablations_dir.glob("*.yaml"))
    assert len(ablation_files) >= 10, "Should have at least 10 ablation configs"
    for f in ablation_files:
        cfg = load_config(f)
        assert "experiment" in cfg, f"Missing 'experiment' key in {f.name}"
        assert "model" in cfg, f"Missing 'model' key in {f.name}"
        assert "training" in cfg, f"Missing 'training' key in {f.name}"
