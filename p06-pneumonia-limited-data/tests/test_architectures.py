"""Unit tests for neural network architectures, layer shapes, parameter counts, and freezing logic."""
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.baseline_cnn import BaselineCNN
from src.models.proposed import build_model, set_trainable, trainable_summary

def test_baseline_cnn_parameter_count():
    model = BaselineCNN(input_channels=3, num_classes=3)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 93635, f"Expected 93,635 params, got {total_params}"

def test_baseline_cnn_forward_shapes():
    model = BaselineCNN(input_channels=3, num_classes=3)
    model.eval()
    
    # Test batch size 1 and batch size 4
    for b in [1, 4]:
        x = torch.randn(b, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (b, 3), f"Expected shape ({b}, 3), got {out.shape}"

def test_proposed_resnet18_parameter_count():
    cfg = {"model": {"name": "resnet18", "pretrained": False, "dropout": 0.2}}
    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 11178051, f"Expected 11,178,051 params, got {total_params}"

def test_proposed_resnet18_forward_shapes():
    cfg = {"model": {"name": "resnet18", "pretrained": False, "dropout": 0.2}}
    model = build_model(cfg)
    model.eval()
    
    for b in [1, 2]:
        x = torch.randn(b, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (b, 3), f"Expected shape ({b}, 3), got {out.shape}"

def test_staged_freezing_logic():
    cfg = {"model": {"name": "resnet18", "pretrained": False, "dropout": 0.2}}
    model = build_model(cfg)
    
    # Stage 1: head unfreeze only
    set_trainable(model, ["fc"])
    summary = trainable_summary(model)
    assert summary["trainable_parameters"] == (512 * 3 + 3), "Only fc layer weights + biases should be trainable"
    assert summary["total_parameters"] == 11178051
    
    # Stage 2: layer4 + fc unfreeze
    set_trainable(model, ["layer4", "fc"])
    summary_s2 = trainable_summary(model)
    assert summary_s2["trainable_parameters"] > summary["trainable_parameters"]
    
    # Stage 3: full unfreeze
    set_trainable(model, "all")
    summary_s3 = trainable_summary(model)
    assert summary_s3["trainable_parameters"] == 11178051
    assert summary_s3["trainable_parameters"] == summary_s3["total_parameters"]
