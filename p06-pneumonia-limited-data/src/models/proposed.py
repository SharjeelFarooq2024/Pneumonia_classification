"""Model factory, staged unfreezing, and discriminative learning rates.

The primary contribution is staged fine-tuning of an ImageNet-pretrained
ResNet-18. EfficientNet-B0 is supported as an optional secondary backbone.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from torch import nn

from src.models.baseline_cnn import BaselineCNN


def _build_resnet18(pretrained: bool, num_classes: int, dropout: float) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    head: nn.Module
    if dropout > 0:
        head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    else:
        head = nn.Linear(in_features, num_classes)
    model.fc = head
    return model


def _build_efficientnet_b0(pretrained: bool, num_classes: int, dropout: float) -> nn.Module:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    return model


def build_model(config: Mapping[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    name = str(model_cfg.get("name", "baseline_cnn")).lower()
    num_classes = int(config.get("data", {}).get("num_classes", 3))
    channels = int(model_cfg.get("input_channels", 3))
    dropout = float(model_cfg.get("dropout", 0.0))
    pretrained = bool(model_cfg.get("pretrained", False))

    if name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes, input_channels=channels)
    if name == "resnet18":
        return _build_resnet18(pretrained=pretrained, num_classes=num_classes, dropout=dropout)
    if name == "efficientnet_b0":
        return _build_efficientnet_b0(pretrained=pretrained, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown model name: {name}")


def _first_component(param_name: str) -> str:
    return param_name.split(".")[0]


def set_trainable(model: nn.Module, unfreeze: Any) -> None:
    """Freeze everything, then unfreeze the requested modules (or all)."""
    if unfreeze in ("all", "ALL", None) or unfreeze == ["all"]:
        for p in model.parameters():
            p.requires_grad = True
        return
    if isinstance(unfreeze, str):
        unfreeze = [unfreeze]
    requested = list(unfreeze or [])
    for p in model.parameters():
        p.requires_grad = False
    for name, module in model.named_modules():
        if name in requested or any(name.startswith(req + ".") for req in requested):
            for p in module.parameters(recurse=False):
                p.requires_grad = True
    # A requested top-level module (e.g. "fc" as a Sequential) covers its children.
    for req in requested:
        if req in dict(model.named_children()):
            for p in dict(model.named_children())[req].parameters():
                p.requires_grad = True


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    layer_lr_mult: Optional[Mapping[str, float]] = None,
    optimizer_name: str = "adam",
) -> torch.optim.Optimizer:
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters for this stage.")
    opt_name = optimizer_name.lower()
    if layer_lr_mult:
        groups: Dict[tuple, List[nn.Parameter]] = {}
        for name, p in trainable:
            key = _first_component(name)
            mult = float(layer_lr_mult.get(key, 1.0))
            groups.setdefault((mult, weight_decay), []).append(p)
        param_groups = [
            {"params": params, "lr": lr * mult, "weight_decay": wd}
            for (mult, wd), params in groups.items()
        ]
    else:
        param_groups = [{"params": [p for _, p in trainable], "lr": lr, "weight_decay": weight_decay}]

    if opt_name == "adam":
        return torch.optim.Adam(param_groups)
    if opt_name == "adamw":
        return torch.optim.AdamW(param_groups)
    if opt_name == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def build_scheduler(optimizer: torch.optim.Optimizer, name: str, epochs: int):
    name = (name or "none").lower()
    if name == "none" or epochs <= 0:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    raise ValueError(f"Unknown scheduler: {name}")


def trainable_summary(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("No Grad-CAM target layer defined for this model.")
