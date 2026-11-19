"""Model factory, staged unfreezing, and discriminative learning rates.

The primary contribution is staged fine-tuning of an ImageNet-pretrained
ResNet-18. EfficientNet-B0 is supported as an optional secondary backbone.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from torch import nn

from src.models.baseline_cnn import BaselineCNN


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class _ResNet18Pure(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.0):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if dropout > 0:
            self.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(512, num_classes))
        else:
            self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [_BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _build_resnet18(pretrained: bool, num_classes: int, dropout: float) -> nn.Module:
    try:
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
    except (ImportError, ModuleNotFoundError):
        return _ResNet18Pure(num_classes=num_classes, dropout=dropout)


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
