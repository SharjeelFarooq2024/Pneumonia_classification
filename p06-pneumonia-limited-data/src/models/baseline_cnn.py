"""The required reference model: a small, randomly initialized CNN.

Architecture (configs/baseline.yaml):
  3 x [Conv2d(3x3, padding=1) -> ReLU -> MaxPool(2x2)] with 32, 64, 128 filters
  -> global average pooling -> Linear(128, 3) logits.

Input: grayscale X-rays are replicated to 3 channels by the shared data pipeline.
No pretrained weights, no augmentation, no class weighting, no label smoothing.
"""
from __future__ import annotations

import torch
from torch import nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 3, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)
