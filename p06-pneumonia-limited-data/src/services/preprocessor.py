"""Preprocessing services implementing the Strategy Pattern for image normalization."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import torch
from PIL import Image

from src.domain.interfaces import IPreprocessor


class INormalizationStrategy(ABC):
    """Strategy interface for channel-wise tensor normalization."""

    @abstractmethod
    def normalize(self, tensor: np.ndarray) -> np.ndarray:
        """Normalizes an image array of shape (C, H, W) in range [0, 1]."""
        pass


class ImageNetNormalizationStrategy(INormalizationStrategy):
    """Standard ImageNet mean/std normalization for transfer learning models."""

    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def normalize(self, tensor: np.ndarray) -> np.ndarray:
        return (tensor - self.mean) / self.std


class SymmetricNormalizationStrategy(INormalizationStrategy):
    """Symmetric [-1, 1] normalization with mean=0.5, std=0.5 for scratch baselines."""

    def __init__(self):
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)

    def normalize(self, tensor: np.ndarray) -> np.ndarray:
        return (tensor - self.mean) / self.std


class ChestXRayPreprocessor(IPreprocessor):
    """Standardized preprocessor for chest radiographs.

    Encapsulates image loading, 3-channel replication, bilinear resizing,
    and configurable normalization strategy without hardcoded global parameters.
    """

    def __init__(
        self,
        image_size: int = 224,
        normalization_strategy: INormalizationStrategy | None = None,
    ) -> None:
        self.image_size = int(image_size)
        self.normalization_strategy = normalization_strategy or ImageNetNormalizationStrategy()

    def preprocess(self, image_input: Union[str, Path, Image.Image]) -> torch.Tensor:
        """Loads and precomputes a normalized tensor of shape (1, 3, image_size, image_size)."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Radiograph image not found at: {path}")
            with Image.open(path) as raw_img:
                pil_img = raw_img.convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # High-quality bilinear resize
        pil_img = pil_img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.array(pil_img, dtype=np.float32) / 255.0

        # Transpose HWC -> CHW
        chw = arr.transpose(2, 0, 1)
        # Apply normalization strategy
        normalized = self.normalization_strategy.normalize(chw)
        # Add batch dimension -> (1, 3, H, W)
        return torch.from_numpy(normalized).unsqueeze(0)
