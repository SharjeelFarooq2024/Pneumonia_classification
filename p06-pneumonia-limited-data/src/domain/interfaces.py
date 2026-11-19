"""Abstract base interfaces and protocols for dependency inversion."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import torch

from src.domain.entities import CalibrationBin, PredictionResult


class IPreprocessor(ABC):
    """Interface for image preprocessing and normalization pipelines."""

    @abstractmethod
    def preprocess(self, image_input: str | Path | Any) -> torch.Tensor:
        """Preprocesses an input image into a normalized 4D tensor: (1, C, H, W)."""
        pass


class IPredictor(ABC):
    """Interface for model inference engines."""

    @abstractmethod
    def predict(self, image_input: str | Path | Any, ground_truth: Any = None) -> PredictionResult:
        """Runs forward inference and returns a validated PredictionResult."""
        pass


class IFormatter(ABC):
    """Strategy interface for output reporting (Console, JSON, Markdown)."""

    @abstractmethod
    def format(self, result: PredictionResult) -> str:
        """Formats a PredictionResult entity into a specific presentation format."""
        pass


class ICalibrationAlgorithm(ABC):
    """Interface for probability calibration and reliability diagram algorithms."""

    @abstractmethod
    def compute_ece(
        self,
        y_true: Sequence[int],
        y_prob: np.ndarray,
        n_bins: int = 15,
    ) -> Tuple[float, List[CalibrationBin]]:
        """Computes Expected Calibration Error and returns structured CalibrationBins."""
        pass
