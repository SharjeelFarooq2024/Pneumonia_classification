"""Domain value objects and entities."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.domain.enums import DiagnosisClass


@dataclass(frozen=True)
class CalibrationBin:
    """Represents a discrete confidence interval in reliability calibration."""
    bin_index: int
    bin_lower: float
    bin_upper: float
    sample_count: int
    mean_confidence: float
    accuracy: float

    @property
    def calibration_error(self) -> float:
        """Absolute discrepancy between perceived confidence and empirical accuracy."""
        return abs(self.accuracy - self.mean_confidence)

    @property
    def is_overconfident(self) -> bool:
        return self.mean_confidence > self.accuracy

    @property
    def is_underconfident(self) -> bool:
        return self.mean_confidence < self.accuracy


@dataclass(frozen=True)
class PredictionResult:
    """Complete diagnostic prediction record for a chest radiograph."""
    predicted_class: DiagnosisClass
    confidence: float
    probabilities: Dict[DiagnosisClass, float]
    image_path: Optional[str] = None
    sample_index: Optional[int] = None
    ground_truth: Optional[DiagnosisClass] = None
    model_name: str = "ResNet-18 (Staged Fine-Tuned)"
    inference_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_correct(self) -> Optional[bool]:
        """Returns True if ground truth exists and matches prediction, False if mismatch, None if unlabelled."""
        if self.ground_truth is None:
            return None
        return self.predicted_class == self.ground_truth

    @property
    def margin_of_confidence(self) -> float:
        """Difference between highest probability and second-highest probability."""
        sorted_probs = sorted(self.probabilities.values(), reverse=True)
        if len(sorted_probs) < 2:
            return 1.0
        return sorted_probs[0] - sorted_probs[1]

    @property
    def is_ambiguous(self) -> bool:
        """Flag cases where the top two predictions are within 15% probability."""
        return self.margin_of_confidence < 0.15

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entity into a clean dictionary with primitive types."""
        return {
            "predicted_class": self.predicted_class.display_name,
            "predicted_label": int(self.predicted_class.value),
            "confidence": float(self.confidence),
            "margin_of_confidence": float(self.margin_of_confidence),
            "is_ambiguous": bool(self.is_ambiguous),
            "is_correct": self.is_correct,
            "ground_truth": self.ground_truth.display_name if self.ground_truth else None,
            "probabilities": {cls.display_name: float(prob) for cls, prob in self.probabilities.items()},
            "image_path": str(self.image_path) if self.image_path else None,
            "sample_index": self.sample_index,
            "model_name": self.model_name,
            "inference_latency_ms": round(self.inference_latency_ms, 2),
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return formatted JSON representation."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class MetricSummary:
    """Comprehensive metric report across a split or experiment."""
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_auroc: float
    expected_calibration_error: float
    per_class_f1: Dict[DiagnosisClass, float]
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "macro_auroc": self.macro_auroc,
            "expected_calibration_error": self.expected_calibration_error,
            "per_class_f1": {k.display_name: v for k, v in self.per_class_f1.items()},
            "sample_count": self.sample_count,
        }
