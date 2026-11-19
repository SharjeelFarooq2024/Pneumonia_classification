"""Domain modeling layer for CS320 Term Project (P06).

Defines strongly-typed domain enums, clinical entities, value objects,
and abstract interfaces following clean architecture and OOP principles.
"""
from __future__ import annotations

from src.domain.enums import DiagnosisClass, DatasetSplit, ModelRegime
from src.domain.entities import PredictionResult, CalibrationBin, MetricSummary
from src.domain.interfaces import (
    IPreprocessor,
    IPredictor,
    IFormatter,
    ICalibrationAlgorithm,
)

__all__ = [
    "DiagnosisClass",
    "DatasetSplit",
    "ModelRegime",
    "PredictionResult",
    "CalibrationBin",
    "MetricSummary",
    "IPreprocessor",
    "IPredictor",
    "IFormatter",
    "ICalibrationAlgorithm",
]
