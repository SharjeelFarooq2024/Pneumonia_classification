"""Predictor service implementing Dependency Injection for model inference."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.domain.entities import PredictionResult
from src.domain.enums import DiagnosisClass
from src.domain.interfaces import IPredictor, IPreprocessor
from src.services.preprocessor import ChestXRayPreprocessor


class ChestXRayPredictor(IPredictor):
    """High-level inference engine for pediatric pneumonia detection.

    Encapsulates tensor transformations, forward pass execution, latency profiling,
    and strongly-typed result formulation via dependency injection.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        preprocessor: Optional[IPreprocessor] = None,
        device: Optional[torch.device] = None,
        model_name: str = "ResNet-18 (Staged Fine-Tuned)",
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocessor = preprocessor or ChestXRayPreprocessor()
        self.model_name = str(model_name)

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, Image.Image],
        ground_truth: Optional[Union[DiagnosisClass, int, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        """Preprocesses an input image, executes inference, and returns a PredictionResult."""
        start_time = time.perf_counter()

        tensor = self.preprocessor.preprocess(image_input).to(self.device)
        logits = self.model(tensor)
        probs_raw = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Formulate probabilities dictionary keyed by DiagnosisClass
        prob_dict: Dict[DiagnosisClass, float] = {}
        for cls in DiagnosisClass:
            prob_dict[cls] = float(probs_raw[cls.value])

        pred_label = int(np.argmax(probs_raw))
        pred_class = DiagnosisClass.from_label(pred_label)
        confidence = float(probs_raw[pred_label])

        gt_enum: Optional[DiagnosisClass] = None
        if ground_truth is not None:
            if isinstance(ground_truth, DiagnosisClass):
                gt_enum = ground_truth
            elif isinstance(ground_truth, (int, np.integer)):
                gt_enum = DiagnosisClass.from_label(int(ground_truth))
            elif isinstance(ground_truth, str):
                gt_enum = DiagnosisClass.from_name(ground_truth)

        img_path_str = str(image_input) if isinstance(image_input, (str, Path)) else None

        return PredictionResult(
            predicted_class=pred_class,
            confidence=confidence,
            probabilities=prob_dict,
            image_path=img_path_str,
            ground_truth=gt_enum,
            model_name=self.model_name,
            inference_latency_ms=latency_ms,
            metadata=metadata or {},
        )

    @classmethod
    def from_test_record(
        cls,
        row: pd.Series,
        sample_index: int,
        model_name: str = "ResNet-18 (Evaluated Test)",
    ) -> PredictionResult:
        """Formulates a PredictionResult directly from evaluated test split dataframe."""
        pred_label = int(row["predicted_label"])
        true_label = int(row["true_label"]) if "true_label" in row else None

        pred_class = DiagnosisClass.from_label(pred_label)
        gt_class = DiagnosisClass.from_label(true_label) if true_label is not None else None

        prob_dict: Dict[DiagnosisClass, float] = {}
        for c in DiagnosisClass:
            col_name = f"prob_{c.display_name}"
            prob_dict[c] = float(row[col_name]) if col_name in row else 0.0

        confidence = float(row.get("confidence", prob_dict[pred_class]))

        return PredictionResult(
            predicted_class=pred_class,
            confidence=confidence,
            probabilities=prob_dict,
            sample_index=sample_index,
            ground_truth=gt_class,
            model_name=model_name,
            inference_latency_ms=0.0,
        )
