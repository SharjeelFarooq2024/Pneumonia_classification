"""Comprehensive test suite for the Object-Oriented Domain, Algorithms, and Services architecture."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from src.algorithms.calibration import AdaptiveQuantileCalibration, FastHistogramCalibration
from src.algorithms.stratification import StratifiedSampler
from src.domain.entities import CalibrationBin, PredictionResult
from src.domain.enums import DiagnosisClass
from src.models.baseline_cnn import BaselineCNN
from src.services.formatters import ConsoleDiagnosticFormatter, JsonDiagnosticFormatter, MarkdownDiagnosticFormatter
from src.services.predictor import ChestXRayPredictor
from src.services.preprocessor import (
    ChestXRayPreprocessor,
    ImageNetNormalizationStrategy,
    SymmetricNormalizationStrategy,
)


# ===========================================================================
# 1. Domain Enums & Entities Tests
# ===========================================================================
def test_diagnosis_class_enum_properties():
    assert DiagnosisClass.NORMAL.value == 0
    assert DiagnosisClass.BACTERIA.value == 1
    assert DiagnosisClass.VIRUS.value == 2

    assert DiagnosisClass.NORMAL.display_name == "NORMAL"
    assert "Bacterial" in DiagnosisClass.BACTERIA.clinical_title
    assert "interstitial" in DiagnosisClass.VIRUS.radiological_hallmark.lower()

    # Enum lookups
    assert DiagnosisClass.from_name("virus") == DiagnosisClass.VIRUS
    assert DiagnosisClass.from_name("BACTERIA") == DiagnosisClass.BACTERIA
    assert DiagnosisClass.from_label(0) == DiagnosisClass.NORMAL

    with pytest.raises(ValueError):
        DiagnosisClass.from_name("COVID19")

    with pytest.raises(ValueError):
        DiagnosisClass.from_label(99)


def test_prediction_result_entity():
    probs = {
        DiagnosisClass.NORMAL: 0.10,
        DiagnosisClass.BACTERIA: 0.75,
        DiagnosisClass.VIRUS: 0.15,
    }
    res = PredictionResult(
        predicted_class=DiagnosisClass.BACTERIA,
        confidence=0.75,
        probabilities=probs,
        ground_truth=DiagnosisClass.BACTERIA,
        inference_latency_ms=12.5,
    )

    assert res.is_correct is True
    assert res.margin_of_confidence == pytest.approx(0.75 - 0.15)
    assert res.is_ambiguous is False

    # Ambiguous case test
    ambiguous_probs = {
        DiagnosisClass.NORMAL: 0.05,
        DiagnosisClass.BACTERIA: 0.48,
        DiagnosisClass.VIRUS: 0.47,
    }
    amb_res = PredictionResult(
        predicted_class=DiagnosisClass.BACTERIA,
        confidence=0.48,
        probabilities=ambiguous_probs,
    )
    assert amb_res.is_ambiguous is True
    assert amb_res.is_correct is None  # No ground truth provided

    # Serialization tests
    data = res.to_dict()
    assert data["predicted_class"] == "BACTERIA"
    assert data["confidence"] == 0.75

    json_str = res.to_json()
    parsed = json.loads(json_str)
    assert parsed["predicted_label"] == 1


# ===========================================================================
# 2. DSA Algorithms Tests
# ===========================================================================
def test_fast_histogram_calibration():
    calib = FastHistogramCalibration()
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    ece, bins = calib.compute_ece(y_true, y_prob, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-5)
    assert len(bins) > 0
    assert all(isinstance(b, CalibrationBin) for b in bins)


def test_adaptive_quantile_calibration():
    calib = AdaptiveQuantileCalibration()
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([[0.9, 0.1]] * 5 + [[0.2, 0.8]] * 5)

    ece, bins = calib.compute_ece(y_true, y_prob, n_bins=2)
    assert ece >= 0.0
    assert len(bins) == 2
    assert all(b.sample_count == 5 for b in bins)


def test_stratified_sampler_quota_allocation():
    # 100 samples with 20, 50, 30 distribution
    labels = np.array([0] * 20 + [1] * 50 + [2] * 30)
    quotas = StratifiedSampler.compute_class_quotas(labels, fraction=0.25)

    assert quotas[0] == 5   # 20 * 0.25 = 5
    assert quotas[1] == 13  # 50 * 0.25 = 12.5 -> rounds to 13 by largest remainder
    assert quotas[2] == 7   # 30 * 0.25 = 7.5 -> rounds to 7 (5+13+7 = 25)
    assert sum(quotas.values()) == 25

    selected = StratifiedSampler.sample_indices(labels, fraction=0.25, seed=42)
    assert len(selected) == 25
    assert np.all(np.diff(selected) >= 0)  # Verify monotonic ascending order


# ===========================================================================
# 3. Strategy Pattern & Preprocessing Tests
# ===========================================================================
def test_normalization_strategies():
    arr = np.ones((3, 224, 224), dtype=np.float32)

    imagenet_strat = ImageNetNormalizationStrategy()
    norm_img = imagenet_strat.normalize(arr)
    assert norm_img.shape == (3, 224, 224)
    # Check channel 0 normalized with mean 0.485, std 0.229
    expected_c0 = (1.0 - 0.485) / 0.229
    assert norm_img[0, 0, 0] == pytest.approx(expected_c0, rel=1e-3)

    symmetric_strat = SymmetricNormalizationStrategy()
    norm_sym = symmetric_strat.normalize(arr)
    # Check channel 0 normalized with mean 0.5, std 0.5 -> (1.0 - 0.5) / 0.5 = 1.0
    assert norm_sym[0, 0, 0] == pytest.approx(1.0, rel=1e-3)


def test_chest_xray_preprocessor(tmp_path):
    img = Image.new("L", (150, 150), color=200)
    img_path = tmp_path / "sample.png"
    img.save(img_path)

    preprocessor = ChestXRayPreprocessor(image_size=224)
    tensor = preprocessor.preprocess(img_path)

    assert tensor.shape == (1, 3, 224, 224)
    assert isinstance(tensor, torch.Tensor)


# ===========================================================================
# 4. Service Layer Formatters & Predictor Tests
# ===========================================================================
def test_formatters():
    probs = {
        DiagnosisClass.NORMAL: 0.05,
        DiagnosisClass.BACTERIA: 0.15,
        DiagnosisClass.VIRUS: 0.80,
    }
    res = PredictionResult(
        predicted_class=DiagnosisClass.VIRUS,
        confidence=0.80,
        probabilities=probs,
        ground_truth=DiagnosisClass.VIRUS,
    )

    console_fmt = ConsoleDiagnosticFormatter(bar_width=20)
    console_out = console_fmt.format(res)
    assert "VIRUS" in console_out
    assert "MATCH" in console_out
    assert "Radiological Hallmark" in console_out

    json_fmt = JsonDiagnosticFormatter()
    json_out = json_fmt.format(res)
    parsed = json.loads(json_out)
    assert parsed["predicted_class"] == "VIRUS"

    md_fmt = MarkdownDiagnosticFormatter()
    md_out = md_fmt.format(res)
    assert "### Diagnosis: **VIRUS**" in md_out


def test_chest_xray_predictor_inference():
    # Use BaselineCNN as lightweight model for testing dependency injection
    model = BaselineCNN(input_channels=3, num_classes=3)
    preprocessor = ChestXRayPreprocessor(image_size=224, normalization_strategy=SymmetricNormalizationStrategy())
    predictor = ChestXRayPredictor(model=model, preprocessor=preprocessor, device=torch.device("cpu"), model_name="TestBaseline")

    # Synthetic image input
    raw_img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    res = predictor.predict(raw_img, ground_truth=DiagnosisClass.NORMAL)

    assert isinstance(res, PredictionResult)
    assert res.predicted_class in [DiagnosisClass.NORMAL, DiagnosisClass.BACTERIA, DiagnosisClass.VIRUS]
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.probabilities) == 3
    assert res.inference_latency_ms > 0.0
