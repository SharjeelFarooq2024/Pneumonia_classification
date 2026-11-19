"""Test the interactive demo function in scripts/demo.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.demo import run_demo


def test_demo_function_virus(capsys):
    run_demo(class_filter="VIRUS")
    captured = capsys.readouterr()
    assert "CS320 DEEP LEARNING TERM PROJECT" in captured.out
    assert "Proposed ResNet-18" in captured.out
    assert "VIRUS" in captured.out


def test_demo_function_normal(capsys):
    run_demo(class_filter="NORMAL")
    captured = capsys.readouterr()
    assert "NORMAL" in captured.out


def test_predict_image_pipeline(tmp_path):
    from PIL import Image
    from scripts.predict_image import preprocess_image, predict_from_image

    # Create a synthetic 100x100 dummy grayscale image
    dummy_img = Image.new("L", (100, 100), color=128)
    dummy_path = tmp_path / "dummy_xray.png"
    dummy_img.save(dummy_path)

    # Test preprocessor shape
    tensor = preprocess_image(dummy_path, image_size=224)
    assert tensor.shape == (1, 3, 224, 224)

    # Test prediction
    res = predict_from_image(dummy_path)
    assert "predicted_class" in res
    assert res["predicted_class"] in ["NORMAL", "BACTERIA", "VIRUS"]
    assert 0.0 <= res["confidence"] <= 1.0

