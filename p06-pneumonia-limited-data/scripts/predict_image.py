"""Modular, Object-Oriented Single-Image Disease Prediction CLI.

Demonstrates Clean Architecture, Dependency Injection, and Strategy Pattern
for pediatric chest radiograph diagnosis.

Usage:
  python scripts/predict_image.py --image path/to/chest_xray.png
  python scripts/predict_image.py --image path/to/chest_xray.jpg --format json
  python scripts/predict_image.py --sample-index 368
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.domain.enums import DiagnosisClass
from src.domain.interfaces import IFormatter
from src.models.proposed import build_model
from src.services.formatters import ConsoleDiagnosticFormatter, JsonDiagnosticFormatter, MarkdownDiagnosticFormatter
from src.services.predictor import ChestXRayPredictor
from src.services.preprocessor import ChestXRayPreprocessor, ImageNetNormalizationStrategy
from src.utils import load_config, resolve_path


def get_formatter(format_name: str) -> IFormatter:
    """Formatter Factory returning the appropriate IFormatter strategy."""
    formatters = {
        "console": ConsoleDiagnosticFormatter(),
        "json": JsonDiagnosticFormatter(),
        "markdown": MarkdownDiagnosticFormatter(),
    }
    return formatters.get(format_name.lower(), ConsoleDiagnosticFormatter())


def initialize_predictor(checkpoint_path: str | Path | None = None, device_str: str = "auto") -> ChestXRayPredictor:
    """Factory creating a fully configured ChestXRayPredictor with dependency injection."""
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) or device_str == "cuda" else "cpu")
    
    cfg = load_config("configs/proposed.yaml")
    model = build_model(cfg)
    
    ckpt_file = Path(checkpoint_path) if checkpoint_path else ROOT / "checkpoints" / "proposed_seed42.pt"
    model_name = "ResNet-18 (Staged Fine-Tuned)"
    if ckpt_file.exists():
        ckpt = torch.load(ckpt_file, map_location=device)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model_name += f" [Weights: {ckpt_file.name}]"
    else:
        model_name += " [Feature Extractor Backbone]"

    preprocessor = ChestXRayPreprocessor(
        image_size=int(cfg["data"].get("image_size", 224)),
        normalization_strategy=ImageNetNormalizationStrategy(),
    )

    return ChestXRayPredictor(
        model=model,
        preprocessor=preprocessor,
        device=device,
        model_name=model_name,
    )


def preprocess_image(image_path: str | Path, image_size: int = 224) -> torch.Tensor:
    """Facade helper for backwards-compatible tensor preprocessing."""
    preprocessor = ChestXRayPreprocessor(image_size=image_size)
    return preprocessor.preprocess(image_path)


def predict_from_image(
    image_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
) -> dict:
    """Facade helper returning dictionary representation of prediction."""
    predictor = initialize_predictor(checkpoint_path, device)
    result = predictor.predict(image_path)
    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Object-Oriented Pediatric Pneumonia Diagnostic Predictor.")
    parser.add_argument("--image", type=str, default=None, help="Path to chest radiograph image (PNG/JPG/TIFF).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch model checkpoint (.pt).")
    parser.add_argument("--sample-index", type=int, default=None, help="Evaluate an official clinical test set case.")
    parser.add_argument("--format", type=str, default="console", choices=["console", "json", "markdown"], help="Output presentation format.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference hardware device.")
    args = parser.parse_args()

    formatter = get_formatter(args.format)

    if args.sample_index is not None:
        from scripts.demo import run_demo
        run_demo(sample_idx=args.sample_index)
        return

    if args.image is None:
        print("Error: No image supplied. Please provide --image <path> or --sample-index <int>.")
        print("\nAvailable classes:", DiagnosisClass.all_names())
        print("\nQuick Start Examples:")
        print("  python scripts/predict_image.py --image path/to/scan.png")
        print("  python scripts/predict_image.py --sample-index 368")
        print("  python scripts/demo.py --class-name VIRUS")
        return

    predictor = initialize_predictor(args.checkpoint, args.device)
    result = predictor.predict(args.image)
    print(formatter.format(result))


if __name__ == "__main__":
    main()
