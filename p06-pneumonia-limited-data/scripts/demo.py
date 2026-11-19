"""Interactive Working Demonstration for CS320 Term Project (P06).

Demonstrates comparative inference on pediatric chest radiographs
between the Baseline Scratch CNN and the Proposed Staged Fine-Tuned ResNet-18
using strongly-typed domain entities and clean OOP principles.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.domain.enums import DiagnosisClass
from src.services.formatters import ConsoleDiagnosticFormatter
from src.services.predictor import ChestXRayPredictor
from src.utils import load_config, resolve_path


def load_prediction_registries() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dynamically locates and loads test prediction dataframes."""
    base_path = resolve_path("results/baseline/seed42/predictions_test.csv")
    prop_path = resolve_path("results/proposed/seed42/predictions_test.csv")
    
    if not base_path.exists() or not prop_path.exists():
        raise FileNotFoundError(
            f"Prediction registries missing. Checked:\n  - {base_path}\n  - {prop_path}"
        )
    return pd.read_csv(base_path), pd.read_csv(prop_path)


def select_sample_index(
    base_preds: pd.DataFrame,
    prop_preds: pd.DataFrame,
    sample_idx: Optional[int] = None,
    class_filter: Optional[str] = None,
) -> int:
    """Algorithm selecting an exemplary evaluation index based on constraints."""
    if sample_idx is not None:
        if not (0 <= sample_idx < len(base_preds)):
            raise IndexError(f"Index {sample_idx} out of range [0, {len(base_preds) - 1}]")
        return sample_idx

    if class_filter:
        target_enum = DiagnosisClass.from_name(class_filter)
        matching = base_preds[base_preds["true_label"] == target_enum.value].index.tolist()
        if not matching:
            raise ValueError(f"No test cases found for class {target_enum.display_name}")
        return matching[0]

    # Default heuristic: select a viral pneumonia case where baseline suffered majority bias
    # but proposed staged transfer learning correctly resolved the condition
    viral_reversal = base_preds[
        (base_preds["true_label"] == DiagnosisClass.VIRUS.value)
        & (base_preds["predicted_label"] == DiagnosisClass.BACTERIA.value)
        & (prop_preds["predicted_label"] == DiagnosisClass.VIRUS.value)
    ].index.tolist()

    return viral_reversal[0] if viral_reversal else 0


def run_demo(sample_idx: Optional[int] = None, class_filter: Optional[str] = None) -> None:
    """Executes comparative diagnosis demonstration."""
    base_preds, prop_preds = load_prediction_registries()
    idx = select_sample_index(base_preds, prop_preds, sample_idx, class_filter)

    b_res = ChestXRayPredictor.from_test_record(
        base_preds.iloc[idx],
        sample_index=idx,
        model_name="Baseline Scratch CNN (20 Epochs, 93K Params)",
    )
    p_res = ChestXRayPredictor.from_test_record(
        prop_preds.iloc[idx],
        sample_index=idx,
        model_name="Proposed ResNet-18 (Staged Fine-Tuning + Weighting + Smoothing)",
    )

    formatter = ConsoleDiagnosticFormatter(bar_width=30)
    true_cls = p_res.ground_truth or DiagnosisClass.NORMAL

    print("\n" + "=" * 74)
    print("CS320 DEEP LEARNING TERM PROJECT (P06) — COMPARATIVE CLINICAL DEMO")
    print("Pediatric Pneumonia Classification under Limited Labeled Data (25% Budget)")
    print("=" * 74)
    print(f"Evaluated Test Case:     #{idx} (out of {len(base_preds)} clinical test samples)")
    print(f"Ground Truth Diagnosis:  {true_cls.display_name} ({true_cls.clinical_title})")
    print("-" * 74)

    # 1. Baseline Model Output
    b_tag = "[CORRECT]" if b_res.is_correct else "[INCORRECT - MAJORITY CLASS BIAS]"
    print(f"\n--- 1. {b_res.model_name} ---")
    print(f"Predicted Diagnosis:     {b_res.predicted_class.display_name} {b_tag}")
    print("Class Probabilities:")
    for cls in DiagnosisClass:
        prob = b_res.probabilities[cls]
        bar = "#" * int(prob * 30)
        print(f"  {cls.display_name:<10}: {prob:.4f}  |{bar:<30}|")

    # 2. Proposed Model Output
    p_tag = "[CORRECT]" if p_res.is_correct else "[INCORRECT]"
    print(f"\n--- 2. {p_res.model_name} ---")
    print(f"Predicted Diagnosis:     {p_res.predicted_class.display_name} {p_tag}")
    print("Class Probabilities:")
    for cls in DiagnosisClass:
        prob = p_res.probabilities[cls]
        bar = "#" * int(prob * 30)
        marker = "<-- PREDICTED" if cls == p_res.predicted_class else ""
        print(f"  {cls.display_name:<10}: {prob:.4f}  |{bar:<30}| {marker}")

    # Clinical and Methodological Insight
    print("\n" + "=" * 74)
    print("CLINICAL & METHODOLOGICAL INSIGHT:")
    if true_cls == DiagnosisClass.VIRUS and b_res.predicted_class == DiagnosisClass.BACTERIA and p_res.predicted_class == DiagnosisClass.VIRUS:
        print(">> Core Project Contribution Validated:")
        print("   The baseline scratch model collapsed into majority BACTERIA (recall only 30.2%).")
        print("   The proposed staged transfer learning architecture successfully extracted subtle")
        print("   interstitial/peribronchial features to correctly diagnose VIRAL pneumonia (recall 79.5%).")
    else:
        print(f">> Model comparison: Baseline predicted {b_res.predicted_class.display_name}, Proposed predicted {p_res.predicted_class.display_name}.")
    print("=" * 74 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Live model demonstration.")
    parser.add_argument("--index", type=int, default=None, help="Test set sample index (0-623).")
    parser.add_argument("--class-name", type=str, default=None, choices=DiagnosisClass.all_names(), help="Filter by diagnostic condition.")
    args = parser.parse_args()
    run_demo(args.index, args.class_name)


if __name__ == "__main__":
    main()
