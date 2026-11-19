"""Output formatters implementing the Strategy Pattern for clinical reporting."""
from __future__ import annotations

import json
from src.domain.entities import PredictionResult
from src.domain.enums import DiagnosisClass
from src.domain.interfaces import IFormatter


class ConsoleDiagnosticFormatter(IFormatter):
    """Generates rich, human-readable terminal diagnostic reports with ASCII probability bars."""

    def __init__(self, bar_width: int = 30) -> None:
        self.bar_width = int(bar_width)

    def format(self, result: PredictionResult) -> str:
        pred_cls = result.predicted_class
        lines = [
            "=" * 72,
            "CS320 DEEP LEARNING PEDIATRIC PNEUMONIA CLASSIFICATION (PROJECT P06)",
            "=" * 72,
        ]

        if result.image_path:
            lines.append(f"Input Radiograph:      {result.image_path}")
        if result.sample_index is not None:
            lines.append(f"Clinical Test Index:   #{result.sample_index}")
        if result.ground_truth is not None:
            status_tag = "[MATCH / CORRECT]" if result.is_correct else "[MISCLASSIFICATION]"
            lines.append(f"Ground Truth:          {result.ground_truth.display_name} {status_tag}")
        lines.append(f"Model Architecture:    {result.model_name}")
        if result.inference_latency_ms > 0:
            lines.append(f"Inference Latency:     {result.inference_latency_ms:.2f} ms")
        lines.append("-" * 72)

        conf_pct = result.confidence * 100
        lines.append(f"PREDICTED DIAGNOSIS:   >>> {pred_cls.display_name} <<< (Confidence: {conf_pct:.2f}%)")
        lines.append(f"Clinical Title:        {pred_cls.clinical_title}")
        lines.append("-" * 72)

        lines.append("Diagnostic Class Probabilities:")
        for cls in DiagnosisClass:
            prob = result.probabilities.get(cls, 0.0)
            bar_len = int(prob * self.bar_width)
            bar = "#" * bar_len
            marker = "<-- PREDICTED" if cls == pred_cls else ""
            lines.append(f"  {cls.display_name:<10}: {prob * 100:6.2f}%  |{bar:<{self.bar_width}}| {marker}")

        lines.append("-" * 72)
        lines.append("Radiological Hallmark:")
        lines.append(f"  • {pred_cls.radiological_hallmark}")
        lines.append("Clinical Pathway:")
        lines.append(f"  • {pred_cls.clinical_pathway}")

        if result.is_ambiguous:
            lines.append("!" * 72)
            lines.append("WARNING: Low margin of confidence (< 15% between top 2 classes).")
            lines.append("Senior pediatric radiologist review strongly advised.")
            lines.append("!" * 72)

        lines.append("=" * 72)
        return "\n".join(lines)


class JsonDiagnosticFormatter(IFormatter):
    """Serializes result into standardized, indented JSON."""

    def __init__(self, indent: int = 2) -> None:
        self.indent = indent

    def format(self, result: PredictionResult) -> str:
        return result.to_json(indent=self.indent)


class MarkdownDiagnosticFormatter(IFormatter):
    """Formats result as a GitHub-flavored markdown table for notebooks and artifacts."""

    def format(self, result: PredictionResult) -> str:
        lines = [
            f"### Diagnosis: **{result.predicted_class.display_name}** ({result.confidence * 100:.2f}%)",
            f"- **Clinical Terminology:** {result.predicted_class.clinical_title}",
            f"- **Model Used:** `{result.model_name}`",
        ]
        if result.ground_truth:
            icon = "✅" if result.is_correct else "❌"
            lines.append(f"- **Ground Truth:** {result.ground_truth.display_name} {icon}")

        lines.append("\n| Condition | Probability | Visualization |")
        lines.append("|---|---:|---|")
        for cls in DiagnosisClass:
            prob = result.probabilities.get(cls, 0.0)
            bar = "█" * int(prob * 20)
            bold = "**" if cls == result.predicted_class else ""
            lines.append(f"| {bold}{cls.display_name}{bold} | {bold}{prob * 100:.2f}%{bold} | `{bar}` |")

        lines.append(f"\n> **Radiological Hallmark:** {result.predicted_class.radiological_hallmark}")
        return "\n".join(lines)
