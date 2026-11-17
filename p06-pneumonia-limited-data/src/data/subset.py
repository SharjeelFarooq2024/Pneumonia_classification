"""Stratified training subsets.

The official P06 subset is 25% of the train split, stratified by class with
seed 42. It is created once and reused by every experiment. Additional
fractions (10%, 100%) are only for the supplementary data-efficiency curve.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from src.data.dataset import get_split_global_indices, get_split_labels
from src.utils import resolve_path, to_jsonable


def stratified_positions(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    if n == 0:
        raise ValueError("Cannot subset an empty split.")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return np.arange(n, dtype=int)
    from sklearn.model_selection import train_test_split

    indices = np.arange(n, dtype=int)
    selected, _ = train_test_split(
        indices,
        train_size=fraction,
        stratify=labels,
        random_state=seed,
    )
    return np.sort(selected.astype(int))


def _class_summary(labels: np.ndarray, class_names):
    labels = np.asarray(labels, dtype=int)
    counts = {name: int((labels == k).sum()) for k, name in enumerate(class_names)}
    n = max(len(labels), 1)
    proportions = {name: counts[name] / n for name in class_names}
    return counts, proportions


def build_subset_payload(
    config: Mapping[str, Any],
    fraction: float,
    seed: int,
    split: str = "train",
) -> Dict[str, Any]:
    labels = get_split_labels(config, split)
    full_global = get_split_global_indices(config, split)
    positions = stratified_positions(labels, fraction, seed)
    subset_labels = labels[positions]
    subset_global = [int(full_global[p]) for p in positions]

    class_names = list(config["data"]["class_names"])
    full_counts, full_props = _class_summary(labels, class_names)
    sub_counts, sub_props = _class_summary(subset_labels, class_names)
    max_diff = max(abs(full_props[c] - sub_props[c]) for c in class_names)

    payload = {
        "name": f"{split}{int(round(fraction * 100))}_seed{seed}",
        "split": split,
        "fraction": float(fraction),
        "seed": int(seed),
        "n_full_split": int(len(labels)),
        "n_subset": int(len(positions)),
        "positions": [int(p) for p in positions],   # positions within the split, sorted
        "indices": subset_global,                    # global MedIMeta image indices
        "class_counts": sub_counts,
        "class_proportions": sub_props,
        "full_split_class_counts": full_counts,
        "full_split_class_proportions": full_props,
        "max_abs_class_proportion_diff": float(max_diff),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "medimeta pneumonia, split file data/raw/pneumonia/splits/train.txt",
    }
    return to_jsonable(payload)


def create_subset(
    config: Mapping[str, Any],
    fraction: float,
    seed: int,
    output_path: str | Path,
    force: bool = False,
    split: str = "train",
) -> Dict[str, Any]:
    output_path = resolve_path(output_path)
    if output_path.exists() and not force:
        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if int(payload.get("seed", -1)) != int(seed) or float(payload.get("fraction", -1)) != float(fraction):
            raise ValueError(
                f"{output_path} already exists with seed={payload.get('seed')} "
                f"fraction={payload.get('fraction')}; refusing to overwrite."
            )
        return payload

    payload = build_subset_payload(config, fraction, seed, split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def verify_subset(payload: Mapping[str, Any], tol: float = 0.01) -> Tuple[bool, str]:
    diffs = {
        c: abs(payload["full_split_class_proportions"][c] - payload["class_proportions"][c])
        for c in payload["class_proportions"]
    }
    worst = max(diffs.values()) if diffs else 0.0
    ok = worst <= tol
    lines = [
        f"Subset {payload['name']}: {payload['n_subset']}/{payload['n_full_split']} samples "
        f"({100 * payload['fraction']:.1f}%).",
        "Class counts (subset vs full train):",
    ]
    for c in payload["class_counts"]:
        lines.append(
            f"  {c}: {payload['class_counts'][c]} vs {payload['full_split_class_counts'][c]} "
            f"| proportion {payload['class_proportions'][c]:.4f} vs "
            f"{payload['full_split_class_proportions'][c]:.4f} | diff {diffs[c]:.4f}"
        )
    lines.append(f"Max absolute proportion difference: {worst:.4f} (tolerance {tol}).")
    return ok, "\n".join(lines)


def save_split_manifests(config: Mapping[str, Any], output_dir: str | Path = "data/splits") -> Dict[str, Any]:
    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        labels = get_split_labels(config, split)
        indices = get_split_global_indices(config, split)
        class_names = list(config["data"]["class_names"])
        counts = {name: int((labels == k).sum()) for k, name in enumerate(class_names)}
        manifest[split] = {
            "n": int(len(indices)),
            "global_indices": indices,
            "class_counts": counts,
        }
    with open(out_dir / "split_indices.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
