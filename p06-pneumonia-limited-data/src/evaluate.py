"""Evaluate a frozen checkpoint on validation and (once, finally) test.

Usage:
    python -m src.evaluate --checkpoint checkpoints/baseline_seed42.pt
    python -m src.evaluate --checkpoint checkpoints/proposed.pt --splits val test

The test split is evaluated only by this script, after all training and model
selection decisions have been frozen. A marker prevents accidental re-runs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.data.dataset import make_dataloader, make_datasets
from src.models.proposed import build_model
from src.utils import (
    amp_is_available,
    compute_metrics,
    experiment_results_dir,
    environment_info,
    get_device,
    git_commit,
    predict_loader,
    resolve_path,
    save_json,
    save_predictions_csv,
    set_seed,
    setup_logger,
    to_jsonable,
)


def plot_confusion(cm: np.ndarray, class_names: List[str], title: str,
                   path: Path, normalize: bool = False) -> None:
    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)
    plt.figure(figsize=(5.2, 4.4))
    sns.heatmap(data, annot=True, fmt=".2f" if normalize else ".0f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate_checkpoint(
    checkpoint_path: str,
    splits: List[str],
    force: bool = False,
    device_override: Optional[str] = None,
    batch_size_override: Optional[int] = None,
    limit_batches: int = 0,
) -> Dict[str, Any]:
    ckpt_path = resolve_path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    exp = str(config.get("experiment", ckpt.get("experiment", "run")))
    seed = int(ckpt.get("seed", config.get("seed", 42)))
    device = get_device(device_override or config.get("device", "auto"))
    set_seed(seed)

    run_dir = experiment_results_dir(config, exp) / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"eval.{exp}.{seed}", run_dir / "evaluate.log",
                          level=config.get("logging", {}).get("level", "INFO"))
    logger.info("Evaluating checkpoint %s on splits=%s device=%s", ckpt_path, splits, device)

    marker = run_dir / "test_eval_done.json"
    evaluated_splits = list(splits)
    if "test" in evaluated_splits and marker.exists() and not force:
        raise RuntimeError(
            f"Test evaluation marker already exists at {marker}. "
            "The test set must be evaluated exactly once per final model. "
            "Use --force only to repair a clearly broken run; never to chase numbers."
        )

    eval_config = {**config, "model": {**config.get("model", {}), "pretrained": False}}
    model = build_model(eval_config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    amp = amp_is_available(config, device)
    class_names = list(config["data"]["class_names"])
    eval_batch_size = int(batch_size_override or config["eval"].get("batch_size", 64))
    bootstrap_n = int(config["eval"].get("bootstrap_n", 1000))
    bootstrap_seed = int(config["eval"].get("bootstrap_seed", 42))

    start = time.time()
    payload: Dict[str, Any] = {
        "experiment": exp,
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "best_validation_checkpoint_epoch": ckpt.get("best", {}).get("epoch"),
        "subset_file": config["data"]["subset_file"],
        "model_name": config["model"]["name"],
        "pretrained": bool(config["model"].get("pretrained", False)),
        "device": str(device),
        "amp": bool(amp),
        "class_names": class_names,
        "splits": {},
    }

    for split in evaluated_splits:
        dataset = make_datasets(config, split, positions=None, train=False)
        loader = make_dataloader(dataset, config, train=False, batch_size=eval_batch_size, seed=seed)
        y_true, y_prob, indices = predict_loader(model, loader, device, amp=amp)
        metrics = compute_metrics(y_true, y_prob, class_names, bootstrap_n=bootstrap_n,
                                  bootstrap_seed=bootstrap_seed)
        metrics["class_counts"] = {name: int((y_true == k).sum()) for k, name in enumerate(class_names)}
        payload["splits"][split] = metrics
        save_predictions_csv(run_dir / f"predictions_{split}.csv", y_true, y_prob, indices, class_names)
        cm = np.asarray(metrics["confusion_matrix"])
        plot_confusion(cm, class_names, f"{exp} seed{seed} — {split} (counts)",
                       run_dir / f"confusion_matrix_{split}.png", normalize=False)
        plot_confusion(cm, class_names, f"{exp} seed{seed} — {split} (row-normalized)",
                       run_dir / f"confusion_matrix_{split}_normalized.png", normalize=True)
        metrics["confusion_matrix_figure"] = str(run_dir / f"confusion_matrix_{split}.png")
        metrics["confusion_matrix_normalized_figure"] = str(run_dir / f"confusion_matrix_{split}_normalized.png")
        logger.info(
            "%s: n=%d acc=%.4f bal_acc=%.4f macro_f1=%.4f weighted_f1=%.4f macro_auroc=%.4f ece=%.4f",
            split, metrics["n_samples"], metrics["accuracy"], metrics["balanced_accuracy"],
            metrics["macro_f1"], metrics["weighted_f1"], metrics["macro_auroc"], metrics["ece"],
        )
        logger.info("%s per-class: %s", split, json.dumps(metrics["per_class"]))
        logger.info("%s confusion matrix (rows=true): %s", split, metrics["confusion_matrix"])

    wall = time.time() - start
    payload["wall_clock_seconds"] = wall
    payload["versions"] = environment_info(device)["versions"]
    payload["git_commit"] = git_commit()
    save_json(payload, run_dir / "metrics.json")
    # Flat copies for convenience.
    for split in evaluated_splits:
        save_json(payload["splits"][split], run_dir / f"metrics_{split}.json")

    if int(seed) == int(config.get("seed", 42)):
        shutil.copyfile(run_dir / "metrics.json", experiment_results_dir(config, exp) / "metrics.json")

    if "test" in evaluated_splits:
        save_json({
            "test_evaluated": True,
            "timestamp": environment_info(device)["timestamp"],
            "checkpoint": str(ckpt_path),
            "split": "test",
            "note": "Test evaluated once after all training/selection decisions were frozen.",
            "metrics_file": str(run_dir / "metrics.json"),
        }, marker)
    logger.info("Evaluation complete in %.1f s", wall)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--force", action="store_true", help="Override the one-time test guard.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None, help="Override the evaluation batch size.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    payload = evaluate_checkpoint(
        args.checkpoint, args.splits, force=args.force, device_override=args.device,
        batch_size_override=args.batch_size, limit_batches=args.limit_batches,
    )
    summary = {
        split: {
            "n": m["n_samples"], "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
            "macro_auroc": m["macro_auroc"], "macro_f1_ci95": m["macro_f1_bootstrap_ci95"],
        }
        for split, m in payload["splits"].items()
    }
    print(json.dumps(to_jsonable(summary), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

