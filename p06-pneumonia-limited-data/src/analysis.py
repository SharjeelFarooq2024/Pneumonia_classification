"""Data exploration, result aggregation, and error/robustness analysis.

All quantities are computed from saved result files; nothing is retrained here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

from src.models.proposed import build_model, get_gradcam_target_layer
from src.utils import (
    compute_metrics,
    experiment_results_dir,
    get_device,
    load_json,
    resolve_path,
    save_json,
    set_seed,
    setup_logger,
)

MAIN_EXPERIMENTS = ("baseline", "proposed")
CLASS_ORDER = ("NORMAL", "BACTERIA", "VIRUS")


# ---------------------------------------------------------------------------
# Data exploration
# ---------------------------------------------------------------------------
def explore_data(config: Mapping[str, Any]) -> None:
    fig_dir = resolve_path(config["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(config["data"]["class_names"])
    
    csv_file = fig_dir / "class_distribution.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print("Class distribution summary:")
        print(df.to_string(index=False))
        return

    from src.data.dataset import get_split_global_indices, get_split_labels
    split_labels = {s: get_split_labels(config, s) for s in ("train", "val", "test")}
    with open(resolve_path(config["data"]["subset_file"]), "r", encoding="utf-8") as f:
        subset = json.load(f)
    subset_labels = np.array([split_labels["train"][p] for p in subset["positions"]])

    rows: List[Dict[str, Any]] = []
    for name, labels in [("full_train", split_labels["train"]),
                         ("train25", subset_labels),
                         ("val", split_labels["val"]),
                         ("test", split_labels["test"])]:
        counts = {c: int((labels == k).sum()) for k, c in enumerate(class_names)}
        for c, n in counts.items():
            rows.append({"set": name, "class": c, "count": n, "proportion": n / max(len(labels), 1)})
    df = pd.DataFrame(rows)
    df.to_csv(fig_dir / "class_distribution.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    ax = sns.barplot(data=df, x="set", y="count", hue="class", hue_order=class_names)
    for container in ax.containers:
        ax.bar_label(container, fontsize=7)
    plt.title("Class distribution: full train, official train25 subset, val, test")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(fig_dir / "class_distribution.png", dpi=150)
    plt.close()

    # Sample image grid, 5 per class from the full train split.
    from src.data.dataset import PneumoniaDataset
    from torchvision import transforms

    vis_transform = transforms.Compose([
        transforms.Resize((int(config["data"]["image_size"]), int(config["data"]["image_size"]))),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train_ds = PneumoniaDataset(config, "train", positions=None, transform=vis_transform)
    rng = np.random.default_rng(42)
    per_class = 5
    fig, axes = plt.subplots(len(class_names), per_class, figsize=(2.0 * per_class, 2.2 * len(class_names)))
    for row_idx, cls_name in enumerate(class_names):
        cls_int = class_names.index(cls_name)
        candidates = np.flatnonzero(train_ds.labels == cls_int)
        chosen = rng.choice(candidates, size=min(per_class, len(candidates)), replace=False)
        for col_idx in range(per_class):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx < len(chosen):
                image, label, _ = train_ds[int(chosen[col_idx])]
                ax.imshow(image.squeeze(0).numpy(), cmap="gray")
                ax.set_title(cls_name, fontsize=8)
    fig.suptitle("Sample training images per class (grayscale)")
    fig.tight_layout()
    fig.savefig(fig_dir / "samples_by_class.png", dpi=150)
    plt.close(fig)

    payload = {
        "splits": {
            name: {"n": int(len(labels)),
                   "class_counts": {c: int((labels == k).sum()) for k, c in enumerate(class_names)}}
            for name, labels in [("full_train", split_labels["train"]), ("train25", subset_labels),
                                 ("val", split_labels["val"]), ("test", split_labels["test"])]
        },
        "subset": {k: subset[k] for k in ("name", "fraction", "seed", "n_subset", "class_counts",
                                          "class_proportions", "max_abs_class_proportion_diff")},
        "train25_class_proportions_match": subset["max_abs_class_proportion_diff"] < 0.01,
    }
    save_json(payload, fig_dir / "class_distribution.json")
    print(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _metrics_row(experiment: str, seed: int, split: str, m: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "experiment": experiment, "seed": int(seed), "split": split,
        "n": int(m.get("n_samples", 0)),
        "accuracy": m.get("accuracy"), "balanced_accuracy": m.get("balanced_accuracy"),
        "macro_f1": m.get("macro_f1"), "weighted_f1": m.get("weighted_f1"),
        "macro_auroc": m.get("macro_auroc"), "ece": m.get("ece"),
        "macro_f1_ci_low": (m.get("macro_f1_bootstrap_ci95") or [None, None])[0],
        "macro_f1_ci_high": (m.get("macro_f1_bootstrap_ci95") or [None, None])[1],
    }
    per_class = m.get("per_class", {})
    for c in CLASS_ORDER:
        stats = per_class.get(c, {})
        row[f"precision_{c}"] = stats.get("precision")
        row[f"recall_{c}"] = stats.get("recall")
        row[f"f1_{c}"] = stats.get("f1")
        row[f"auroc_{c}"] = stats.get("auroc_ovr")
        row[f"support_{c}"] = stats.get("support")
    return row


def scan_result_rows(config: Mapping[str, Any]) -> pd.DataFrame:
    results_root = resolve_path(config["paths"]["results"])
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(results_root.glob("**/seed*/metrics.json")):
        if "smoke" in metrics_path.parts:
            continue
        payload = load_json(metrics_path)
        exp = payload.get("experiment", metrics_path.parents[1].name)
        seed = int(payload.get("seed", 42))
        for split, m in payload.get("splits", {}).items():
            rows.append(_metrics_row(exp, seed, split, m))
    for val_path in sorted(results_root.glob("**/seed*/val_metrics.json")):
        if "smoke" in val_path.parts:
            continue
        if (val_path.parent / "metrics.json").exists():
            continue  # already covered, and metrics.json includes val
        payload = load_json(val_path)
        exp = val_path.parents[1].name
        seed = int(val_path.parent.name.replace("seed", ""))
        rows.append(_metrics_row(exp, seed, "val", payload))
    return pd.DataFrame(rows)


def _mean_std(df: pd.DataFrame, exp: str, split: str, col: str) -> str:
    vals = df[(df.experiment == exp) & (df.split == split)][col].dropna().astype(float)
    if len(vals) == 0:
        return "—"
    if len(vals) == 1:
        return f"{vals.iloc[0]:.4f}"
    return f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"


def aggregate_results(config: Mapping[str, Any]) -> pd.DataFrame:
    df = scan_result_rows(config)
    out = resolve_path(config["paths"]["results"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary_table.csv", index=False)

    lines: List[str] = ["# Result summary", "",
                        "All values are computed from saved predictions/metrics files; no hand-edited numbers.", ""]
    lines += ["## Main results (test, mean ± std over seeds)", "",
              "| Model | N seeds | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 | Macro-AUROC | ECE |",
              "|---|---|---|---|---|---|---|---|"]
    for exp in MAIN_EXPERIMENTS:
        sub = df[(df.experiment == exp) & (df.split == "test")]
        if len(sub) == 0:
            lines.append(f"| {exp} | 0 | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {exp} | {sub.seed.nunique()} | {_mean_std(df, exp, 'test', 'accuracy')} | "
            f"{_mean_std(df, exp, 'test', 'balanced_accuracy')} | {_mean_std(df, exp, 'test', 'macro_f1')} | "
            f"{_mean_std(df, exp, 'test', 'weighted_f1')} | {_mean_std(df, exp, 'test', 'macro_auroc')} | "
            f"{_mean_std(df, exp, 'test', 'ece')} |"
        )

    lines += ["", "## Per-class test metrics (mean ± std over seeds)", "",
              "| Model | Class | Precision | Recall | F1 | AUROC (OvR) | Support |",
              "|---|---|---|---|---|---|---|"]
    for exp in MAIN_EXPERIMENTS:
        for c in CLASS_ORDER:
            sub = df[(df.experiment == exp) & (df.split == "test")]
            if len(sub) == 0:
                continue
            support = sub[f"support_{c}"].dropna()
            support_text = f"{int(support.iloc[0])}" if len(support) else "—"
            lines.append(
                f"| {exp} | {c} | {_mean_std(df, exp, 'test', f'precision_{c}')} | "
                f"{_mean_std(df, exp, 'test', f'recall_{c}')} | {_mean_std(df, exp, 'test', f'f1_{c}')} | "
                f"{_mean_std(df, exp, 'test', f'auroc_{c}')} | {support_text} |"
            )

    ablations = df[df.experiment.str.startswith("ablation_")].copy()
    lines += ["", "## Controlled ablations (validation macro-F1; one seed unless noted)", "",
              "| Experiment | Seeds | Val macro-F1 | Val balanced acc. | Val macro-AUROC |",
              "|---|---|---|---|---|"]
    if len(ablations):
        for exp in sorted(ablations.experiment.unique()):
            sub = ablations[(ablations.experiment == exp) & (ablations.split == "val")]
            lines.append(
                f"| {exp} | {sub.seed.nunique()} | {_mean_std(ablations, exp, 'val', 'macro_f1')} | "
                f"{_mean_std(ablations, exp, 'val', 'balanced_accuracy')} | "
                f"{_mean_std(ablations, exp, 'val', 'macro_auroc')} |"
            )
    else:
        lines.append("| (none yet) | — | — | — | — |")

    dataeff = df[df.experiment.str.contains("_dataeff_", regex=False)].copy()
    lines += ["", "## Data-efficiency curve (test macro-F1; supplementary — 100% is context only)", "",
              "| Model | Train fraction | Test macro-F1 | Seeds |", "|---|---|---|---|"]
    if len(dataeff):
        for exp in sorted(dataeff.experiment.unique()):
            frac = exp.rsplit("_", 1)[-1] + "%"
            model = "baseline" if exp.startswith("baseline") else ("proposed" if exp.startswith("proposed") else exp)
            sub = dataeff[(dataeff.experiment == exp) & (dataeff.split == "test")]
            lines.append(f"| {model} | {frac} | {_mean_std(dataeff, exp, 'test', 'macro_f1')} | {sub.seed.nunique()} |")
        for exp in MAIN_EXPERIMENTS:
            if (df.experiment == exp).any() and (df[(df.experiment == exp) & (df.split == 'test')]).shape[0]:
                lines.append(f"| {exp} | 25% | {_mean_std(df, exp, 'test', 'macro_f1')} | "
                             f"{df[(df.experiment == exp) & (df.split == 'test')].seed.nunique()} |")
    else:
        lines.append("| (none yet) | — | — | — |")

    test_rows = df[(df.split == "test") & (df.experiment.isin(MAIN_EXPERIMENTS))]
    lines += ["", "## Test set sizes per class", ""]
    if len(test_rows):
        for c in CLASS_ORDER:
            vals = test_rows[f"support_{c}"].dropna().astype(int)
            if len(vals):
                lines.append(f"- {c}: {int(vals.iloc[0])}")
        lines.append(f"- Total: {int(test_rows['n'].iloc[0])}")
    else:
        lines.append("- Not yet evaluated.")

    lines += ["", "## Notes", "",
              "- Baseline uses random initialization, unweighted CE, Adam 1e-3, 20 epochs, no augmentation, no label smoothing.",
              "- Proposed uses ImageNet-pretrained ResNet-18 with staged fine-tuning plus the components selected on validation.",
              "- The 25% stratified subset (seed 42) is identical across every experiment and was created once.",
              "- Test was evaluated once per final model, after training and model selection were frozen.",
              "- All reported main runs use batch size 16.",
              "- Ablations and data-efficiency models were not completed on the CPU host; no numbers are claimed for them.",
              ""]
    (out / "summary_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out / 'summary_table.csv'} and {out / 'summary_table.md'}")
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_training_curves(config: Mapping[str, Any]) -> None:
    fig_dir = resolve_path(config["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    for exp in MAIN_EXPERIMENTS:
        seed_dirs = sorted(experiment_results_dir(config, exp).glob("seed*/history.csv"))
        if not seed_dirs:
            continue
        frames = []
        for path in seed_dirs:
            h = pd.read_csv(path)
            h["seed"] = int(path.parent.name.replace("seed", ""))
            frames.append(h)
        hist = pd.concat(frames, ignore_index=True)
        agg = hist.groupby("epoch").agg(
            train_loss=("train_loss", "mean"), train_loss_std=("train_loss", "std"),
            val_loss=("val_loss", "mean"), val_loss_std=("val_loss", "std"),
            val_macro_f1=("val_macro_f1", "mean"), val_macro_f1_std=("val_macro_f1", "std"),
            val_accuracy=("val_accuracy", "mean"), val_accuracy_std=("val_accuracy", "std"),
        ).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(agg.epoch, agg.train_loss, label="train")
        axes[0].fill_between(agg.epoch, agg.train_loss - agg.train_loss_std.fillna(0),
                             agg.train_loss + agg.train_loss_std.fillna(0), alpha=0.2)
        axes[0].plot(agg.epoch, agg.val_loss, label="validation")
        axes[0].fill_between(agg.epoch, agg.val_loss - agg.val_loss_std.fillna(0),
                             agg.val_loss + agg.val_loss_std.fillna(0), alpha=0.2)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss"); axes[0].legend()
        axes[0].set_title(f"{exp}: loss")
        axes[1].plot(agg.epoch, agg.val_macro_f1, color="darkgreen", label="val macro-F1")
        axes[1].fill_between(agg.epoch, agg.val_macro_f1 - agg.val_macro_f1_std.fillna(0),
                             agg.val_macro_f1 + agg.val_macro_f1_std.fillna(0), alpha=0.2, color="darkgreen")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1"); axes[1].legend()
        axes[1].set_title(f"{exp}: validation macro-F1")
        fig.tight_layout()
        fig.savefig(fig_dir / f"training_curves_{exp}.png", dpi=150)
        plt.close(fig)

    # Side-by-side validation macro-F1 for both models.
    plt.figure(figsize=(7, 4))
    for exp, color in [("baseline", "#4c72b0"), ("proposed", "#dd8452")]:
        seed_dirs = sorted(experiment_results_dir(config, exp).glob("seed*/history.csv"))
        if not seed_dirs:
            continue
        frames = [pd.read_csv(p) for p in seed_dirs]
        hist = pd.concat(frames, ignore_index=True)
        agg = hist.groupby("epoch")["val_macro_f1"].agg(["mean", "std"]).reset_index()
        plt.plot(agg.epoch, agg["mean"], label=exp, color=color)
        plt.fill_between(agg.epoch, agg["mean"] - agg["std"].fillna(0), agg["mean"] + agg["std"].fillna(0),
                         alpha=0.2, color=color)
    plt.xlabel("Epoch"); plt.ylabel("Validation macro-F1"); plt.legend()
    plt.title("Validation macro-F1 during training")
    plt.tight_layout()
    plt.savefig(fig_dir / "val_macro_f1_comparison.png", dpi=150)
    plt.close()


def _predictions_df(config: Mapping[str, Any], exp: str, seed: int, split: str = "test") -> Optional[pd.DataFrame]:
    path = experiment_results_dir(config, exp) / f"seed{seed}" / f"predictions_{split}.csv"
    return pd.read_csv(path) if path.exists() else None


def plot_confusions(config: Mapping[str, Any], seed: int = 42) -> None:
    fig_dir = resolve_path(config["paths"]["figures"])
    class_names = list(config["data"]["class_names"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    any_plot = False
    for row, exp in enumerate(MAIN_EXPERIMENTS):
        df = _predictions_df(config, exp, seed, "test")
        if df is None:
            continue
        y_true = df["true_label"].to_numpy()
        y_pred = df["predicted_label"].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        for col, (mat, fmt, title) in enumerate([
            (cm, "d", f"{exp}: counts"),
            (cm_norm, ".2f", f"{exp}: row-normalized"),
        ]):
            ax = axes[row, col]
            sns.heatmap(mat, annot=True, fmt=fmt, cmap="Blues", cbar=False,
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        any_plot = True
    if any_plot:
        fig.suptitle(f"Test confusion matrices (seed {seed})")
        fig.tight_layout()
        fig.savefig(fig_dir / "confusion_matrices_baseline_vs_proposed.png", dpi=150)
    plt.close(fig)


def bacterial_viral_analysis(config: Mapping[str, Any], seed: int = 42) -> None:
    fig_dir = resolve_path(config["paths"]["figures"])
    class_names = list(config["data"]["class_names"])
    bi, vi = class_names.index("BACTERIA"), class_names.index("VIRUS")
    payload: Dict[str, Any] = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, exp in zip(axes, MAIN_EXPERIMENTS):
        df = _predictions_df(config, exp, seed, "test")
        if df is None:
            ax.axis("off")
            continue
        y_true = df["true_label"].to_numpy()
        y_pred = df["predicted_label"].to_numpy()
        micro_true = np.where(np.isin(y_true, [bi, vi]), y_true, -1)
        micro_pred = np.where(np.isin(y_pred, [bi, vi]), y_pred, -1)
        mask = micro_true >= 0
        cm = confusion_matrix(micro_true[mask], micro_pred[mask], labels=[bi, vi])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False,
                    xticklabels=["BACTERIA", "VIRUS"], yticklabels=["BACTERIA", "VIRUS"], ax=ax)
        ax.set_title(f"{exp}: bacterial vs viral")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        payload[exp] = {
            "bacterial_true": int(cm[0].sum()), "viral_true": int(cm[1].sum()),
            "bacterial_to_viral": int(cm[0, 1]), "viral_to_bacterial": int(cm[1, 0]),
            "bacterial_to_viral_rate": float(cm[0, 1] / max(cm[0].sum(), 1)),
            "viral_to_bacterial_rate": float(cm[1, 0] / max(cm[1].sum(), 1)),
        }
    fig.suptitle(f"Bacterial vs viral confusion on test (seed {seed})")
    fig.tight_layout()
    fig.savefig(fig_dir / "bacterial_viral_confusion.png", dpi=150)
    plt.close(fig)
    save_json(payload, resolve_path(config["paths"]["results"]) / "bacterial_viral_confusion.json")


def plot_per_class_f1(config: Mapping[str, Any], seed: int = 42) -> None:
    fig_dir = resolve_path(config["paths"]["figures"])
    class_names = list(config["data"]["class_names"])
    rows = []
    for exp in MAIN_EXPERIMENTS:
        path = experiment_results_dir(config, exp) / f"seed{seed}" / "metrics.json"
        if not path.exists():
            continue
        payload = load_json(path)
        per_class = payload["splits"]["test"]["per_class"]
        for c in class_names:
            rows.append({"model": exp, "class": c, "f1": per_class[c]["f1"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    plt.figure(figsize=(6.5, 4))
    sns.barplot(data=df, x="class", y="f1", hue="model", order=class_names)
    plt.ylim(0, 1); plt.ylabel("Test F1"); plt.title(f"Per-class test F1 (seed {seed})")
    plt.tight_layout()
    plt.savefig(fig_dir / "per_class_f1.png", dpi=150)
    plt.close()


def plot_reliability(config: Mapping[str, Any], seed: int = 42) -> None:
    fig_dir = resolve_path(config["paths"]["figures"])
    plt.figure(figsize=(6, 4.5))
    any_plot = False
    for exp, color in [("baseline", "#4c72b0"), ("proposed", "#dd8452")]:
        path = experiment_results_dir(config, exp) / f"seed{seed}" / "metrics.json"
        if not path.exists():
            continue
        payload = load_json(path)
        bins = payload["splits"]["test"].get("reliability_bins", [])
        if not bins:
            continue
        xs = [b["confidence"] for b in bins]
        ys = [b["accuracy"] for b in bins]
        plt.plot(xs, ys, marker="o", label=f"{exp} (ECE={payload['splits']['test']['ece']:.3f})", color=color)
        any_plot = True
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    plt.xlabel("Mean predicted confidence"); plt.ylabel("Empirical accuracy")
    plt.title(f"Reliability diagram on test (seed {seed})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "reliability_diagram.png", dpi=150)
    plt.close()


def plot_data_efficiency(config: Mapping[str, Any]) -> None:
    fig_dir = resolve_path(config["paths"]["figures"])
    df = scan_result_rows(config)
    test = df[df.split == "test"]
    fractions = {"10": 10, "25": 25, "100": 100}

    def gather(prefix: str) -> List[Tuple[int, float, float, float]]:
        points = []
        for exp in test.experiment.unique():
            if not exp.startswith(prefix):
                continue
            if "_dataeff_" in exp:
                frac = exp.rsplit("_", 1)[-1]
            elif exp == prefix:
                frac = "25"
            else:
                continue
            if frac not in fractions:
                continue
            vals = test[(test.experiment == exp) & (test.split == "test")]["macro_f1"].dropna().astype(float)
            if len(vals):
                points.append((fractions[frac], vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0, len(vals)))
        return sorted(points)

    plt.figure(figsize=(6.5, 4.2))
    any_points = False
    for prefix, color in [("baseline", "#4c72b0"), ("proposed", "#dd8452")]:
        points = gather(prefix)
        if len(points) < 2:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        errs = [p[2] for p in points]
        plt.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=prefix, color=color)
        any_points = True
    if any_points:
        plt.xscale("log")
        plt.xticks([10, 25, 100], ["10%", "25% (official)", "100% (context)"])
        plt.xlabel("Fraction of the training split used")
        plt.ylabel("Test macro-F1")
        plt.title("Data efficiency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "data_efficiency_curve.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Bootstrap comparison
# ---------------------------------------------------------------------------
def paired_bootstrap(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    labels = list(range(probs_a.shape[1]))
    pred_a, pred_b = probs_a.argmax(1), probs_b.argmax(1)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    a_scores = np.empty(n_boot)
    b_scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = f1_score(y_true[idx], pred_a[idx], labels=labels, average="macro", zero_division=0)
        b = f1_score(y_true[idx], pred_b[idx], labels=labels, average="macro", zero_division=0)
        a_scores[i], b_scores[i], diffs[i] = a, b, b - a
    return {
        "n_boot": int(n_boot),
        "baseline_macro_f1_ci95": [float(np.percentile(a_scores, 2.5)), float(np.percentile(a_scores, 97.5))],
        "proposed_macro_f1_ci95": [float(np.percentile(b_scores, 2.5)), float(np.percentile(b_scores, 97.5))],
        "difference_mean": float(diffs.mean()),
        "difference_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "probability_proposed_better": float((diffs > 0).mean()),
        "observed_baseline_macro_f1": float(f1_score(y_true, pred_a, labels=labels, average="macro", zero_division=0)),
        "observed_proposed_macro_f1": float(f1_score(y_true, pred_b, labels=labels, average="macro", zero_division=0)),
    }


def bootstrap_comparison(config: Mapping[str, Any]) -> None:
    out = resolve_path(config["paths"]["results"])
    fig_dir = resolve_path(config["paths"]["figures"])
    class_names = list(config["data"]["class_names"])
    per_seed: Dict[str, Any] = {}
    for seed in (42, 43, 44):
        a = _predictions_df(config, "baseline", seed, "test")
        b = _predictions_df(config, "proposed", seed, "test")
        if a is None or b is None:
            continue
        merged = pd.merge(a, b, on="image_index", suffixes=("_a", "_b"))
        if len(merged) == 0:
            continue
        y_true = merged["true_label_a"].to_numpy()
        probs_a = merged[[f"prob_{c}_a" for c in class_names]].to_numpy()
        probs_b = merged[[f"prob_{c}_b" for c in class_names]].to_numpy()
        per_seed[str(seed)] = paired_bootstrap(y_true, probs_a, probs_b, n_boot=2000, seed=seed)
    if not per_seed:
        print("No paired baseline/proposed test predictions available for bootstrap.")
        return
    diffs = [v["difference_mean"] for v in per_seed.values()]
    summary = {
        "per_seed": per_seed,
        "mean_difference_across_seeds": float(np.mean(diffs)),
        "note": "Paired bootstrap over test images; positive difference favours the proposed model.",
    }
    save_json(summary, out / "bootstrap_comparison.json")

    # Histogram for seed 42.
    if "42" in per_seed:
        a = _predictions_df(config, "baseline", 42, "test")
        b = _predictions_df(config, "proposed", 42, "test")
        merged = pd.merge(a, b, on="image_index", suffixes=("_a", "_b"))
        y_true = merged["true_label_a"].to_numpy()
        probs_a = merged[[f"prob_{c}_a" for c in class_names]].to_numpy()
        probs_b = merged[[f"prob_{c}_b" for c in class_names]].to_numpy()
        labels = list(range(len(class_names)))
        pred_a, pred_b = probs_a.argmax(1), probs_b.argmax(1)
        rng = np.random.default_rng(42)
        diffs_boot = []
        for _ in range(2000):
            idx = rng.integers(0, len(y_true), size=len(y_true))
            da = f1_score(y_true[idx], pred_a[idx], labels=labels, average="macro", zero_division=0)
            db = f1_score(y_true[idx], pred_b[idx], labels=labels, average="macro", zero_division=0)
            diffs_boot.append(db - da)
        plt.figure(figsize=(6.5, 4))
        plt.hist(diffs_boot, bins=40, color="#dd8452", alpha=0.8)
        plt.axvline(0, color="black", linestyle="--", label="no difference")
        plt.axvline(np.percentile(diffs_boot, 2.5), color="red", linestyle=":", label="95% CI")
        plt.axvline(np.percentile(diffs_boot, 97.5), color="red", linestyle=":")
        plt.xlabel("Macro-F1 difference (proposed − baseline)")
        plt.ylabel("Bootstrap count")
        plt.title("Paired bootstrap over test images (seed 42)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "bootstrap_macro_f1_difference.png", dpi=150)
        plt.close()
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Error analysis: misclassified grid and Grad-CAM
# ---------------------------------------------------------------------------
def _load_grayscale(config: Mapping[str, Any], image_index: int) -> np.ndarray:
    root = resolve_path(config["data"]["root"]) / config["data"].get("dataset_id", "pneumonia")
    path = root / "images" / f"{int(image_index):06d}.tiff"
    return np.asarray(Image.open(path).convert("L"))


def _tensor_from_gray(config: Mapping[str, Any], image: np.ndarray) -> torch.Tensor:
    from src.data.dataset import build_transform
    pil = Image.fromarray(image)
    return build_transform(config, train=False)(pil).unsqueeze(0)


def plot_misclassified(config: Mapping[str, Any], seed: int = 42, max_images: int = 16) -> None:
    df = _predictions_df(config, "proposed", seed, "test")
    if df is None:
        print("No proposed test predictions for misclassified grid.")
        return
    fig_dir = resolve_path(config["paths"]["figures"])
    wrong = df[df.true_label != df.predicted_label].sort_values("confidence", ascending=False).head(max_images)
    if len(wrong) == 0:
        print("No misclassified test images.")
        return
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    class_names = list(config["data"]["class_names"])
    for ax, (_, row) in zip(axes.flat, wrong.iterrows()):
        image = _load_grayscale(config, int(row["image_index"]))
        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"true: {class_names[int(row['true_label'])]}\n"
            f"pred: {class_names[int(row['predicted_label'])]} ({row['confidence']:.2f})",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle(f"Proposed model: misclassified test images (seed {seed})")
    fig.tight_layout()
    fig.savefig(fig_dir / "proposed_misclassified_test.png", dpi=150)
    plt.close(fig)


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd = target_layer.register_forward_hook(self._forward_hook)
        self._bwd = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze(0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

    def close(self) -> None:
        self._fwd.remove()
        self._bwd.remove()


def plot_gradcam(config: Mapping[str, Any], seed: int = 42, per_group: int = 4) -> None:
    exp = "proposed"
    ckpt_path = resolve_path(config["paths"]["checkpoints"]) / f"{exp}_seed{seed}.pt"
    df = _predictions_df(config, exp, seed, "test")
    if not ckpt_path.exists() or df is None:
        print("Proposed checkpoint/predictions missing; skipping Grad-CAM.")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    eval_config = {**ckpt["config"], "model": {**ckpt["config"].get("model", {}), "pretrained": False}}
    model = build_model(eval_config)
    model.load_state_dict(ckpt["model_state"])
    device = get_device("auto")
    model.to(device).eval()
    cam_engine = GradCAM(model, get_gradcam_target_layer(model))
    try:
        correct = df[df.true_label == df.predicted_label].sort_values("confidence", ascending=False).head(per_group)
        wrong = df[df.true_label != df.predicted_label].sort_values("confidence", ascending=False).head(per_group)
        selected = pd.concat([correct.assign(group="correct"), wrong.assign(group="incorrect")])
        cols = per_group
        rows = 2
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.3 * rows))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")
        class_names = list(config["data"]["class_names"])
        for ax, (_, row) in zip(axes.flat, selected.iterrows()):
            image = _load_grayscale(config, int(row["image_index"]))
            tensor = _tensor_from_gray(config, image).to(device)
            cam = cam_engine(tensor, int(row["predicted_label"]))
            cam_resized = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (image.shape[1], image.shape[0]), resample=Image.BILINEAR)) / 255.0
            ax.imshow(image, cmap="gray")
            ax.imshow(cam_resized, cmap="jet", alpha=0.45)
            ax.set_title(
                f"{row['group']} | true {class_names[int(row['true_label'])]} | "
                f"pred {class_names[int(row['predicted_label'])]} ({row['confidence']:.2f})",
                fontsize=7,
            )
            ax.axis("off")
        fig.suptitle(f"Grad-CAM for the proposed ResNet-18 (test, seed {seed})")
        fig.tight_layout()
        fig.savefig(resolve_path(config["paths"]["figures"]) / "gradcam_proposed_test.png", dpi=150)
        plt.close(fig)
    finally:
        cam_engine.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analysis utilities.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=42)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("explore")
    sub.add_parser("aggregate")
    sub.add_parser("figures")
    sub.add_parser("bootstrap")
    sub.add_parser("gradcam")
    sub.add_parser("all")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    from src.utils import load_config

    args = parse_args(argv)
    config = load_config(args.config)
    logger = setup_logger("analysis")
    if args.command in ("explore", "all"):
        explore_data(config)
    if args.command in ("aggregate", "all"):
        df = aggregate_results(config)
        logger.info("Aggregated %d result rows.", len(df))
    if args.command in ("figures", "all"):
        plot_training_curves(config)
        plot_confusions(config, seed=args.seed)
        bacterial_viral_analysis(config, seed=args.seed)
        plot_per_class_f1(config, seed=args.seed)
        plot_reliability(config, seed=args.seed)
        plot_data_efficiency(config)
        plot_misclassified(config, seed=args.seed)
    if args.command in ("bootstrap", "all"):
        bootstrap_comparison(config)
    if args.command in ("gradcam", "all"):
        plot_gradcam(config, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
