"""Generate honest, reproducible ablation study and data-efficiency framework artifacts and figures.

Adheres strictly to academic integrity and logs/DECISIONS.md:
- Only verified, real experimental results are reported as empirical data.
- Unexecuted multi-epoch ablation sweeps (A1-A4) and 10%/100% scaling runs are
  formally documented as experimental designs and hypotheses, not fabricated numbers.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def generate_ablation_artifacts():
    ablations_dir = ROOT / "results" / "ablations"
    ablations_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Honest Controlled Ablations Design Matrix
    # Real anchor results are drawn from saved metrics; planned intermediate runs are documented.
    ablations_data = [
        {
            "experiment": "Baseline CNN (Scratch)",
            "category": "A1: Backbone Regime",
            "config_file": "configs/baseline.yaml",
            "val_macro_f1": "0.5994 (Real)",
            "val_balanced_acc": "0.6138 (Real)",
            "status": "Trained & Evaluated",
            "hypothesis": "Scratch 3-block CNN (93K params) serves as official reference baseline; suffers majority class collapse."
        },
        {
            "experiment": "ResNet-18 (Scratch)",
            "category": "A1: Backbone Regime",
            "config_file": "configs/ablations/a1_resnet18_scratch.yaml",
            "val_macro_f1": "— (Unperformed)",
            "val_balanced_acc": "— (Unperformed)",
            "status": "Designed (Hypothesis)",
            "hypothesis": "High parameter count (11.2M) without pretraining severely overfits 1,103 training images."
        },
        {
            "experiment": "ResNet-18 (Frozen Backbone)",
            "category": "A1: Backbone Regime",
            "config_file": "configs/ablations/a1_resnet18_frozen.yaml",
            "val_macro_f1": "— (Unperformed)",
            "val_balanced_acc": "— (Unperformed)",
            "status": "Designed (Hypothesis)",
            "hypothesis": "Linear probe tests generalizability of frozen ImageNet features without domain fine-tuning."
        },
        {
            "experiment": "ResNet-18 (Full Fine-Tuning)",
            "category": "A1: Backbone Regime",
            "config_file": "configs/ablations/a1_resnet18_full.yaml",
            "val_macro_f1": "— (Unperformed)",
            "val_balanced_acc": "— (Unperformed)",
            "status": "Designed (Hypothesis)",
            "hypothesis": "End-to-end unfreezing from epoch 1 causes catastrophic forgetting in early convolutional layers."
        },
        {
            "experiment": "ResNet-18 (Staged Fine-Tuning)",
            "category": "A1: Backbone Regime",
            "config_file": "configs/ablations/a1_resnet18_staged.yaml",
            "val_macro_f1": "— (Unperformed)",
            "val_balanced_acc": "— (Unperformed)",
            "status": "Designed (Hypothesis)",
            "hypothesis": "Progressive unfreezing (head -> layer4 -> full) protects foundational edge filters while adapting semantics."
        },
        {
            "experiment": "Staged ResNet-18 (No Augmentation)",
            "category": "A2: Augmentation",
            "config_file": "configs/ablations/a4_staged_noaug_weighted.yaml",
            "val_macro_f1": "— (Unperformed)",
            "val_balanced_acc": "— (Unperformed)",
            "status": "Designed (Hypothesis)",
            "hypothesis": "Isolates the regularization benefit of affine transforms on small pediatric datasets."
        },
        {
            "experiment": "Staged ResNet-18 (With Safe Aug)",
            "category": "A2: Augmentation",
            "config_file": "configs/ablations/a2_staged_aug.yaml",
            "val_macro_f1": "— (Unperformed)",
            "val_balanced_acc": "— (Unperformed)",
            "status": "Designed (Hypothesis)",
            "hypothesis": "Tests rotation/translation/scale while strictly avoiding anatomically invalid horizontal flips."
        },
        {
            "experiment": "Staged ResNet-18 (CE, No Smoothing)",
            "category": "A3: Loss Regularization",
            "config_file": "configs/ablations/a2_staged_aug.yaml",
            "val_macro_f1": "— (Unperformed)",
            "val_balanced_acc": "— (Unperformed)",
            "status": "Designed (Hypothesis)",
            "hypothesis": "Hard cross-entropy targets lead to overconfident probability calibration."
        },
        {
            "experiment": "Staged ResNet-18 (Label Smoothing 0.1)",
            "category": "A3: Loss Regularization",
            "config_file": "configs/ablations/a3_staged_aug_ls.yaml",
            "val_macro_f1": "— (Unperformed)",
            "val_balanced_acc": "— (Unperformed)",
            "status": "Designed (Hypothesis)",
            "hypothesis": "Uniform label smoothing prevents extreme logit penalties, lowering ECE."
        },
        {
            "experiment": "Staged ResNet-18 (Mixup alpha=0.2)",
            "category": "A3: Loss Regularization",
            "config_file": "configs/ablations/a3_staged_aug_ls_mixup.yaml",
            "val_macro_f1": "— (Unperformed)",
            "val_balanced_acc": "— (Unperformed)",
            "status": "Designed (Hypothesis)",
            "hypothesis": "Pixel-space interpolation may blur distinct focal vs diffuse pulmonary boundaries."
        },
        {
            "experiment": "Staged ResNet-18 (Unweighted CE)",
            "category": "A4: Imbalance Handling",
            "config_file": "configs/ablations/a3_staged_aug_ls.yaml",
            "val_macro_f1": "— (Unperformed)",
            "val_balanced_acc": "— (Unperformed)",
            "status": "Designed (Hypothesis)",
            "hypothesis": "Unweighted loss causes predictions to bias toward majority BACTERIA class."
        },
        {
            "experiment": "Staged ResNet-18 (Focal Loss gamma=2)",
            "category": "A4: Imbalance Handling",
            "config_file": "configs/ablations/a4_staged_aug_focal.yaml",
            "val_macro_f1": "— (Unperformed)",
            "val_balanced_acc": "— (Unperformed)",
            "status": "Designed (Hypothesis)",
            "hypothesis": "Focal modulation dynamic down-weights easy normal/bacterial negatives."
        },
        {
            "experiment": "Full Proposed Model (Staged+Aug+LS+Weighted)",
            "category": "Final Integrated Model",
            "config_file": "configs/proposed.yaml",
            "val_macro_f1": "0.8051 (Real)",
            "val_balanced_acc": "0.8142 (Real)",
            "status": "Trained & Evaluated",
            "hypothesis": "Full integrated recipe: staged fine-tuning, safe augmentations, label smoothing, and subset class weighting."
        }
    ]

    df_ablations = pd.DataFrame(ablations_data)
    df_ablations.to_csv(ablations_dir / "ablation_summary.csv", index=False)
    with open(ablations_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(ablations_data, f, indent=2)

    # 2. Honest Data Efficiency Protocol Framework (10%, 25%, 100%)
    data_eff_data = [
        {
            "split_name": "train10_seed42",
            "budget_percent": 10,
            "training_samples": 441,
            "status": "Stratified Subset Verified (Model Training: Future Work)",
            "baseline_test_macro_f1": "— (Unperformed)",
            "proposed_test_macro_f1": "— (Unperformed)",
            "gain": "— (Unperformed)",
            "config_baseline": "configs/ablations/dataeff_baseline_10.yaml",
            "config_proposed": "configs/ablations/dataeff_proposed_10.yaml"
        },
        {
            "split_name": "train25_seed42 (Task P06 Benchmark)",
            "budget_percent": 25,
            "training_samples": 1103,
            "status": "Fully Trained & Evaluated (3 Seeds)",
            "baseline_test_macro_f1": "0.5296 (Seed 42)",
            "proposed_test_macro_f1": "0.8273 (Seed 42)",
            "gain": "+0.2977 (Seed 42; 3-seed mean: +0.2492, bootstrap P=1.0)",
            "config_baseline": "configs/baseline.yaml",
            "config_proposed": "configs/proposed.yaml"
        },
        {
            "split_name": "train100_seed42",
            "budget_percent": 100,
            "training_samples": 4415,
            "status": "Stratified Split Verified (Model Training: Future Work)",
            "baseline_test_macro_f1": "— (Unperformed)",
            "proposed_test_macro_f1": "— (Unperformed)",
            "gain": "— (Unperformed)",
            "config_baseline": "configs/ablations/dataeff_baseline_100.yaml",
            "config_proposed": "configs/ablations/dataeff_proposed_100.yaml"
        }
    ]
    df_data_eff = pd.DataFrame(data_eff_data)
    df_data_eff.to_csv(ablations_dir / "data_efficiency_summary.csv", index=False)
    with open(ablations_dir / "data_efficiency_summary.json", "w", encoding="utf-8") as f:
        json.dump(data_eff_data, f, indent=2)

    # 3. Figure 1: Anchor Models Comparison on Validation & Test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    models = ["Baseline CNN\n(Scratch)", "Proposed Model\n(Staged ResNet-18)"]
    val_f1s = [0.5994, 0.8051]
    test_f1s = [0.5855, 0.8347]
    colors = ["#d95f02", "#1b9e77"]

    # Val comparison
    bars1 = ax1.bar(models, val_f1s, color=colors, width=0.55)
    ax1.set_ylabel("Validation Macro-F1 (Seed 42)")
    ax1.set_title("Validation Model Selection Benchmark")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10, weight="bold")

    # Test comparison
    bars2 = ax2.bar(models, test_f1s, color=colors, width=0.55)
    ax2.set_ylabel("Test Macro-F1 (Mean over Seeds)")
    ax2.set_title("Official Test Split Benchmark (N=624)")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10, weight="bold")

    plt.suptitle("CS320 P06: Baseline vs Proposed Anchor Performance (25% Training Budget)", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "ablation_backbone_regimes.png", dpi=150)
    plt.close()

    # 4. Figure 2: Conceptual Ablation Hierarchy (Design Framework)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    components = [
        "1. Scratch CNN Reference\n(Baseline: F1 = 0.5855)",
        "2. Pretrained Inductive Bias\n(ImageNet ResNet-18)",
        "3. Staged Progressive Unfreezing\n(Head -> Layer4 -> Full)",
        "4. X-Ray Safe Augmentation\n(No Horizontal Flips)",
        "5. Label Smoothing Regularization\n(eps = 0.1 for ECE)",
        "6. Subset Class Weighting\n(Fixes Majority Bias)",
        "7. Final Integrated Proposed Model\n(F1 = 0.8347)",
    ]
    y_pos = np.arange(len(components))[::-1]
    ax.scatter([1] * len(components), y_pos, color="#2b8cbe", s=120, zorder=3)
    ax.plot([1] * len(components), y_pos, color="#a6bddb", linewidth=2.5, zorder=2)
    for idx, (y, comp) in enumerate(zip(y_pos, components)):
        color = "#d95f02" if idx == 0 else ("#1b9e77" if idx == len(components)-1 else "#252525")
        weight = "bold" if idx in (0, len(components)-1) else "normal"
        ax.text(1.05, y, comp, va="center", fontsize=10, color=color, weight=weight)

    ax.set_xlim(0.9, 2.5)
    ax.set_ylim(-0.8, len(components) - 0.2)
    ax.axis("off")
    ax.set_title("Proposed Solution Component Hierarchy & Ablation Framework", fontsize=12, weight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(fig_dir / "ablation_component_contributions.png", dpi=150)
    plt.close()

    # 5. Figure 3: Data Efficiency Sampling Framework (10%, 25%, 100%)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    budgets = ["10% Budget\n(train10)", "25% Budget [Task P06]\n(train25)", "100% Budget\n(train100)"]
    normal_counts = [116, 290, 1161]
    bacteria_counts = [211, 526, 2107]
    virus_counts = [114, 287, 1147]

    x = np.arange(len(budgets))
    w = 0.25

    ax.bar(x - w, normal_counts, width=w, label="NORMAL", color="#4daf4a")
    ax.bar(x, bacteria_counts, width=w, label="BACTERIA", color="#e41a1c")
    ax.bar(x + w, virus_counts, width=w, label="VIRUS", color="#377eb8")

    ax.set_ylabel("Number of Training Radiographs")
    ax.set_title("Data Efficiency Framework: Class Distributions Across Training Budgets")
    ax.set_xticks(x)
    ax.set_xticklabels(budgets)
    ax.legend(title="Class")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(budgets)):
        total = normal_counts[i] + bacteria_counts[i] + virus_counts[i]
        ax.annotate(f"Total: {total}\n{'[TRAINED]' if i==1 else '[FUTURE WORK]'}",
                    xy=(x[i], max(bacteria_counts[i], normal_counts[i], virus_counts[i]) + 80),
                    ha="center", fontsize=9, weight="bold")

    ax.set_ylim(0, 2600)
    plt.tight_layout()
    plt.savefig(fig_dir / "data_efficiency_curve.png", dpi=150)
    plt.close()

    print("Honest ablation study design matrix and figures generated successfully!")


if __name__ == "__main__":
    generate_ablation_artifacts()
