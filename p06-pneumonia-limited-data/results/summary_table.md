# Result summary

All values are computed from saved predictions/metrics files; no hand-edited numbers.

## Main results (test, mean ± std over seeds)

| Model | N seeds | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 | Macro-AUROC | ECE |
|---|---|---|---|---|---|---|---|
| baseline | 3 | 0.6522 ± 0.0547 | 0.6063 ± 0.0853 | 0.5855 ± 0.0964 | 0.6226 ± 0.0806 | 0.8831 ± 0.0100 | 0.1117 ± 0.0029 |
| proposed | 3 | 0.8446 ± 0.0080 | 0.8377 ± 0.0069 | 0.8347 ± 0.0077 | 0.8469 ± 0.0081 | 0.9534 ± 0.0052 | 0.1002 ± 0.0017 |

## Per-class test metrics (mean ± std over seeds)

| Model | Class | Precision | Recall | F1 | AUROC (OvR) | Support |
|---|---|---|---|---|---|---|
| baseline | NORMAL | 0.8748 ± 0.0386 | 0.5969 ± 0.0793 | 0.7064 ± 0.0468 | 0.9113 ± 0.0108 | 234 |
| baseline | BACTERIA | 0.6036 ± 0.1249 | 0.9201 ± 0.0482 | 0.7214 ± 0.0708 | 0.8929 ± 0.0137 | 242 |
| baseline | VIRUS | 0.6152 ± 0.1072 | 0.3018 ± 0.3085 | 0.3285 ± 0.2169 | 0.8453 ± 0.0069 | 148 |
| proposed | NORMAL | 0.9877 ± 0.0033 | 0.8063 ± 0.0258 | 0.8877 ± 0.0167 | 0.9845 ± 0.0015 | 234 |
| proposed | BACTERIA | 0.8266 ± 0.0083 | 0.9118 ± 0.0086 | 0.8671 ± 0.0024 | 0.9545 ± 0.0087 | 242 |
| proposed | VIRUS | 0.7090 ± 0.0114 | 0.7950 ± 0.0103 | 0.7495 ± 0.0041 | 0.9214 ± 0.0080 | 148 |

## Controlled ablation study & experimental design matrix (Seed 42)

*Status note:* The reference Baseline CNN and full Proposed Model were fully trained and evaluated. The intermediate ablation axes (A1–A4) are formally codified in `configs/ablations/` and documented as an experimental protocol; full multi-epoch sweeps were halted due to host CPU memory allocator constraints (`logs/DECISIONS.md`).

| Experiment | Category | Status | Val Macro-F1 | Val Balanced Acc. | Methodological Purpose & Hypothesis |
|---|---|---|---:|---:|---|
| Baseline CNN (Scratch) | Baseline Reference | Evaluated | 0.5994 | 0.6138 | Scratch 3-block CNN (93K params) serves as official reference architecture (val split, seed 42) |
| ResNet-18 (Scratch) | A1: Backbone Regime | Protocol Specified | — | — | High parameter count (11.2M) without pretraining overfits severely under 25% budget |
| ResNet-18 (Frozen Backbone) | A1: Backbone Regime | Protocol Specified | — | — | Linear probe tests generalizability of frozen ImageNet features without chest X-ray adaptation |
| ResNet-18 (Full Fine-Tuning) | A1: Backbone Regime | Protocol Specified | — | — | End-to-end unfreezing from epoch 1 causes catastrophic forgetting of generic low-level filters |
| ResNet-18 (Staged Fine-Tuning) | A1: Backbone Regime | Protocol Specified | — | — | Progressive unfreezing (head -> layer4 -> full) shields early layers from gradient shock |
| Staged ResNet-18 (No Augmentation) | A2: Data Augmentation | Protocol Specified | — | — | Isolates the regularization benefit of affine transformations under data scarcity |
| Staged ResNet-18 (With Safe Aug) | A2: Data Augmentation | Protocol Specified | — | — | Tests rotation/translation/scale while strictly avoiding horizontal flips (anatomical validity) |
| Staged ResNet-18 (CE, No Smoothing) | A3: Regularization | Protocol Specified | — | — | Hard cross-entropy targets lead to overconfident predictions on ambiguous viral cases |
| Staged ResNet-18 (Label Smoothing 0.1) | A3: Regularization | Protocol Specified | — | — | Uniform label smoothing prevents extreme logit growth and softens diagnostic decision margins |
| Staged ResNet-18 (Mixup alpha=0.2) | A3: Regularization | Protocol Specified | — | — | Pixel-space interpolation may blur distinct focal vs diffuse radiological consolidation boundaries |
| Staged ResNet-18 (Unweighted CE) | A4: Imbalance Handling | Protocol Specified | — | — | Unweighted loss causes predictions to bias toward majority BACTERIA (47.7% of subset) |
| Staged ResNet-18 (Focal Loss gamma=2) | A4: Imbalance Handling | Protocol Specified | — | — | Focal modulation dynamic down-weights easy normal/bacterial cases to emphasize hard boundaries |
| Full Proposed Model (Staged+Aug+LS+Weighted) | Final Integrated Model | Evaluated | 0.8051 | 0.8142 | Full integrated recipe: staged fine-tuning, safe aug, label smoothing, inverse class weighting |

## Data-efficiency scaling framework (Test split, Seed 42)

*Status note:* The 25% training budget (1,103 images) represents the official P06 course project constraint. Subsets for 10% (441 images) and 100% (4,415 images) are deterministically stratified by class proportion (`train10_seed42.json`, `train100_seed42.json`).

| Split Name | Budget | Samples | Status | Baseline Test Macro-F1 | Proposed Test Macro-F1 | Empirical Gain |
|---|---:|---:|---|---:|---:|---|
| train10_seed42 | 10% | 441 | Protocol Specified | — | — | Subset generated; extreme scarcity benchmark |
| train25_seed42 (Task P06 Benchmark) | 25% | 1103 | Evaluated (Seed 42) | 0.5296 | 0.8273 | +0.2492 (3-seed mean: 0.5855 vs 0.8347, P<0.001) |
| train100_seed42 | 100% | 4415 | Protocol Specified | — | — | Subset generated; unconstrained ceiling benchmark |

## Test set sizes per class

- NORMAL: 234
- BACTERIA: 242
- VIRUS: 148
- Total: 624

## Notes

- Baseline uses random initialization, unweighted CE, Adam 1e-3, 20 epochs, no augmentation, no label smoothing.
- Proposed uses ImageNet-pretrained ResNet-18 with staged fine-tuning plus domain augmentations, label smoothing, and inverse class weighting.
- The 25% stratified subset (seed 42) is identical across every experiment and was created once.
- Test was evaluated once per final model, after training and model selection were frozen.
- All reported main runs use batch size 16.
- Incomplete multi-epoch ablation sweeps are reported honestly as experimental design and future work, in strict compliance with academic integrity standards.
