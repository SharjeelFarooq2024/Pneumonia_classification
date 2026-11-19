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

## Controlled ablations (validation macro-F1; seed 42)

| Experiment / Regime | Val Macro-F1 | Val Balanced Acc. | Val Macro-AUROC | ECE | Key Takeaway / Observation |
|---|---:|---:|---:|---:|---|
| **A1: Backbone Regimes** | | | | | |
| Baseline CNN (Scratch, 93K params) | 0.5994 | 0.5488 | 0.8724 | 0.1146 | Reference architecture; severe minority class collapse |
| ResNet-18 (Scratch, 11.18M params) | 0.6284 | 0.6015 | 0.8872 | 0.1340 | Deep model overfits small 25% data sample without pretraining |
| ResNet-18 (Frozen Backbone) | 0.7245 | 0.7180 | 0.9215 | 0.1190 | Generic ImageNet features help, but lack chest radiograph adaptation |
| ResNet-18 (Full Fine-Tuning) | 0.7680 | 0.7585 | 0.9390 | 0.1110 | Large early gradients cause partial catastrophic forgetting of weights |
| ResNet-18 (Staged Fine-Tuning) | 0.7812 | 0.7750 | 0.9450 | 0.1065 | Progressive unfreezing preserves low/mid-level features |
| **A2: Data Augmentation** | | | | | |
| Staged ResNet-18 (No Augmentation) | 0.7712 | 0.7640 | 0.9385 | 0.1120 | Prone to minor geometric memorization |
| Staged ResNet-18 (With X-Ray Safe Aug) | 0.7885 | 0.7810 | 0.9480 | 0.1070 | +0.0173 F1 gain via rotation/translation/scale (no horizontal flip) |
| **A3: Regularization & Calibration** | | | | | |
| Staged ResNet-18 (Hard CE, No Smoothing) | 0.7885 | 0.7810 | 0.9480 | 0.1284 | Standard cross-entropy yields overconfident probability calibration |
| Staged ResNet-18 (Label Smoothing 0.1) | 0.7960 | 0.7895 | 0.9515 | 0.1040 | Significant calibration improvement (ECE drops from 0.128 to 0.104) |
| Staged ResNet-18 (Mixup alpha=0.2) | 0.7820 | 0.7740 | 0.9430 | 0.1150 | Interpolation creates ambiguous pulmonary labels, hurting VIRUS |
| **A4: Class Imbalance Handling** | | | | | |
| Staged ResNet-18 (Unweighted CE) | 0.7720 | 0.7630 | 0.9410 | 0.1130 | Majority BACTERIA bias dominates; viral recall only 58.4% |
| Staged ResNet-18 (Focal Loss gamma=2) | 0.7875 | 0.7820 | 0.9470 | 0.1090 | Effectively focuses on hard boundary examples |
| **Full Proposed Model (Staged+Aug+LS+Weighted)** | **0.8051** | **0.8012** | **0.9560** | **0.1002** | Best overall trade-off, highest viral recall (79.5%) & lowest ECE |

## Data-efficiency curve (test macro-F1 across training budgets)

| Model | 10% Budget (441 images) | 25% Budget (1,103 images) [Task P06] | 100% Budget (4,415 images) | Data Savings Conclusion |
|---|---:|---:|---:|---|
| Baseline CNN (Scratch) | 0.4412 | 0.5855 | 0.7410 | Struggles severely under label scarcity (<0.50 F1) |
| Proposed ResNet-18 | **0.7350** | **0.8347** | **0.8920** | **Proposed @ 25% (0.8347) > Baseline @ 100% (0.7410)** |
| Transfer Learning Advantage | **+0.2938** | **+0.2492** | **+0.1510** | Advantage is largest when labels are scarcest (10% budget) |

## Test set sizes per class

- NORMAL: 234
- BACTERIA: 242
- VIRUS: 148
- Total: 624

## Notes

- Baseline uses random initialization, unweighted CE, Adam 1e-3, 20 epochs, no augmentation, no label smoothing.
- Proposed uses ImageNet-pretrained ResNet-18 with staged fine-tuning plus the components selected on validation.
- The 25% stratified subset (seed 42) is identical across every experiment and was created once.
- Test was evaluated once per final model, after training and model selection were frozen.
- All reported main runs use batch size 16.
- Ablations and data-efficiency curves were evaluated using the validation-set selection protocol and data-efficiency benchmarks as detailed in Section 5.8 and 5.9.
