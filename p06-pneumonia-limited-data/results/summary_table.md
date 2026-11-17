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

## Controlled ablations (validation macro-F1; one seed unless noted)

| Experiment | Seeds | Val macro-F1 | Val balanced acc. | Val macro-AUROC |
|---|---|---|---|---|
| (none yet) | — | — | — | — |

## Data-efficiency curve (test macro-F1; supplementary — 100% is context only)

| Model | Train fraction | Test macro-F1 | Seeds |
|---|---|---|---|
| (none yet) | — | — | — |

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
- Ablations and data-efficiency models were not completed on the CPU host; no numbers are claimed for them.
