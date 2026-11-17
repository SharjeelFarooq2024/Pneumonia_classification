# Design and experiment decisions

This file records the decisions that determine what the reported numbers mean. It is deliberately dated and explicit about deviations.

## 1. Task and data

- **Dataset:** MedIMeta `pneumonia`, task `disease class`.
- **Classes:** `NORMAL`, `BACTERIA`, `VIRUS` (integer labels 0, 1, 2).
- **Splits:** the official MedIMeta `train`, `val`, and `test` split files. No resplitting was performed.
- **Data path:** `data/raw` is passed to MedIMeta and the dataset lives in `data/raw/pneumonia`.
- **Images:** grayscale TIFFs, resized to 224×224, replicated to three channels, and normalized. Baseline uses mean/std 0.5; the pretrained model uses ImageNet normalization.

Official split sizes and class counts were verified against the dataset metadata:

| Split | NORMAL | BACTERIA | VIRUS | Total |
|---|---:|---:|---:|---:|
| Train | 1161 | 2107 | 1147 | 4415 |
| Val | 188 | 431 | 198 | 817 |
| Test | 234 | 242 | 148 | 624 |

## 2. Fixed limited-data subset

- The main experiments use a **25% stratified sample of the official training split**, seed 42.
- The subset is stored in `data/splits/train25_seed42.json` and contains 1103 images: NORMAL 290, BACTERIA 526, VIRUS 287.
- Its maximum absolute class-proportion difference from the full training split is 0.0004.
- The subset was created once and is reused by every baseline, proposed, and optional experiment. It is never regenerated for a new run.
- Supplementary 10% and 100% subsets were created only for a planned data-efficiency study. They do not affect the main results.

Rationale: a fixed, documented subset makes the limited-data claim reproducible and prevents silent changes in the training distribution between runs.

## 3. Baseline model

- Three convolutional blocks: 32, 64, 128 filters, each followed by ReLU and 2×2 max pooling (no batch normalization).
- Global average pooling and a linear classifier with 3 outputs.
- 93,635 parameters.
- Adam, learning rate 1e-3, weight decay 0, batch size 16, 20 epochs.
- No pretrained weights, no augmentation, no class weighting, no label smoothing, no mixup/cutmix.
- Model selection: best validation macro-F1 over the 20 epochs.

Rationale: the baseline is intentionally a plain scratch CNN so that the effect of transfer learning and the proposed training recipe is not confounded by modern regularization tricks.

## 4. Proposed model

- ImageNet-pretrained ResNet-18, dropout 0.2 before the classifier.
- Three stages:
  1. **head:** 3 epochs, only `fc` trainable, lr 1e-3, cosine schedule.
  2. **layer4:** 4 epochs, `layer4` and `fc` trainable, lr 1e-4, cosine schedule.
  3. **full:** 5 epochs, all parameters trainable, base lr 1e-5, cosine schedule, discriminative layer multipliers (`conv1`/`bn1` 0.1, `layer1` 0.1, `layer2` 0.3, `layer3` 0.5, `layer4` 0.8, `fc` 1.0).
- Weight decay 1e-4 in stages 2 and 3; Adam.
- X-ray-safe augmentation: ±10° rotation, ±5% translation, 0.95–1.05 scale, brightness/contrast jitter 0.1. No horizontal flip.
- Loss: weighted cross-entropy with weights computed from the 25% training subset only, plus label smoothing 0.1.
- Batch size 16 (see deviations below), 12 epochs total.
- Model selection: best validation macro-F1 across the 12 epochs.

Rationale for no horizontal flip: left/right is clinically meaningful in chest radiography and flipping can create anatomically implausible images. The remaining augmentations are mild and intensity/geometry preserving.

Rationale for class weights: the 25% subset is imbalanced (BACTERIA 526 vs VIRUS 287). Weights are computed from the training subset, never from validation or test.

## 5. Validation and test protocol

- Validation macro-F1 is the only model-selection criterion.
- The test split is never loaded by `src/train.py`.
- `src/evaluate.py` evaluates validation and test once for each final model and writes `results/<experiment>/seed<seed>/test_eval_done.json`.
- A second test evaluation is refused unless `--force` is passed. `--force` is only for repairing a clearly broken evaluation, not for selecting a better result.
- Protocol exception: the earlier proposed seed-44 batch-8 model was evaluated on test before it was replaced by the final batch-16 retrain. The new seed-44 model was then evaluated once. This means seed 44 has effectively seen the test set twice; the reported result is the final batch-16 model, and the batch-8 result is not used.
- Test predictions and metrics are saved as CSV/JSON so the report can be audited.

## 6. Seeds and aggregation

- Main models were trained with seeds 42, 43, and 44.
- Results are reported as mean ± sample standard deviation (`ddof=1`).
- Paired bootstrap resamples test images (2000 draws per seed) to compare baseline and proposed macro-F1. Positive differences favour the proposed model.

## 7. Deviations forced by the host

The machine is CPU-only with about 8 GB RAM and a small Windows paging file. Several runs hit native allocator failures, not model errors.

- **Batch size:** the initial configured batch size was 64. Full baseline runs at 64 caused a native access violation / OOM before the first epoch. All reported runs therefore use **batch size 16**.
- **Proposed seed 44:** seeds 42 and 43 completed at batch size 16. Seed 44 initially failed at batch 16 with allocator errors and was completed at batch size 8. On 2026-11-17 it was retrained successfully at **batch size 16** after freeing host memory. The batch-8 artifacts are preserved only in a temporary backup outside the reported results; the reported seed-44 checkpoint, config, history, and metrics are the batch-16 versions.
- **Evaluation:** `src/evaluate.py` builds the model with `pretrained=False` before loading the checkpoint. This avoids downloading/loading ImageNet weights during evaluation and removes an unnecessary allocation.
- **Data loading:** `num_workers=0` for all reported runs. Windows worker processes and the small paging file made multi-worker loading unstable.
- **No AMP:** CPU autocast was not used because the CPU reports AVX-512 but not AVX-512 BF16 support; CUDA is unavailable.

These deviations are also visible in each run's `config.yaml` and `run_info.json`.

## 8. What was not completed

- **Ablations are not reported.** Configs exist in `configs/ablations/` for backbone regimes (frozen, full, scratch, staged), augmentation, label smoothing, mixup, focal loss, and class weighting. The A1 runs were started but the host ran out of memory and/or crashed. Incomplete result directories were removed so no partial run can be mistaken for a finished experiment.
- **Data-efficiency models are not reported.** The 10% and 100% subsets exist, but no models were trained on them within the available budget. No data-efficiency figure is produced because fewer than two operating points were trained.
- **No test-set ablation comparison was performed.** Because ablations were not completed, the test split was used only for the six final main models (3 baseline + 3 proposed).

No numbers were fabricated or copied from an unfinished run.

## 9. Software details that affected results

- MedIMeta replaces `transform=None` with its own default transform, which includes `ToTensor`. The wrapper therefore passes an explicit no-op transform (`_identity`) so that the project's own transform pipeline receives PIL images.
- The evaluation confusion-matrix plot originally passed float data with an integer format specifier (`fmt="d"`), which raised an error after metrics had been computed. It now uses `.0f` for counts and `.2f` for row-normalized matrices.
- Smoke mode originally re-read the config after setting one epoch per stage, which reset the stage length to 20. The smoke override now builds the stage list once, sets one epoch per stage, and reuses that list.

## 10. Interpretation

- The proposed model improves test macro-F1 from 0.5855 ± 0.0964 to 0.8347 ± 0.0077, and the paired bootstrap difference is positive for all three seeds.
- The largest baseline failure is VIRUS recall: the baseline often predicts BACTERIA for viral images. The proposed model substantially reduces this confusion but VIRUS remains the hardest class.
- AUROC is high for both models (baseline about 0.88, proposed about 0.96), while thresholded macro-F1 is lower, indicating that ranking is easier than choosing the correct class at the default threshold.
- Calibration is moderate: proposed ECE is about 0.100, so predicted probabilities should not be interpreted as clinical risks.
- These results are from a 25% training subset and a single pediatric dataset. They are not evidence of clinical validity.


