# Pediatric Pneumonia Classification under Limited Data
**CS320: Artificial Neural Networks & Deep Learning — Term Project P06 (BSAI-2k24, SEECS)**  
**Team Members:** Sharjeel Farooq, Affan Ahmad, Mohammad Asef

Three-class pediatric chest X-ray classification (**NORMAL**, **BACTERIA**, **VIRUS**) on the MedIMeta pneumonia dataset. The main experiments use a fixed, stratified **25% subset of the official training split** (1103 images) and compare a scratch CNN baseline against an ImageNet-pretrained ResNet-18 with staged fine-tuning.

Validation macro-F1 is used for model selection. The official test split is evaluated once per final model, after all training and selection decisions are frozen.

---

## Results at a glance

Test set: 624 images (NORMAL 234, BACTERIA 242, VIRUS 148). Mean ± sample standard deviation over seeds 42, 43, 44.

| Model | Accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 | Macro-AUROC | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline CNN | 0.6522 ± 0.0547 | 0.6063 ± 0.0853 | 0.5855 ± 0.0964 | 0.6226 ± 0.0806 | 0.8831 ± 0.0100 | 0.1117 ± 0.0029 |
| Proposed ResNet-18 | **0.8446 ± 0.0080** | **0.8377 ± 0.0069** | **0.8347 ± 0.0077** | **0.8469 ± 0.0081** | **0.9534 ± 0.0052** | **0.1002 ± 0.0017** |

Per-seed test macro-F1:

| Seed | Baseline | Proposed |
|---|---:|---:|
| 42 | 0.5296 | 0.8273 |
| 43 | 0.5299 | 0.8343 |
| 44 | 0.6968 | 0.8426 |

Paired bootstrap over test images (2000 resamples per seed): mean proposed-minus-baseline macro-F1 difference **+0.249**, with a 95% interval excluding zero for every seed. Details: `results/bootstrap_comparison.json`.

The baseline collapses toward BACTERIA on viral images. In the seed-42 bacterial-vs-viral analysis, 82.98% of viral images are predicted as bacterial by the baseline versus 19.05% by the proposed model (`results/bacterial_viral_confusion.json`).

---

## Dataset and fixed subset

MedIMeta `pneumonia`, task `disease class`, official splits:

| Split | NORMAL | BACTERIA | VIRUS | Total |
|---|---:|---:|---:|---:|
| Train | 1161 | 2107 | 1147 | 4415 |
| Validation | 188 | 431 | 198 | 817 |
| Test | 234 | 242 | 148 | 624 |

The official experimental subset is `data/splits/train25_seed42.json`:

- 1103 images: NORMAL 290, BACTERIA 526, VIRUS 287
- maximum absolute class-proportion difference from the full training split: 0.0004
- created once and reused by every reported experiment

Supplementary stratified splits `train10_seed42.json` (441 images) and `train100_seed42.json` (4415 images) were generated as formal benchmarks for data-efficiency scaling analysis (Section 5.9 and `results/ablations/data_efficiency_summary.csv`).

---

## Architecture

### 1. Shared data pipeline

MedIMeta is accessed with `use_hdf5=False`. The wrapper passes an explicit no-op transform because MedIMeta otherwise substitutes a default transform that includes `ToTensor`.

```text
MedIMeta TIFF + official split files
        │
        ▼
PneumoniaDataset(split, positions)
  - positions come from data/splits/train25_seed42.json for training
  - returns (image, label, global_image_index)
        │
        ▼
Transform
  - Resize 224×224
  - Grayscale(num_output_channels=3)
  - ToTensor
  - Normalize
      baseline: mean=std=0.5
      proposed: ImageNet mean/std
        │
        ├── train loader: shuffle=True
        ├── val loader:   shuffle=False
        └── test loader:  only in src/evaluate.py
```

Training augmentation is used only by the proposed model and only on the training split:

- ±10° rotation
- ±5% translation
- 0.95–1.05 scale
- brightness/contrast jitter 0.1
- **no horizontal flip** (left/right orientation is clinically meaningful in chest radiography)

### 2. Baseline CNN (exact architecture)

The baseline is a small scratch network with no pretrained weights, no batch normalization, no dropout, no augmentation, no class weighting, and no label smoothing.

```text
Input: 3 × 224 × 224
  Conv2d(3 → 32, kernel 3×3, padding 1) → ReLU → MaxPool2d(2)   → 32 × 112 × 112
  Conv2d(32 → 64, kernel 3×3, padding 1) → ReLU → MaxPool2d(2)  → 64 × 56 × 56
  Conv2d(64 → 128, kernel 3×3, padding 1) → ReLU → MaxPool2d(2) → 128 × 28 × 28
  AdaptiveAvgPool2d(1) → flatten → 128
  Linear(128 → 3)
```

| Property | Value |
|---|---|
| Parameters | 93,635 |
| Loss | unweighted cross-entropy |
| Optimizer | Adam, lr 1e-3, weight decay 0 |
| Epochs | 20 |
| Batch size | 16 |
| Checkpoint | best validation macro-F1 |

### 3. Proposed ResNet-18 (exact architecture)

The proposed model is torchvision `resnet18` with ImageNet weights, a replaced classifier head, and staged unfreezing.

```text
Input: 3 × 224 × 224 (ImageNet normalization)
  conv1: 7×7, 64, stride 2 → BatchNorm → ReLU → MaxPool 3×3 stride 2
  layer1: 2 × BasicBlock(64)
  layer2: 2 × BasicBlock(128, stride 2)
  layer3: 2 × BasicBlock(256, stride 2)
  layer4: 2 × BasicBlock(512, stride 2)
  AdaptiveAvgPool2d(1) → 512
  Dropout(0.2)
  Linear(512 → 3)
```

| Property | Value |
|---|---|
| Parameters | 11,178,051 |
| Loss | weighted cross-entropy + label smoothing 0.1 |
| Class weights | computed from the 25% training subset only |
| Optimizer | Adam, cosine schedule per stage |
| Epochs | 3 + 4 + 5 = 12 |
| Batch size | 16 |
| Checkpoint | best validation macro-F1 across all stages |

### 4. Staged fine-tuning schedule

| Stage | Epochs | Trainable modules | Base LR | Weight decay | Scheduler |
|---|---:|---|---:|---:|---|
| `head` | 3 | `fc` only | 1e-3 | 0 | cosine |
| `layer4` | 4 | `layer4`, `fc` | 1e-4 | 1e-4 | cosine |
| `full` | 5 | all parameters | 1e-5 | 1e-4 | cosine |

In the full stage, layer-wise learning-rate multipliers are applied: `conv1`/`bn1` 0.1, `layer1` 0.1, `layer2` 0.3, `layer3` 0.5, `layer4` 0.8, `fc` 1.0. This lets lower layers move more slowly than the classifier.

### 5. Why this design

The baseline isolates the effect of the proposed recipe. The proposed model adds:

1. ImageNet feature transfer for low- and mid-level filters.
2. Progressive unfreezing to avoid destroying pretrained features with large early gradients.
3. Class weighting to reduce majority-class bias (BACTERIA is over-represented in the subset).
4. Label smoothing to reduce overconfidence.
5. Mild augmentation to improve robustness without flipping anatomy.

These components are evaluated as a bundle. Ablations are **not** implemented (see below).

---

## End-to-end pipeline

```text
configs/*.yaml
      │
      ▼
load_config
  - recursively merges base_config
  - applies CLI --set overrides
      │
      ▼
Seed, device selection, environment logging
      │
      ▼
Load fixed subset positions from data/splits/train25_seed42.json
      │
      ▼
Build train / validation datasets and loaders
  (test is NOT loaded here)
      │
      ▼
Build model (baseline_cnn | resnet18 | efficientnet_b0)
      │
      ▼
Build criterion, optimizer, scheduler
      │
      ▼
For each stage:
    set_trainable(stage.unfreeze)
    for each epoch:
        train one epoch
        evaluate on validation
        scheduler.step()
        append history.csv
        if val macro-F1 improves:
            save best model state
      │
      ▼
Save checkpoint + run_info.json + config.yaml
      │
      ▼
src/evaluate.py
  - load best-validation checkpoint
  - evaluate validation and test once
  - write metrics_*.json, predictions_*.csv, confusion matrices
  - write test_eval_done.json marker
      │
      ▼
src/analysis.py
  - aggregate tables
  - training curves, confusion matrices, per-class F1, reliability diagram
  - paired bootstrap
  - misclassified examples and Grad-CAM
```

### Stage-to-code map

| Stage | Main code |
|---|---|
| Subset creation / verification | `scripts/make_subset.py`, `src/data/subset.py` |
| MedIMeta access + transforms | `src/data/dataset.py` |
| Baseline model | `src/models/baseline_cnn.py` |
| Proposed model + staged freezing + discriminative LRs | `src/models/proposed.py` |
| Training loop + validation + checkpointing | `src/train.py` |
| One-time final evaluation | `src/evaluate.py` |
| Aggregation, figures, bootstrap, Grad-CAM | `src/analysis.py` |
| Reproducible command runner | `scripts/run_experiments.py` |

---

## How to run

### 1. Environment

The reported runs used Python 3.10.11, torch 2.13.0+cpu, torchvision 0.28.0+cpu, CPU only, 8 logical cores, and about 8 GB RAM.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Data layout

Place the MedIMeta pneumonia data here:

```text
data/raw/pneumonia/
├── annotations.csv
├── images/
├── images.hdf5
├── splits/
└── task_labels/
```

`src/data/dataset.py` passes `data/raw` as the MedIMeta `data_path` and uses:

- `dataset_id="pneumonia"`
- `task_name="disease class"`
- `use_hdf5=False`

### 3. Check or recreate the fixed subset

The fixed subset already exists. Running the command below loads it and verifies it; it does not overwrite it unless `--force` is supplied.

```powershell
python scripts/make_subset.py --fraction 0.25 --seed 42
```

Do not recreate `train25_seed42.json` for a new experiment. All reported models use the same file.

### 4. Smoke tests

A smoke test runs one epoch per stage and only two batches per epoch. It checks the full pipeline without spending the full training budget.

```powershell
python -m src.train --config configs/baseline.yaml --seed 42 --smoke
python -m src.train --config configs/proposed.yaml --seed 42 --smoke
```

### 5. Train the main models

```powershell
# Baseline, seeds 42/43/44
foreach ($s in 42,43,44) {
  python -m src.train --config configs/baseline.yaml --seed $s
}

# Proposed ResNet-18, seeds 42/43/44
foreach ($s in 42,43,44) {
  python -m src.train --config configs/proposed.yaml --seed $s
}
```

Runtime on the CPU host: about 22–25 minutes per baseline seed and 21–31 minutes per proposed seed.

### 6. Evaluate once on validation and test

```powershell
foreach ($s in 42,43,44) {
  python -m src.evaluate --checkpoint checkpoints/baseline_seed$s.pt --splits val test
  python -m src.evaluate --checkpoint checkpoints/proposed_seed$s.pt --splits val test
}
```

`test_eval_done.json` prevents accidental re-evaluation. Use `--force` only to repair a clearly broken evaluation, never to chase a better number.

### 7. Run the analysis

```powershell
python -m src.analysis --config configs/base.yaml aggregate
python -m src.analysis --config configs/base.yaml --seed 42 figures
python -m src.analysis --config configs/base.yaml --seed 42 bootstrap
python -m src.analysis --config configs/base.yaml --seed 42 gradcam
```

### 8. Use the one-command runner

```powershell
# Everything: train both models, evaluate, and run analysis
python scripts/run_experiments.py main --seeds 42 43 44

# Just the analysis from existing results
python scripts/run_experiments.py analysis --seeds 42

# Just the one-time evaluation
python scripts/run_experiments.py evaluate --seeds 42 43 44
```

The runner skips existing checkpoints unless `--force` is supplied. All reported main runs use batch size 16 and `data.num_workers=0`. If you must reproduce on a machine with less memory, lowering the batch size is possible but would be a deviation from the reported setup.

---

## Outputs

| Path | Contents |
|---|---|
| `results/summary_table.md`, `results/summary_table.csv` | Main tables generated from saved metrics |
| `results/<experiment>/seed<seed>/metrics_{val,test}.json` | Full metrics, per-class stats, reliability bins |
| `results/<experiment>/seed<seed>/predictions_{val,test}.csv` | Per-image probabilities and predictions |
| `results/<experiment>/seed<seed>/history.csv` | Per-epoch training/validation history |
| `results/<experiment>/seed<seed>/test_eval_done.json` | One-time test-evaluation marker |
| `results/bootstrap_comparison.json` | Paired bootstrap statistics |
| `results/bacterial_viral_confusion.json` | Bacterial-vs-viral error rates |
| `results/figures/` | Training curves, confusion matrices, per-class F1, reliability, Grad-CAM |
| `checkpoints/` | Best-validation weights (not committed) |
| `logs/runs/` | Run configs and environment logs |
| `logs/DECISIONS.md` | Design decisions, protocol, and deviations |

---

## Reproducibility and test protocol

- `data/splits/train25_seed42.json` is created once and reused by every experiment.
- Training and model selection never load the test split.
- `src/evaluate.py` writes a one-time marker and refuses a second test evaluation unless `--force` is used.
- Protocol exception: seed 44 was retrained at batch 16 after an earlier batch-8 model had already been evaluated, so test was seen twice for this seed. The reported seed-44 result is from the final batch-16 model.
- The subset is deterministic from seed 42 and stratified by class.
- Seeds 42, 43, and 44 are used for both main models.
- Checkpoints, configs, environment versions, and histories are saved for every run.
- Results are reported as mean ± sample standard deviation (`ddof=1`).

---

## Ablation protocol & experimental design matrix

The 14 ablation experiments under `configs/ablations/` are formally codified as an experimental protocol across four axes (backbone regimes, augmentation, loss regularization, class weighting). Due to host CPU memory allocator constraints (`logs/DECISIONS.md`), multi-epoch sweeps were halted, and are reported honestly as an experimental design matrix and future work rather than simulated with unverified numbers. Real validation anchors are evaluated for Baseline CNN (0.5994 macro-F1, 0.6138 balanced acc) and Proposed Staged ResNet-18 (0.8051 macro-F1, 0.8142 balanced acc).

---

## Trained model checkpoints & download links

Because model weights files (`.pt`) exceed standard Git tracking limits and are excluded by `.gitignore`, all 6 trained model checkpoints (Seeds 42, 43, 44 for Baseline CNN and Proposed ResNet-18) are archived externally:

- **GitHub Release:** [v1.0 Checkpoints Bundle](https://github.com/SharjeelFarooq2024/Pneumonia_classification/releases/tag/v1.0-checkpoints)
- **Google Drive Mirror:** [CS320-P06 Checkpoints Mirror](https://drive.google.com/drive/folders/1A_P06_Pneumonia_Model_Weights_CS320) *(Placeholder: Update with actual Drive folder link upon upload)*

To run evaluation or inference demos from scratch, download the weights into:
```text
p06-pneumonia-limited-data/
└── checkpoints/
    ├── baseline/
    │   ├── seed42/best_model.pt
    │   ├── seed43/best_model.pt
    │   └── seed44/best_model.pt
    └── proposed/
        ├── seed42/best_model.pt
        ├── seed43/best_model.pt
        └── seed44/best_model.pt
```

---

## Limitations and deviations

- **All reported main runs use batch size 16.** An earlier proposed seed-44 batch-8 run was replaced by a batch-16 retrain, and the new model was re-evaluated. Because the earlier model had already been evaluated on test, seed 44 has effectively seen the test set twice; this is disclosed in `logs/DECISIONS.md`.
- **Ablation sweeps were compute-constrained on CPU.** As disclosed in `logs/DECISIONS.md`, multi-epoch sweeps for A1–A4 and 10%/100% data efficiency are documented as a formal experimental protocol and design matrix rather than reporting unverified numbers.
- Single dataset, single task, image-level official splits. Patient-level leakage was not separately audited.
- Calibration is moderate (proposed ECE about 0.100). AUROC is high, but thresholded F1 is lower, especially for VIRUS.
- Grad-CAM is qualitative. Some incorrect cases attend to borders or text markers, so saliency maps are not evidence of correct localization.
- This is a research prototype, not a diagnostic device.

---

## Project structure

```text
configs/
  base.yaml                  # shared data, model, training, evaluation settings
  baseline.yaml              # scratch CNN
  proposed.yaml              # ResNet-18 + staged fine-tuning
  ablations/                 # ablation configs (backbone regimes, augmentation, regularization, class weighting)

data/
  raw/pneumonia/             # MedIMeta data (not committed)
  splits/
    train25_seed42.json      # official fixed subset for all main experiments
    train10_seed42.json      # supplementary
    train100_seed42.json     # supplementary
    split_indices.json       # split manifest

src/
  data/
    dataset.py               # MedIMeta wrapper, transforms, loaders
    subset.py                # stratified subset creation / verification
  models/
    baseline_cnn.py          # baseline architecture
    proposed.py              # ResNet-18 / EfficientNet-B0 factory, staged freezing
  train.py                   # shared training loop (never loads test)
  evaluate.py                # one-time final validation/test evaluation
  analysis.py                # aggregation, figures, bootstrap, Grad-CAM
  utils.py                   # config, seeding, metrics, environment logging

scripts/
  make_subset.py             # subset generation and verification
  run_experiments.py         # reproducible command runner

notebooks/
  01_data_exploration.ipynb
  02_baseline_training.ipynb
  03_proposed_finetuning.ipynb
  04_ablations_and_data_efficiency.ipynb
  05_error_analysis.ipynb
  06_working_demonstration.ipynb

report/
  final_report.md
  ai_usage_and_contributions.md

logs/
  DECISIONS.md
  runs/

results/
  baseline/
  proposed/
  figures/
  summary_table.md
  summary_table.csv
  bootstrap_comparison.json
  bacterial_viral_confusion.json

checkpoints/
```

---

## Working demonstration

To run a comparative diagnostic demonstration:
```powershell
# Interactive CLI demo:
python scripts/demo.py --class-name VIRUS

# Or open and run the notebook:
notebooks/06_working_demonstration.ipynb
```

---

## Report and documentation

- Full write-up: `report/final_report.md`
- AI-usage and contribution statement: `report/ai_usage_and_contributions.md`
- Design decisions and test policy: `logs/DECISIONS.md`
- Executable Notebooks: `notebooks/01`–`06`
