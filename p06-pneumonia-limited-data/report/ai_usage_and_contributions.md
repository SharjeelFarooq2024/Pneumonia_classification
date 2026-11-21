# AI Usage and Contributions Statement
**CS320: Artificial Neural Networks & Deep Learning — Term Project P06**  
**Group:** P06 (Members: Sharjeel Farooq, Affan Ahmad, and [Member 3 Name — To Be Confirmed])  
**National University of Sciences & Technology (NUST), SEECS**

---

## 1. Tools and Computing Environment
- **AI Tools Consulted:** Generative AI assistants (including Google Antigravity / Gemini CLI / conversational LLM interfaces) were actively consulted and utilized across various phases of project development.
- **Scientific Stack:** Python 3.10+, PyTorch (`torch`), `torchvision`, `scikit-learn`, `pandas`, `NumPy`, `matplotlib`, `seaborn`, `medimeta`, and `pytest`.
- **Hardware & Execution Environment:** Local Intel x86_64 CPU workstation (8 logical cores, Windows 10/11 environment, batch size 16, `num_workers=0`).

---

## 2. Nature and Scope of AI Assistance (Full Disclosure)
In accordance with Step 9 of the CS320 project instructions, we declare that generative AI tools were used for code generation, report drafting, testing, and debugging:

1. **Code Generation & Architecture Refactoring:**
   - Generating modular OOP domain abstractions (`ChestXRayImage`, `DiagnosticPrediction`, `ModelArchitecture`, `EvaluationRun`).
   - Implementing DSA-inspired algorithms for metric tracking (min-heap top-$K$ error analysis) and vector-based calibration curve bins.
   - Refactoring procedural scripts into structured service layers (`src/services/`) and building interactive CLI tools (`scripts/demo.py`, `scripts/predict_image.py`).
   - Scaffolding a comprehensive automated test suite (43 unit and integration tests across data pipelines, architectures, configs, and integrity checks in `tests/`).

2. **Technical Report & Documentation Drafting:**
   - Drafting initial sections of the comprehensive technical report (`report/final_report.md`) based on experimental logs and project specifications.
   - Generating markdown and LaTeX summary tables (`results/summary_table.md`, `results/summary_table.csv`).
   - Structuring reproducible Jupyter notebooks (`notebooks/01` through `06`).

3. **Debugging, Diagnostics & Environment Fixes:**
   - Resolving Windows-specific path handling (`pathlib.Path` cross-platform resolutions).
   - Diagnosing non-interactive terminal encoding errors (`colorama` / `TERM=dumb`).
   - Optimizing PyTorch CPU DataLoader configurations to mitigate host memory exhaustion.

4. **Verification & Authenticity Audit:**
   - AI tools assisted in identifying inconsistencies in handwritten preliminary ablation tables, leading to their complete removal and replacement with an authentic, verified experimental design matrix.

---

## 3. Human Intellectual Contribution (What the Group Did)
The group members conceived, directed, validated, and maintain full intellectual ownership and understanding of the project:
- **Scientific Formulation:** Selected Project P06 and framed the core research hypothesis: assessing how ImageNet transfer learning with staged progressive unfreezing compensates for severe training label scarcity (25% budget).
- **Experimental Protocol:** Enforced strict methodological integrity: single generation of the stratified 25% training split (`data/splits/train25_seed42.json`), validation-only model selection, and one-time test split evaluation guarded by filesystem markers (`test_eval_done.json`).
- **Architectural Design & Decisions:** Formulated the 3-stage fine-tuning schedule (head -> layer4 -> full network), discriminative layer learning rates ($10^{-5}$ to $10^{-3}$), domain-appropriate augmentations (strictly no horizontal flips to preserve thoracic anatomical laterality), subset inverse class weighting, and label smoothing.
- **Critical Error Analysis:** Analyzed confusion matrices and Grad-CAM saliency heatmaps to identify clinical mechanisms (bacterial consolidation vs. viral interstitial markings) behind the failure modes.
- **Authenticity & Integrity:** All primary benchmark numbers (3-seed Baseline vs. 3-seed Proposed) were produced by actual model training and evaluated on disk. Where compute limits prevented multi-epoch ablation sweeps on CPU, the group made the deliberate choice to report them honestly as an experimental design matrix and future work rather than presenting fabricated metrics.

---

## 4. Division of Group Responsibilities
In accordance with course guidelines for 3-member projects:
- **Sharjeel Farooq (Data Engineering & Baseline Architecture):**
  - MedIMeta data pipeline and environment setup.
  - Stratified 25% subset generation (`data/splits/train25_seed42.json`) and data sanity checks.
  - Implementation and training of the reference 3-block scratch CNN baseline across seeds 42, 43, and 44.
  - Logging of initial baseline metrics and failure mode documentation.

- **Affan Ahmad (Proposed Model, Transfer Learning & OOP/DSA System):**
  - Implementation of ImageNet-pretrained ResNet-18 architecture with staged progressive unfreezing.
  - Discriminative learning rate schedulers, subset inverse class weighting, and label smoothing loss functions.
  - Domain-specific augmentation pipeline (affine transforms without horizontal flips).
  - OOP and DSA modular architectural refactoring and unit test suite design.
  - Git repository management and technical report synthesis.

- **[Member 3 Name — To Be Confirmed] (Evaluation, Statistics & Working Demonstration):**
  - Implementation of the one-time evaluation harness (`src/evaluate.py`).
  - Paired bootstrap statistical analysis (2,000 resamples per seed, 95% confidence intervals).
  - Bacterial-versus-viral confusion metrics and diagnostic analysis.
  - Grad-CAM visual explanation generation (`src/analysis.py`).
  - Interactive demonstration tools (`notebooks/06_working_demonstration.ipynb` and `scripts/demo.py`).

- **All Members:**
  - Joint failure mode analysis, report review, code review, and full end-to-end understanding in preparation for the project viva examination.

---

## 5. Group Declaration
> *"We confirm that the conceptual design, experimental strategy, scientific analysis, and conclusions in this project represent our own intellectual work. Generative AI tools were used transparently as programming and drafting assistants across code, tests, documentation, and diagnostics as detailed above. All group members understand the entire system pipeline and are fully prepared to explain and defend every component during the oral viva."*
