# Pediatric Pneumonia Classification with Limited Labeled Data (Project P06)
**CS320: Artificial Neural Networks & Deep Learning — Term Project (BSAI-2k24, Fall 2026)**  
**National University of Sciences & Technology (NUST), SEECS**  
**Team Members:** Sharjeel Farooq, Affan Ahmad, Mohammad Asef

All source code, configuration files, trained evaluation outputs, executed notebooks, live demonstration tools, and technical reports are located inside the main project directory:

👉 **[`p06-pneumonia-limited-data/`](./p06-pneumonia-limited-data/)**

---

### Quick Start & Demonstration
To launch the interactive model demonstration from this root directory:
```powershell
python p06-pneumonia-limited-data/scripts/demo.py --class-name VIRUS
```
Or open the executed demonstration notebook:
[`p06-pneumonia-limited-data/notebooks/06_working_demonstration.ipynb`](./p06-pneumonia-limited-data/notebooks/06_working_demonstration.ipynb)

---

### Key Deliverables Directory Map
- **Technical Report:** [`p06-pneumonia-limited-data/report/final_report.md`](./p06-pneumonia-limited-data/report/final_report.md)
- **AI Tool & Contribution Statement:** [`p06-pneumonia-limited-data/report/ai_usage_and_contributions.md`](./p06-pneumonia-limited-data/report/ai_usage_and_contributions.md)
- **Experimental Decisions & Protocol:** [`p06-pneumonia-limited-data/logs/DECISIONS.md`](./p06-pneumonia-limited-data/logs/DECISIONS.md)
- **Results & Figures:** [`p06-pneumonia-limited-data/results/`](./p06-pneumonia-limited-data/results/)
- **Executed Jupyter Notebooks:** [`p06-pneumonia-limited-data/notebooks/`](./p06-pneumonia-limited-data/notebooks/)
  - `01_data_exploration.ipynb`
  - `02_baseline_training.ipynb`
  - `03_proposed_finetuning.ipynb`
  - `04_ablations_and_data_efficiency.ipynb`
  - `05_error_analysis.ipynb`
  - `06_working_demonstration.ipynb`

---

### Trained Model Checkpoints
Because PyTorch weights files (`.pt`) exceed standard Git repository storage limits and are ignored by `.gitignore`, all 6 trained model checkpoints (Seeds 42, 43, 44 for Baseline CNN and Proposed ResNet-18) are archived externally:
- **GitHub Release Archive:** [v1.0 Checkpoints Bundle](https://github.com/SharjeelFarooq2024/Pneumonia_classification/releases/tag/v1.0-checkpoints)
- **Google Drive Mirror:** [CS320-P06 Checkpoints Mirror](https://drive.google.com/drive/folders/1A_P06_Pneumonia_Model_Weights_CS320) *(Placeholder: Update with actual Drive folder link upon upload)*

To run evaluation scripts (`src/evaluate.py`) or demo scripts without retraining, place the downloaded weights in:
`p06-pneumonia-limited-data/checkpoints/{baseline,proposed}/seed{42,43,44}/best_model.pt`
