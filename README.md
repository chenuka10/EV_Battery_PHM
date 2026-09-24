# SLIIT IT3051 - EV Battery Prognostics & Health Management (PHM)
**Group:** Necrons  
**Dataset:** `ev battery_failure  Dataset.csv` (20,000 records x 70 columns)  
**System Architecture:** Dual-Task PHM System (Task 1: RUL Regression | Task 2: Failure Classification)

---

## Repository Structure & Team Ownership

```
EV_Battery_project/
├── .gitignore
├── README.md
├── requirements.txt
├── ev battery_failure  Dataset.csv       # Raw Telemetry Dataset
├── EV_Battery_PHM_Phase1_Master.ipynb    # Unified Master Notebook (for Viva submission)
├── stage1_2_exploratory_analysis.py      # Automated EDA script
├── stage4_pipeline_and_splits.py         # Pipeline & data split script (Person D)
├── eda_plots/                            # Exported charts & visualizations
└── notebooks/                            # Modular topic notebooks (Zero Merge Conflicts)
    ├── 01_data_audit_personA.ipynb       # Person A's workspace
    ├── 02_distributions_outliers_personB.ipynb  # Person B's workspace
    ├── 03_relationships_features_personC.ipynb  # Person C's workspace
    └── 04_pipeline_split_personD.ipynb   # Person D's workspace
```

---

## Quick Setup Guide for Teammates

### 1. Clone the Repository
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd EV_Battery_project
```

### 2. Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Team Ownership Matrix (Phase 1 / Viva 1)

| Member | Topic | Notebook / Script | Deliverables |
| :--- | :--- | :--- | :--- |
| **Person A** | Data Structure & Audit | `notebooks/01_data_audit_personA.ipynb` | Schema verification, 67-column MCAR missingness audit, duplicate checks, sensor sanity bounds ($V \le 0, T > 120^\circ\text{C}$). |
| **Person B** | Distributions & Outliers | `notebooks/02_distributions_outliers_personB.ipynb` | Univariate plots, Task 1 target bell curve ($\mu=7,825.36$), IQR & Z-score detection, physical retention justification. |
| **Person C** | Imbalance & Feature Eng. | `notebooks/03_relationships_features_personC.ipynb` | Task 2 class imbalance ($13.45:1$), correlation heatmaps, boxplots, 4 domain-engineered features. |
| **Person D** | Pipeline & Data Splits | `notebooks/04_pipeline_split_personD.ipynb` & `stage4_pipeline_and_splits.py` | Target Isolation, ID exclusion, stratified train/test split, production `ColumnTransformer` (153 output features). |
