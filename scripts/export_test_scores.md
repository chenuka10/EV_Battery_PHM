# Artifact Export Documentation: Held-Out Test Scores & Feature Ranges

**Project:** EV Battery Prognostics & Health Management (PHM)  
**Course:** SLIIT IT3051 Fundamentals of Data Mining (Mini Project 2026)  
**Group:** Necrons  

---

## 1. Overview
To ensure strict separation of concerns, zero data leakage, and lightning-fast edge performance, two pre-computed evaluation artifacts are persisted in the `models/` directory:
1. `models/test_scores.csv`: Ground-truth labels and predicted probability vectors from both candidate classification engines across the held-out 4,000-row test partition ($y_{\text{true}}$, $p_{\text{lr}}$, $p_{\text{xgb}}$).
2. `models/feature_ranges.json`: Distributional metadata (min, max, median, 25th percentile, 75th percentile, IQR, and upper outlier cutoff $Q_3 + 1.5 \times \text{IQR}$) computed exclusively on the training partition.

---

## 2. Export Notebook / Python Snippet
The following script can be executed directly from Jupyter (`EV_battery_PHM_viva2.ipynb`) or as a standalone script using `.venv\Scripts\python.exe`:

```python
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load Dataset
data_path = "dataset/ev_battery_health_data.csv"
df = pd.read_csv(data_path)

# Drop missing target rows for Task 1 as established in Section 3 of Technical Report
df_clean = df.dropna(subset=['predicted_remaining_life_cycles', 'battery_failure']).copy()

# 2. Re-create Domain Feature Engineering (The 7 Verified Features)
df_clean['temperature_spread'] = df_clean['cell_temperature_max'] - df_clean['cell_temperature_avg']
df_clean['efficiency_gap'] = (df_clean['charge_efficiency'] - df_clean['discharge_efficiency']).abs()
df_clean['health_loss_interaction'] = (df_clean['battery_health_percent'] * df_clean['capacity_loss_percent']) / 100.0
df_clean['resistance_per_1000_cycles'] = (df_clean['internal_resistance'] / (df_clean['cycle_count'] + 1.0)) * 1000.0
df_clean['c_rate_proxy'] = df_clean['average_charge_power_kw'] / (df_clean['battery_capacity_kwh'] + 1e-5)
df_clean['cell_voltage_spread'] = df_clean['cell_voltage_std'] / (df_clean['cell_voltage_avg'] + 1e-5)
df_clean['stress_index'] = df_clean['aggressive_acceleration_score'] * df_clean['hard_braking_score']

# 3. Load Trained Artifacts
MODELS_DIR = "models"
ct = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
model_lr = joblib.load(os.path.join(MODELS_DIR, "champion_failure_lr.joblib"))
model_xgb = joblib.load(os.path.join(MODELS_DIR, "champion_failure_xgb.joblib"))
with open(os.path.join(MODELS_DIR, "metadata.json"), "r") as f:
    meta = json.load(f)

# 4. Perform Exactly Identical 80/20 Stratified Partition
X = df_clean[meta['features']]
y_cls = df_clean['battery_failure'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cls, test_size=0.20, random_state=42, stratify=y_cls
)

# 5. Transform Held-Out Test Set & Infer Probabilities
X_test_trans = ct.transform(X_test)
p_lr = model_lr.predict_proba(X_test_trans)[:, 1]
p_xgb = model_xgb.predict_proba(X_test_trans)[:, 1]

# 6. Export test_scores.csv
df_scores = pd.DataFrame({
    'y_true': y_test.values,
    'p_lr': p_lr,
    'p_xgb': p_xgb
})
df_scores.to_csv(os.path.join(MODELS_DIR, "test_scores.csv"), index=False)
print(f"Exported test_scores.csv: {len(df_scores)} records, {df_scores['y_true'].sum()} ground-truth failures.")

# 7. Compute Training Set Bounds & Export feature_ranges.json
ranges = {}
for col in X_train.columns:
    if np.issubdtype(X_train[col].dtype, np.number):
        s = X_train[col].dropna()
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        ranges[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "iqr_lower": float(q1 - 1.5 * iqr),
            "iqr_upper": float(q3 + 1.5 * iqr)
        }

with open(os.path.join(MODELS_DIR, "feature_ranges.json"), "w") as f:
    json.dump(ranges, f, indent=2)
print(f"Exported feature_ranges.json: {len(ranges)} numerical attributes recorded.")
```

---

## 3. Mathematical Traceability Verification
When loaded by the backend:
- `y_true.sum()` evaluates to exactly **277 failures** in **4,000 held-out test rows** (prevalence = $6.925\%$, class imbalance ratio = $13.45:1$).
- At safety decision threshold $\tau = 0.19259366$ (printed as $\tau = 0.193$ in Technical Report Table 13):
  - True Positives (TP): **250**
  - False Negatives (Missed Failures, FN): **27**
  - False Positives (False Alarms, FP): **139**
  - True Negatives (TN): **3,584**
  - Recall: $\frac{250}{250 + 27} = 90.25\%$
  - Precision: $\frac{250}{250 + 139} = 64.27\%$
  - F1-Score: $75.08\%$
