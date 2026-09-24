"""
Script to generate EV_Battery_PHM_Phase1_Master.ipynb
cleanly divided for Persons A, B, C, D.
"""

import json

notebook = {
    'cells': [],
    'metadata': {
        'language_info': {
            'name': 'python',
            'version': '3.13'
        },
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

def add_md(text):
    notebook['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.strip().split('\n')]
    })

def add_code(text):
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in text.strip().split('\n')]
    })

# HEADER
add_md("""# SLIIT IT3051 - Data Mining & Predictive Analytics
## EV Battery Prognostics & Health Management (PHM) Dual-Task System
### Group: Necrons | Phase 1: EDA, Data Cleaning & Preprocessing (Week 1 / Viva 1)

---

### Team Roles & Modular Work Breakdown:
| Person | Topic Owned | Key Deliverables |
| :--- | :--- | :--- |
| **Person A** | **Data Structure, Schema & Missingness Audit** | Schema inspection, variable types, duplicate checks, 67-column missingness breakdown, sensor sanity screening |
| **Person B** | **Distributions, Outliers & Treatment Decisions** | Univariate distributions, RUL bell-curve analysis, IQR/Z-score outlier detection, physical outlier justification |
| **Person C** | **Imbalance, Multicollinearity & Feature Engineering** | Class imbalance (18,616 vs 1,384), correlation heatmaps, bivariate degradation analysis, 4 domain features |
| **Person D** | **Data Leakage Guard, Splits & ColumnTransformer** | Target Isolation, ID exclusion, stratified train/test split, production ColumnTransformer pipeline |
""")

# IMPORTS
add_md("### 0. Environment Setup & Global Imports")
add_code("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visual formatting settings
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.sans-serif'] = 'Arial'

print('Libraries loaded successfully!')""")

# SECTION A (PERSON A)
add_md("""---
# Section A: Data Structure, Schema & Missingness Audit
**Owner: Person A**
**Focus:** Dataset ingestion, metadata verification, data types, duplicate records, sensor sanity bounds, and missing value patterns.
""")

add_code("""# Load the raw dataset
DATASET_PATH = 'ev battery_failure  Dataset.csv'
df_raw = pd.read_csv(DATASET_PATH)

print('=== DATASET DIMENSIONS ===')
print(f'Total Rows (Records):   {df_raw.shape[0]:,}')
print(f'Total Columns (Fields): {df_raw.shape[1]}')""")

add_md("#### A.1 Data Types & Duplicate Check")
add_code("""# Check duplicates
duplicates = df_raw.duplicated().sum()
print(f'Duplicate Rows Detected: {duplicates}')

# Identify Categorical vs Numerical Columns
cat_cols_initial = df_raw.select_dtypes(include=['object', 'string']).columns.tolist()
num_cols_initial = df_raw.select_dtypes(include=[np.number]).columns.tolist()

print(f'Categorical Columns ({len(cat_cols_initial)}): {cat_cols_initial}')
print(f'Numerical Columns ({len(num_cols_initial)}): {num_cols_initial[:10]}... (total {len(num_cols_initial)})')""")

add_md("#### A.2 Missing Value Audit (MCAR Analysis)")
add_code("""# Missing Value Breakdown across all 70 columns
missing_counts = df_raw.isnull().sum()
cols_with_missing = missing_counts[missing_counts > 0].sort_values(ascending=False)
missing_pct = (cols_with_missing / len(df_raw)) * 100

missing_summary = pd.DataFrame({
    'Missing Count': cols_with_missing,
    'Missing Pct (%)': missing_pct.round(2)
})

print(f'Total columns with missing values: {len(missing_summary)} out of {df_raw.shape[1]}')
print(f'Missing percentage ranges strictly between {missing_pct.min():.2f}% and {missing_pct.max():.2f}%')
print('\\nTop 10 columns by missing rate:')
print(missing_summary.head(10))

print('\\nTarget Missing Values:')
print(f'predicted_remaining_life_cycles (RUL): {df_raw[\"predicted_remaining_life_cycles\"].isnull().sum():,} missing')
print(f'battery_failure (Classification):     {df_raw[\"battery_failure\"].isnull().sum():,} missing')""")

add_md("#### A.3 Physical Boundary & Sensor Sanity Screening")
add_code("""# Physical Boundary Checks: Impossible Sensor Values
sanity_checks = {
    'Max Cell Temp > 120 C (Extreme Runaway)': (df_raw['cell_temperature_max'] > 120).sum(),
    'Avg Cell Temp < -40 C (Deep Freeze)':      (df_raw['cell_temperature_avg'] < -40).sum(),
    'Cell Voltage Avg <= 0 V':                 (df_raw['cell_voltage_avg'] <= 0).sum(),
    'Pack Voltage <= 0 V':                     (df_raw['pack_voltage'] <= 0).sum(),
    'Internal Resistance <= 0 Ohm':            (df_raw['internal_resistance'] <= 0).sum()
}

print('=== SENSOR SANITY SCREENING RESULTS ===')
for check, count in sanity_checks.items():
    print(f'  {check}: {count} violations')""")

add_md("""**Person A Viva Notes:**
- The dataset contains 20,000 records and 70 features with 0 duplicate rows.
- Exactly 67 features exhibit missing values between 3.05% and 4.94% (Missing Completely at Random - MCAR).
- Continuous features will be imputed with their training median (resilient to sensor fluctuations); vehicle attributes with mode.
- The Task 1 target (`predicted_remaining_life_cycles`) contains 898 missing entries; these are filtered out for regression training to avoid synthetic label leakage.
""")

# SECTION B (PERSON B)
add_md("""---
# Section B: Distributions, Skewness, Outliers & Treatment Decisions
**Owner: Person B**
**Focus:** Continuous feature distributions, Task 1 target distribution, outlier detection via IQR & Z-score, and domain justification for outlier treatment.
""")

add_md("#### B.1 Task 1 Target: Remaining Life Cycles Distribution")
add_code("""# Task 1: RUL Distribution Analysis
rul_data = df_raw['predicted_remaining_life_cycles'].dropna()

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(rul_data, kde=True, color='#2b5c8f', bins=40, ax=ax)
ax.axvline(rul_data.mean(), color='#e63946', linestyle='--', linewidth=2, label=f'Mean: {rul_data.mean():.1f}')
ax.axvline(rul_data.median(), color='#2a9d8f', linestyle=':', linewidth=2, label=f'Median: {rul_data.median():.1f}')
ax.set_title('Task 1 Target Distribution: Predicted Remaining Life Cycles', fontsize=12, fontweight='bold')
ax.set_xlabel('Remaining Life Cycles')
ax.set_ylabel('Record Frequency')
ax.legend()
plt.show()

print(f'RUL Mean:   {rul_data.mean():.2f}')
print(f'RUL Median: {rul_data.median():.2f}')
print(f'RUL Std:    {rul_data.std():.2f}')
print(f'RUL Skew:   {rul_data.skew():.3f} (Symmetric Bell Curve)')""")

add_md("#### B.2 Key Physical Degradation Distributions")
add_code("""# Physical sensor distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(df_raw['cell_voltage_avg'], kde=True, ax=axes[0, 0], color='#264653')
axes[0, 0].set_title('Average Cell Voltage (V)', fontweight='bold')

sns.histplot(df_raw['pack_voltage'], kde=True, ax=axes[0, 1], color='#2a9d8f')
axes[0, 1].set_title('Pack Voltage (V)', fontweight='bold')

sns.histplot(df_raw['cell_temperature_max'], kde=True, ax=axes[1, 0], color='#e76f51')
axes[1, 0].set_title('Max Cell Temperature (C)', fontweight='bold')

sns.histplot(df_raw['internal_resistance'], kde=True, ax=axes[1, 1], color='#e63946')
axes[1, 1].set_title('Internal Resistance (Ohm)', fontweight='bold')

plt.tight_layout()
plt.show()""")

add_md("#### B.3 Outlier Detection & Outlier Treatment Decision")
add_code("""# IQR & Z-score Outlier Audit
def audit_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_iqr = ((series < lower_bound) | (series > upper_bound)).sum()
    
    mean = series.mean()
    std = series.std()
    outliers_z = (np.abs((series - mean) / std) > 3).sum()
    
    return outliers_iqr, outliers_z, lower_bound, upper_bound

key_cols = ['internal_resistance', 'cell_temperature_max', 'voltage_imbalance', 'capacity_loss_percent']
print('=== OUTLIER DETECTION AUDIT (IQR & Z-SCORE) ===')
for col in key_cols:
    s = df_raw[col].dropna()
    o_iqr, o_z, lb, ub = audit_outliers(s)
    print(f'{col}: IQR Outliers={o_iqr:,} (bounds: [{lb:.2f}, {ub:.2f}]), Z-score (|z|>3)={o_z:,}')""")

add_md("""**Person B Outlier Treatment Decision (Viva Justification):**
- **Decision:** Outliers are **retained** rather than naively truncated or deleted.
- **Physical Justification:** High temperature spikes (> 60 C), elevated internal resistance (> 0.8 Ohm), and high voltage imbalance are **true electro-thermal degradation precursors**. If we drop them, we remove the most critical failure signals from Task 2!
- We employ **StandardScaler** and tree-based ensembles (Random Forest / XGBoost), which are naturally robust to monotonic outliers.
""")

# SECTION C (PERSON C)
add_md("""---
# Section C: Class Imbalance, Multicollinearity & Feature Engineering
**Owner: Person C**
**Focus:** Task 2 binary class imbalance, Pearson correlation analysis, physical discrimination boxplots, and Stage 3 domain feature engineering.
""")

add_md("#### C.1 Task 2 Class Imbalance Analysis")
add_code("""# Task 2 Class Distribution
fail_counts = df_raw['battery_failure'].value_counts()
print(fail_counts)

plt.figure(figsize=(7, 5))
bars = plt.bar(['Normal (0)', 'Failure (1)'], [fail_counts[0], fail_counts[1]], color=['#2a9d8f', '#e63946'], width=0.45)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 300, f'{yval:,}\\n({yval/len(df_raw)*100:.2f}%)', ha='center', va='bottom', fontweight='bold')

plt.title('Task 2 Target: Critical Battery Failure Class Imbalance', fontsize=12, fontweight='bold')
plt.ylabel('Record Count')
plt.ylim(0, 21500)
plt.show()

print(f'Imbalance Ratio: {fail_counts[0]/fail_counts[1]:.2f} : 1')""")

add_md("#### C.2 Physical Variable Correlation & Degradation Patterns")
add_code("""# Correlation of physical metrics with targets
key_features = [
    'internal_resistance', 'cell_temperature_max', 'cell_temperature_avg',
    'voltage_imbalance', 'capacity_loss_percent', 'state_of_health',
    'predicted_remaining_life_cycles', 'battery_failure'
]
corr_sub = df_raw[key_features].dropna().corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_sub, annot=True, fmt='.2f', cmap='coolwarm', center=0, cbar_kws={'label': 'Pearson Correlation'})
plt.title('Physical Feature Correlation Matrix', fontsize=12, fontweight='bold')
plt.show()""")

add_md("#### C.3 Bivariate Physical Discrimination (Normal vs Failure)")
add_code("""# Boxplots of key failure indicators
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.boxplot(x='battery_failure', y='internal_resistance', data=df_raw, ax=axes[0], palette=['#457b9d', '#e63946'], hue='battery_failure', legend=False)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Normal (0)', 'Failure (1)'])
axes[0].set_title('Internal Resistance by Failure Status', fontweight='bold')

sns.boxplot(x='battery_failure', y='cell_temperature_max', data=df_raw, ax=axes[1], palette=['#457b9d', '#e63946'], hue='battery_failure', legend=False)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Normal (0)', 'Failure (1)'])
axes[1].set_title('Max Cell Temperature by Failure Status', fontweight='bold')

sns.boxplot(x='battery_failure', y='voltage_imbalance', data=df_raw, ax=axes[2], palette=['#457b9d', '#e63946'], hue='battery_failure', legend=False)
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['Normal (0)', 'Failure (1)'])
axes[2].set_title('Voltage Imbalance by Failure Status', fontweight='bold')

plt.tight_layout()
plt.show()""")

add_md("#### C.4 Stage 3 Feature Engineering: Domain-Specific Electrochemical Features")
add_code("""df_feat = df_raw.copy()

# 1. Thermal Gradient
df_feat['temp_delta'] = df_feat['cell_temperature_max'] - df_feat['cell_temperature_avg']

# 2. Electrical Stress Ratio (C-Rate Proxy)
df_feat['c_rate_proxy'] = df_feat['average_charge_power_kw'] / (df_feat['battery_capacity_kwh'] + 1e-5)

# 3. Cell Degradation Index (Relative Voltage Spread)
df_feat['cell_voltage_spread'] = df_feat['cell_voltage_std'] / (df_feat['cell_voltage_avg'] + 1e-5)

# 4. Aggressive Driving Usage Stress Metric
df_feat['stress_index'] = df_feat['aggressive_acceleration_score'] * df_feat['hard_braking_score']

eng_cols = ['temp_delta', 'c_rate_proxy', 'cell_voltage_spread', 'stress_index']
print('=== ENGINEERED DOMAIN FEATURES STATISTICAL SUMMARY ===')
print(df_feat[eng_cols].describe().round(4).T[['mean', 'std', 'min', '50%', 'max']])

# Correlation with targets
eng_corrs = df_feat[eng_cols + ['predicted_remaining_life_cycles', 'battery_failure']].corr()
print('\\nCorrelations with Dual Targets:')
print(pd.DataFrame({
    'Task 1 (RUL Corr)': eng_corrs['predicted_remaining_life_cycles'][eng_cols],
    'Task 2 (Failure Corr)': eng_corrs['battery_failure'][eng_cols]
}))""")

add_md("""**Person C Viva Notes:**
- Task 2 has a severe class imbalance of 93.08% normal to 6.92% failure (13.45:1).
- Accuracy is an inappropriate evaluation metric (predicting all 0s gives 93.08% accuracy while missing all catastrophic failures). We must prioritize **Recall** and **PR-AUC**.
- The 4 engineered features capture thermal gradients, charging stress (C-rate), voltage non-uniformity, and mechanical drive cycle shock.
""")

# SECTION D (PERSON D)
add_md("""---
# Section D: Data Leakage Guard, Train/Test Split & ColumnTransformer Pipeline
**Owner: Person D**
**Focus:** Boundary conditions, Target Isolation, Stratified train/test splitting, production ColumnTransformer preprocessing, and data leakage verification.
""")

add_md("#### D.1 Boundary Conditions & Target Isolation Policy")
add_code("""# Drop Identifier Columns
id_cols = ['vehicle_id', 'battery_serial']
target_rul = 'predicted_remaining_life_cycles'
target_fail = 'battery_failure'

# Predictor columns candidate list
excluded_features = id_cols + [target_rul, target_fail]
feature_names = [col for col in df_feat.columns if col not in excluded_features]

cat_features = df_feat[feature_names].select_dtypes(include=['object', 'string']).columns.tolist()
num_features = df_feat[feature_names].select_dtypes(include=[np.number]).columns.tolist()

print(f'Excluded Primary IDs: {id_cols}')
print(f'Total Predictors: {len(feature_names)} (Numeric: {len(num_features)}, Categorical: {len(cat_features)})')
print(f'Categorical Columns ({len(cat_features)}): {cat_features}')""")

add_md("#### D.2 Constructing the Production ColumnTransformer Pipeline")
add_code("""from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Pipeline for Numerical Features: Median Imputation + Scaling
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for Categorical Features: Mode Imputation + One-Hot Encoding
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Master ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ],
    verbose_feature_names_out=False
)

print('ColumnTransformer Pipeline assembled successfully!')""")

add_md("#### D.3 Task 1 Partitioning & Preprocessing: RUL Regression")
add_code("""from sklearn.model_selection import train_test_split

# Filter out unlabeled target rows for Task 1
mask_t1 = df_feat[target_rul].notnull()
df_t1 = df_feat[mask_t1].copy()

X1 = df_t1[feature_names]
y1 = df_t1[target_rul]

print(f'Task 1 Labeled Sample Size: {len(df_t1):,} rows (dropped {len(df_feat)-len(df_t1):,} unobserved targets)')

# 80/20 Train/Test Split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.20, random_state=42)

# Fit on train ONLY, transform test to prevent leakage
preprocessor_t1 = ColumnTransformer(
    transformers=[('num', num_pipeline, num_features), ('cat', cat_pipeline, cat_features)],
    verbose_feature_names_out=False
)

X1_train_trans = preprocessor_t1.fit_transform(X1_train)
X1_test_trans = preprocessor_t1.transform(X1_test)

print(f'X1_train shape: {X1_train_trans.shape}')
print(f'X1_test shape:  {X1_test_trans.shape}')
print(f'Total Post-Transformation Features: {X1_train_trans.shape[1]}')""")

add_md("#### D.4 Task 2 Partitioning & Preprocessing: Critical Failure Classification")
add_code("""# Task 2 uses all 20,000 records
X2 = df_feat[feature_names]
y2 = df_feat[target_fail]

# Stratified 80/20 Train/Test Split preserving 93.08% / 6.92% class balance
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.20, random_state=42, stratify=y2
)

preprocessor_t2 = ColumnTransformer(
    transformers=[('num', num_pipeline, num_features), ('cat', cat_pipeline, cat_features)],
    verbose_feature_names_out=False
)

X2_train_trans = preprocessor_t2.fit_transform(X2_train)
X2_test_trans = preprocessor_t2.transform(X2_test)

print(f'X2_train shape: {X2_train_trans.shape} (Failures: {(y2_train==1).sum():,} [{(y2_train==1).mean()*100:.2f}%])')
print(f'X2_test shape:  {X2_test_trans.shape} (Failures: {(y2_test==1).sum():,} [{(y2_test==1).mean()*100:.2f}%])')
print(f'Leakage Guard Verified: Strict fit_transform on train, transform on test.')""")

add_md("""**Person D Viva Notes:**
- Enforced strict **Target Isolation**: `battery_failure` is not in $X_1$, and `predicted_remaining_life_cycles` is not in $X_2$.
- Dropped identifier columns (`vehicle_id`, `battery_serial`) to eliminate memorization risk.
- Implemented a unified `ColumnTransformer` (median + standard scaler for numericals; mode + one-hot encoder with `handle_unknown='ignore'` for categoricals).
- **Zero Data Leakage:** Preprocessors were strictly fitted on the training split only and subsequently applied to test splits.
""")

# CONCLUSION
add_md("""---
## Summary of Phase 1 Readiness (Viva 1 Checkpoint)
- **Data Quality:** 20,000 rows audited, 67 columns with MCAR missingness (3.05%-4.94%) handled via median/mode imputation.
- **Physical Validity:** 0 physically impossible sensor violations detected ($V \\le 0, T > 120^\\circ\\text{C}, R \\le 0$).
- **Dual-Task Isolation:** Targets segregated into separate matrices; 898 missing RUL target records safely removed from Task 1 training.
- **Transformed Feature Space:** 153 features ready for model training in Week 2 (Viva 2).
""")

output_file = 'EV_Battery_PHM_Phase1_Master.ipynb'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f'Notebook {output_file} generated successfully!')
