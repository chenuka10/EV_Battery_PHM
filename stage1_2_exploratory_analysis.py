"""
SLIIT IT3051 - Data Mining & Predictive Analytics
EV Battery Prognostics & Health Management (PHM) System
Group: Necrons
Week 1: Stage 1 (Problem Definition & Scope) & Stage 2 (EDA & Data Quality Audit)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visual aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Create directory for output plots
OUTPUT_DIR = "eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# STAGE 1: PROBLEM DEFINITION & DUAL-TASK SCOPE AUDIT
# ---------------------------------------------------------
def stage_1_problem_definition(df: pd.DataFrame):
    print("=" * 75)
    print("STAGE 1: DUAL-TASK PHM PROBLEM DEFINITION & TARGET ISOLATION")
    print("=" * 75)
    
    print("\n[1.1] Dataset Dimensions & Metadata:")
    print(f"  - Total Observations (Records): {df.shape[0]:,}")
    print(f"  - Total Features & Attributes:   {df.shape[1]}")
    
    # Task 1: RUL Regression
    target_rul = "predicted_remaining_life_cycles"
    rul_valid = df[target_rul].dropna()
    print("\n[1.2] Task 1: Remaining Life Cycles Regression (RUL)")
    print(f"  - Target Variable: '{target_rul}' (Continuous)")
    print(f"  - Objective: Strategic maintenance planning and battery retirement forecasting")
    print(f"  - Primary Evaluation Metrics: R^2, RMSE, MAE")
    print(f"  - Valid Labels Count: {len(rul_valid):,} | Missing Labels: {df[target_rul].isnull().sum():,} ({df[target_rul].isnull().mean()*100:.2f}%)")
    print(f"  - Distribution Stats: Mean={rul_valid.mean():.2f}, Median={rul_valid.median():.2f}, Std={rul_valid.std():.2f}, Min={rul_valid.min():.1f}, Max={rul_valid.max():.1f}")
    
    # Task 2: Critical Failure Risk Classification
    target_fail = "battery_failure"
    print("\n[1.3] Task 2: Critical Failure Risk Classification")
    print(f"  - Target Variable: '{target_fail}' (Binary: 0=Normal, 1=Critical Risk)")
    print(f"  - Objective: Tactical safety protection preventing catastrophic failure & thermal runaway")
    print(f"  - Primary Evaluation Metrics: ROC-AUC, PR-AUC, Recall (False Negative minimization)")
    fail_counts = df[target_fail].value_counts()
    print(f"  - Class Balance: 0 (Normal) = {fail_counts.get(0, 0):,} ({fail_counts.get(0, 0)/len(df)*100:.2f}%), "
          f"1 (Failure) = {fail_counts.get(1, 0):,} ({fail_counts.get(1, 0)/len(df)*100:.2f}%)")
    
    # Boundary Conditions
    id_cols = ['vehicle_id', 'battery_serial']
    print("\n[1.4] Strict Boundary Conditions & Leakage Prevention:")
    print(f"  - Excluded Primary Identifiers: {id_cols} (Zero predictive generalization; prevent memorization)")
    print("  - Target Isolation Policy:")
    print("    * 'battery_failure' MUST NEVER be used as a predictor for 'predicted_remaining_life_cycles'")
    print("    * 'predicted_remaining_life_cycles' MUST NEVER be used as a predictor for 'battery_failure'")
    print("    * Reason: Simultaneous deployment at runtime; using one target to predict another causes circular data leakage.")


# ---------------------------------------------------------
# STAGE 2: EXPLORATORY DATA ANALYSIS & DATA CLEANING AUDIT
# ---------------------------------------------------------
def stage_2_eda_and_cleaning(df: pd.DataFrame):
    print("\n" + "=" * 75)
    print("STAGE 2: DATA QUALITY AUDIT, PHYSICAL DISTRIBUTIONS & OUTLIERS")
    print("=" * 75)
    
    # 2.1 Missing Value Analysis
    missing_series = df.isnull().sum()
    missing_cols = missing_series[missing_series > 0]
    missing_pct = (missing_cols / len(df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': missing_cols, 'Missing_Pct': missing_pct}).sort_values(by='Missing_Pct', ascending=False)
    
    print("\n[2.1] Missing Value Audit Summary:")
    print(f"  - Total Columns with Missing Values: {len(missing_df)} out of {df.shape[1]}")
    print(f"  - Missing Percentage Range: {missing_df['Missing_Pct'].min():.2f}% to {missing_df['Missing_Pct'].max():.2f}%")
    print("  - Top 5 Columns with Highest Missing Rates:")
    for col, row in missing_df.head(5).iterrows():
        print(f"    * {col}: {int(row['Missing_Count']):,} missing ({row['Missing_Pct']:.2f}%)")
    print("  - Imputation Strategy Justification:")
    print("    * Numeric Sensor Features: Median imputation (robust against local skewed fluctuations) or KNN Imputer.")
    print("    * Categorical Vehicle Specs: Mode / 'Unknown' category imputation.")
    print("    * Target RUL: When training Task 1 models, rows missing RUL labels (898 rows) must be dropped or reserved.")

    # 2.2 Univariate Analysis: Target Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot RUL (Task 1)
    target_rul = "predicted_remaining_life_cycles"
    rul_data = df[target_rul].dropna()
    sns.histplot(rul_data, kde=True, ax=axes[0], color="#2b5c8f", bins=40)
    axes[0].axvline(rul_data.mean(), color='#e63946', linestyle='--', linewidth=2, label=f"Mean: {rul_data.mean():.0f}")
    axes[0].axvline(rul_data.median(), color='#2a9d8f', linestyle=':', linewidth=2, label=f"Median: {rul_data.median():.0f}")
    axes[0].set_title("Task 1: RUL Distribution (Remaining Life Cycles)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicted Remaining Life Cycles")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    
    # Plot Failure (Task 2)
    target_fail = "battery_failure"
    fail_counts = df[target_fail].value_counts()
    bars = axes[1].bar(["Normal (0)", "Failure (1)"], [fail_counts[0], fail_counts[1]], color=["#2a9d8f", "#e63946"], width=0.45)
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 250, f"{h:,}\n({h/len(df)*100:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].set_title("Task 2: Critical Battery Failure Class Imbalance", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Record Count")
    axes[1].set_ylim(0, 22000)
    
    plt.tight_layout()
    target_plot_path = os.path.join(OUTPUT_DIR, "stage2_target_distributions.png")
    plt.savefig(target_plot_path, dpi=200)
    plt.close()
    print(f"\n[2.2] Generated Target Distribution Plot -> {target_plot_path}")

    # 2.3 Physical Variables & Correlation Analysis
    key_physical_vars = [
        'internal_resistance', 'cell_temperature_max', 'cell_temperature_avg',
        'voltage_imbalance', 'capacity_loss_percent', 'state_of_health',
        'predicted_remaining_life_cycles', 'battery_failure'
    ]
    sub_df = df[key_physical_vars].dropna()
    corr_matrix = sub_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': 'Pearson Correlation'})
    plt.title("Correlation Matrix of Key Electrochemical & Operational Variables", fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    corr_plot_path = os.path.join(OUTPUT_DIR, "stage2_physical_correlations.png")
    plt.savefig(corr_plot_path, dpi=200)
    plt.close()
    print(f"[2.3] Generated Physical Correlation Heatmap -> {corr_plot_path}")

    # 2.4 Bivariate Physical Discrimination (Normal vs Failure)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_to_plot = [
        ('internal_resistance', 'Internal Resistance (Ω)', '#e76f51'),
        ('cell_temperature_max', 'Max Cell Temperature (°C)', '#f4a261'),
        ('voltage_imbalance', 'Voltage Imbalance (mV)', '#264653')
    ]
    for idx, (col, label, color) in enumerate(metrics_to_plot):
        sns.boxplot(x='battery_failure', y=col, data=df, ax=axes[idx], palette=['#457b9d', '#e63946'], hue='battery_failure', legend=False)
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(['Normal (0)', 'Failure (1)'])
        axes[idx].set_xlabel("Failure Status")
        axes[idx].set_ylabel(label)
        axes[idx].set_title(f"{label} by Failure Status", fontweight='bold')
    plt.tight_layout()
    bivariate_plot_path = os.path.join(OUTPUT_DIR, "stage2_bivariate_physical_boxplots.png")
    plt.savefig(bivariate_plot_path, dpi=200)
    plt.close()
    print(f"[2.4] Generated Bivariate Boxplots -> {bivariate_plot_path}")

    # 2.5 Physical Boundary Sanity & Sensor Anomaly Detection
    print("\n[2.5] Physical Boundary & Sensor Sanity Screening:")
    
    # Temperature Checks
    t_over_120 = (df['cell_temperature_max'] > 120).sum()
    t_sub_minus40 = (df['cell_temperature_avg'] < -40).sum()
    print(f"  - Sensor Overheat Check (cell_temperature_max > 120 C): {t_over_120} records (physically impossible under normal ops)")
    print(f"  - Sensor Deep Freeze Check (cell_temperature_avg < -40 C): {t_sub_minus40} records")
    
    # Voltage Checks
    v_cell_neg = (df['cell_voltage_avg'] <= 0).sum()
    v_pack_neg = (df['pack_voltage'] <= 0).sum()
    print(f"  - Non-positive Cell Voltage Check (cell_voltage_avg <= 0V): {v_cell_neg} records")
    print(f"  - Non-positive Pack Voltage Check (pack_voltage <= 0V): {v_pack_neg} records")
    
    # Internal Resistance Checks
    ir_neg = (df['internal_resistance'] <= 0).sum()
    print(f"  - Non-positive Internal Resistance Check (internal_resistance <= 0 Ohm): {ir_neg} records")
    
    # Range Sanity Summary Table
    sanity_summary = pd.DataFrame({
        'Feature': ['cell_voltage_avg (V)', 'pack_voltage (V)', 'cell_temperature_avg (C)', 'cell_temperature_max (C)', 'internal_resistance (Ohm)'],
        'Min Observed': [df['cell_voltage_avg'].min(), df['pack_voltage'].min(), df['cell_temperature_avg'].min(), df['cell_temperature_max'].min(), df['internal_resistance'].min()],
        'Max Observed': [df['cell_voltage_avg'].max(), df['pack_voltage'].max(), df['cell_temperature_avg'].max(), df['cell_temperature_max'].max(), df['internal_resistance'].max()],
        'Mean Observed': [df['cell_voltage_avg'].mean(), df['pack_voltage'].mean(), df['cell_temperature_avg'].mean(), df['cell_temperature_max'].mean(), df['internal_resistance'].mean()]
    })
    print("\n  Summary Table of Key Physical Sensor Signals:")
    print(sanity_summary.to_string(index=False))


# ---------------------------------------------------------
# STAGE 3 PREVIEW: DOMAIN FEATURE ENGINEERING PREVIEW
# ---------------------------------------------------------
def stage_3_feature_engineering_preview(df: pd.DataFrame):
    print("\n" + "=" * 75)
    print("STAGE 3 PREVIEW: DOMAIN-SPECIFIC ELECTROCHEMICAL FEATURE CREATION")
    print("=" * 75)
    
    df_feat = df.copy()
    
    # 1. Thermal Gradient
    df_feat['temp_delta'] = df_feat['cell_temperature_max'] - df_feat['cell_temperature_avg']
    
    # 2. Electrical Stress Ratio (C-Rate Proxy)
    df_feat['c_rate_proxy'] = df_feat['average_charge_power_kw'] / (df_feat['battery_capacity_kwh'] + 1e-5)
    
    # 3. Cell Degradation Index (Relative Voltage Spread)
    df_feat['cell_voltage_spread'] = df_feat['cell_voltage_std'] / (df_feat['cell_voltage_avg'] + 1e-5)
    
    # 4. Aggressive Usage Stress Metric
    df_feat['stress_index'] = df_feat['aggressive_acceleration_score'] * df_feat['hard_braking_score']
    
    print("\n[3.1] Engineered Domain Feature Statistics:")
    eng_cols = ['temp_delta', 'c_rate_proxy', 'cell_voltage_spread', 'stress_index']
    print(df_feat[eng_cols].describe().round(4).T[['mean', 'std', 'min', '50%', 'max']])
    
    # Correlation of engineered features with targets
    eng_corr = df_feat[eng_cols + ['predicted_remaining_life_cycles', 'battery_failure']].corr()
    print("\n[3.2] Correlation of Engineered Features with Target Tasks:")
    corr_task1 = eng_corr['predicted_remaining_life_cycles'][eng_cols]
    corr_task2 = eng_corr['battery_failure'][eng_cols]
    corr_summary = pd.DataFrame({'Task 1 (RUL Corr)': corr_task1, 'Task 2 (Failure Corr)': corr_task2})
    print(corr_summary.to_string())
    
    print("\n" + "=" * 75)
    print("AUDIT & STAGES 1-2 SCRIPT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    csv_file = "ev battery_failure  Dataset.csv"
    if not os.path.exists(csv_file):
        print(f"Error: Dataset file not found at '{csv_file}'")
    else:
        print(f"Loading '{csv_file}'...")
        data = pd.read_csv(csv_file)
        stage_1_problem_definition(data)
        stage_2_eda_and_cleaning(data)
        stage_3_feature_engineering_preview(data)
