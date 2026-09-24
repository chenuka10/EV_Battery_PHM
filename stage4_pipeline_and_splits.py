"""
SLIIT IT3051 - Data Mining & Predictive Analytics
EV Battery Prognostics & Health Management (PHM) System
Group: Necrons
Week 1: Stage 4 - Data Leakage Guard, Target Isolation, Train/Test Splits & ColumnTransformer Pipeline
Author: Person D (Integration & Pipeline Specialist)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 3 Feature Engineering:
    Derives 4 domain-specific electrochemical and operational stress metrics.
    """
    df_out = df.copy()
    
    # 1. Thermal Gradient (Max vs Avg Cell Temperature)
    df_out['temp_delta'] = df_out['cell_temperature_max'] - df_out['cell_temperature_avg']
    
    # 2. Electrical Stress Ratio (C-Rate Proxy: Power / Capacity)
    df_out['c_rate_proxy'] = df_out['average_charge_power_kw'] / (df_out['battery_capacity_kwh'] + 1e-5)
    
    # 3. Cell Degradation Index (Relative Voltage Spread)
    df_out['cell_voltage_spread'] = df_out['cell_voltage_std'] / (df_out['cell_voltage_avg'] + 1e-5)
    
    # 4. Aggressive Driving Usage Index
    df_out['stress_index'] = df_out['aggressive_acceleration_score'] * df_out['hard_braking_score']
    
    return df_out

def build_preprocessing_pipeline(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Constructs a production-grade ColumnTransformer pipeline:
    - Numeric: Median Imputation + Standard Scaling
    - Categorical: Mode Imputation + One-Hot Encoding (ignoring unseen categories)
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_cols),
            ('cat', cat_pipeline, categorical_cols)
        ],
        verbose_feature_names_out=False
    )
    
    return preprocessor

def run_stage_4_pipeline(csv_path: str = "ev battery_failure  Dataset.csv"):
    print("=" * 80)
    print("STAGE 4: TARGET ISOLATION, TRAIN/TEST SPLIT & COLUMNTRANSFORMER PIPELINE")
    print("=" * 80)
    
    # 1. Load Data
    print(f"\n[4.1] Loading raw dataset: '{csv_path}'...")
    df = pd.read_csv(csv_path)
    print(f"  - Initial Raw Dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # 2. Add Engineered Features (Stage 3 Integration)
    df_engineered = add_engineered_features(df)
    print(f"  - Post-Feature Engineering Dimensions: {df_engineered.shape[0]:,} rows x {df_engineered.shape[1]} columns")
    
    # 3. Boundary Conditions: Exclude Identifiers
    id_cols = ['vehicle_id', 'battery_serial']
    target_rul = 'predicted_remaining_life_cycles'
    target_fail = 'battery_failure'
    
    # List of all feature column candidates
    excluded_for_features = id_cols + [target_rul, target_fail]
    feature_cols = [c for c in df_engineered.columns if c not in excluded_for_features]
    
    cat_cols = df_engineered[feature_cols].select_dtypes(include=['object', 'string']).columns.tolist()
    num_cols = df_engineered[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    print("\n[4.2] Feature Space Segregation:")
    print(f"  - Excluded Primary IDs: {id_cols} (Zero physical generalization)")
    print(f"  - Total Predictor Features: {len(feature_cols)} (Numeric: {len(num_cols)}, Categorical: {len(cat_cols)})")
    print(f"  - Categorical Features ({len(cat_cols)}): {cat_cols}")
    
    # 4. Strict Target Isolation
    print("\n[4.3] Enforcing Target Isolation:")
    print("  * Task 1 Feature Set: excludes both 'battery_failure' and 'predicted_remaining_life_cycles'")
    print("  * Task 2 Feature Set: excludes both 'predicted_remaining_life_cycles' and 'battery_failure'")
    print("  * GUARANTEE: Zero circular leakage between simultaneous prediction tasks.")
    
    # -------------------------------------------------------------
    # TASK 1: REMAINING LIFE CYCLES REGRESSION (RUL)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TASK 1 DATA PARTITIONING: RUL REGRESSION")
    print("-" * 60)
    # Filter out rows with missing RUL labels
    mask_task1_valid = df_engineered[target_rul].notnull()
    df_task1 = df_engineered[mask_task1_valid].copy()
    
    X1 = df_task1[feature_cols]
    y1 = df_task1[target_rul]
    
    print(f"  - Filtered Unlabeled Target Records: dropped {len(df_engineered) - len(df_task1):,} rows missing RUL labels")
    print(f"  - Valid Task 1 Sample Size: {len(df_task1):,} instances")
    
    # 80/20 Train/Test Split
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.20, random_state=42
    )
    print(f"  - Train Set: {X1_train.shape[0]:,} samples | Test Set: {X1_test.shape[0]:,} samples")
    
    # Fit Pipeline on Task 1 Train ONLY
    preprocessor_task1 = build_preprocessing_pipeline(num_cols, cat_cols)
    X1_train_transformed = preprocessor_task1.fit_transform(X1_train)
    X1_test_transformed = preprocessor_task1.transform(X1_test)
    
    feature_names_out_1 = preprocessor_task1.get_feature_names_out()
    print(f"  - Transformed Feature Matrix Shape (Train): {X1_train_transformed.shape}")
    print(f"  - Transformed Feature Matrix Shape (Test):  {X1_test_transformed.shape}")
    print(f"  - Total Transformed Dimensions (post-one-hot): {len(feature_names_out_1)} features")
    print(f"  - Leakage Guard Verified: Mean of scaled training data ~ {np.mean(X1_train_transformed[:, :len(num_cols)]):.4f}")
    
    # -------------------------------------------------------------
    # TASK 2: CRITICAL FAILURE RISK CLASSIFICATION
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TASK 2 DATA PARTITIONING: CRITICAL FAILURE CLASSIFICATION")
    print("-" * 60)
    X2 = df_engineered[feature_cols]
    y2 = df_engineered[target_fail]
    
    print(f"  - Total Sample Size: {len(df_engineered):,} instances")
    print(f"  - Overall Class Distribution: 0 (Normal) = {(y2 == 0).sum():,} ({(y2 == 0).mean()*100:.2f}%), "
          f"1 (Failure) = {(y2 == 1).sum():,} ({(y2 == 1).mean()*100:.2f}%)")
    
    # Stratified 80/20 Train/Test Split (Preserving 93.08% / 6.92% ratio)
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.20, random_state=42, stratify=y2
    )
    print(f"  - Stratified Train Set: {X2_train.shape[0]:,} samples (Failures: {(y2_train == 1).sum():,} [{(y2_train == 1).mean()*100:.2f}%])")
    print(f"  - Stratified Test Set:  {X2_test.shape[0]:,} samples (Failures: {(y2_test == 1).sum():,} [{(y2_test == 1).mean()*100:.2f}%])")
    
    # Fit Pipeline on Task 2 Train ONLY
    preprocessor_task2 = build_preprocessing_pipeline(num_cols, cat_cols)
    X2_train_transformed = preprocessor_task2.fit_transform(X2_train)
    X2_test_transformed = preprocessor_task2.transform(X2_test)
    
    print(f"  - Transformed Feature Matrix Shape (Train): {X2_train_transformed.shape}")
    print(f"  - Transformed Feature Matrix Shape (Test):  {X2_test_transformed.shape}")
    print(f"  - Leakage Guard Verified: Strict fit_transform on train, transform on test.")

    print("\n" + "=" * 80)
    print("STAGE 4 PIPELINE EXECUTION COMPLETED WITH ZERO DATA LEAKAGE!")
    print("=" * 80)
    
    return {
        'preprocessor_task1': preprocessor_task1,
        'preprocessor_task2': preprocessor_task2,
        'X1_train': X1_train_transformed, 'X1_test': X1_test_transformed,
        'y1_train': y1_train, 'y1_test': y1_test,
        'X2_train': X2_train_transformed, 'X2_test': X2_test_transformed,
        'y2_train': y2_train, 'y2_test': y2_test,
        'feature_names': feature_names_out_1
    }

if __name__ == "__main__":
    run_stage_4_pipeline()
