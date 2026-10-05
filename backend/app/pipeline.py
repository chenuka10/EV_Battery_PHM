import datetime
import json
import os
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException

# Locate models directory relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class ArtifactContainer:
    """Cached container holding loaded ML artifacts."""
    def __init__(self):
        self.ct = None
        self.model_rul = None
        self.model_lr = None
        self.model_xgb = None
        self.metadata = None
        self.scenarios = None
        self.feature_ranges = None
        self.test_scores_df = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        self.ct = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
        self.model_rul = joblib.load(os.path.join(MODELS_DIR, "champion_rul_hgb.joblib"))
        self.model_lr = joblib.load(os.path.join(MODELS_DIR, "champion_failure_lr.joblib"))
        self.model_xgb = joblib.load(os.path.join(MODELS_DIR, "champion_failure_xgb.joblib"))

        with open(os.path.join(MODELS_DIR, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

        with open(os.path.join(MODELS_DIR, "preset_scenarios.json"), "r") as f:
            self.scenarios = json.load(f)

        ranges_path = os.path.join(MODELS_DIR, "feature_ranges.json")
        if os.path.exists(ranges_path):
            with open(ranges_path, "r") as f:
                self.feature_ranges = json.load(f)
        else:
            self.feature_ranges = {}

        scores_path = os.path.join(MODELS_DIR, "test_scores.csv")
        if os.path.exists(scores_path):
            self.test_scores_df = pd.read_csv(scores_path)
        else:
            self.test_scores_df = None

        self.loaded = True


artifacts = ArtifactContainer()


def featurize(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Derive the 7 verified domain engineering features with epsilon protection."""
    df = df_raw.copy()
    df['temperature_spread'] = df['cell_temperature_max'] - df['cell_temperature_avg']
    df['efficiency_gap'] = (df['charge_efficiency'] - df['discharge_efficiency']).abs()
    df['health_loss_interaction'] = (df['battery_health_percent'] * df['capacity_loss_percent']) / 100.0
    df['resistance_per_1000_cycles'] = (df['internal_resistance'] / (df['cycle_count'] + 1.0)) * 1000.0
    df['c_rate_proxy'] = df['average_charge_power_kw'] / (df['battery_capacity_kwh'] + 1e-5)
    df['cell_voltage_spread'] = df['cell_voltage_std'] / (df['cell_voltage_avg'] + 1e-5)
    df['stress_index'] = df['aggressive_acceleration_score'] * df['hard_braking_score']
    return df


def validate_and_impute_inputs(inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Validate types, physical boundary constraints, and impute missing training features."""
    artifacts.load()
    meta_features = set(artifacts.metadata['features'])
    excluded_allowed = {
        'vehicle_id', 'battery_serial', 'scenario_label', 'scenario_name',
        'id', 'row_id', 'predicted_remaining_life_cycles', 'battery_failure'
    }

    # 1. Unknown field validation
    unknown_fields = [k for k in inputs.keys() if k not in meta_features and k not in excluded_allowed]
    if unknown_fields:
        raise HTTPException(
            status_code=422,
            detail=f"Validation Error: Unknown telemetry attributes received: {unknown_fields}. Supported model features: {len(meta_features)} attributes."
        )

    clean_inputs = {}
    warnings = []

    # 2. Type casting and sanitization
    for k, v in inputs.items():
        if k in meta_features:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if k in artifacts.metadata.get('num_defaults', {}):
                try:
                    clean_inputs[k] = float(v)
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Validation Error: Field '{k}' must be numerical, received '{v}'."
                    )
            else:
                clean_inputs[k] = str(v)

    # 3. Physical Consistency Rules
    t_max = clean_inputs.get('cell_temperature_max')
    t_avg = clean_inputs.get('cell_temperature_avg')
    if t_max is not None and t_avg is not None:
        if t_max < t_avg:
            raise HTTPException(
                status_code=422,
                detail=f"Physical Anomaly: Maximum Cell Hotspot Temperature ({t_max:.1f}°C) cannot be lower than Average Pack Temperature ({t_avg:.1f}°C)."
            )

    # Collinearity & Outlier Warnings
    h = clean_inputs.get('battery_health_percent')
    l = clean_inputs.get('capacity_loss_percent')
    if h is not None and l is not None:
        s = h + l
        if abs(s - 100.0) > 6.0:
            warnings.append(
                f"Collinearity Notice: State of Health ({h:.1f}%) and Capacity Loss ({l:.1f}%) sum to {s:.1f}% (Nominal conservation sum is ~100%)."
            )

    if t_max is not None and t_max > 72.98:
        warnings.append(
            f"Thermal Outlier Notice (Report Table 4): Max temperature ({t_max:.1f}°C) exceeds 72.98°C, corresponding to a 56.67% historical failure rate in the training dataset."
        )

    res = clean_inputs.get('internal_resistance')
    if res is not None and res > 0.85:
        warnings.append(
            f"Impedance Warning: Internal resistance ({res:.3f} Ω) lies in the top 1% extreme wear tail of the dataset."
        )

    # 4. Imputation of Missing Fields
    imputed_fields = []
    full_data = {}
    for col in artifacts.metadata['features']:
        if col in clean_inputs:
            full_data[col] = clean_inputs[col]
        else:
            imputed_fields.append(col)
            if col in artifacts.metadata['num_defaults']:
                full_data[col] = artifacts.metadata['num_defaults'][col]
            elif col in artifacts.metadata['cat_options']:
                full_data[col] = artifacts.metadata['cat_options'][col][0]
            else:
                full_data[col] = 0.0

    return full_data, warnings, imputed_fields


def predict_battery(engine: str, tau: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute end-to-end feature engineering, transformation, and dual-task inference."""
    artifacts.load()
    full_data, warnings, imputed_fields = validate_and_impute_inputs(inputs)

    # Featurize
    raw_df = pd.DataFrame([full_data])
    feat_df = featurize(raw_df)

    # Transform through ColumnTransformer
    X_trans = artifacts.ct.transform(feat_df[artifacts.metadata['features']])

    # Task 1 RUL
    rul_cycles = float(artifacts.model_rul.predict(X_trans)[0])

    # Task 2 Failure Risk
    if engine.lower() == "xgboost":
        p_failure = float(artifacts.model_xgb.predict_proba(X_trans)[0, 1])
        model_name = "Tuned XGBoost (PR-AUC Champion)"
    else:
        p_failure = float(artifacts.model_lr.predict_proba(X_trans)[0, 1])
        model_name = "Safety-Calibrated Logistic Regression"

    flagged = bool(p_failure >= tau)

    # Operational status determination
    if not flagged:
        status = "nominal"
        status_label = "NOMINAL"
        alert_title = "BMS Telemetry Status: Normal Operation"
        alert_action = "No elevated-risk flag raised. Battery pack operating within nominal electro-thermal boundaries."
        alert_note = f"Estimated failure probability ({p_failure:.2%}) is below active cutoff (τ = {tau:.3f}). Regular fleet telemetry logging maintained."
    elif p_failure < 0.50:
        status = "early_warning"
        status_label = "EARLY WARNING"
        alert_title = f"Early Warning: Intercepted by Safety Cutoff (τ = {tau:.3f})"
        alert_action = "Recommended action: Schedule preventative dealer diagnostic; inspect thermal harness and cell balance."
        alert_note = f"Standard cutoff (0.50) clears this vehicle as normal (P = {p_failure:.2%}). Safety cutoff (τ = {tau:.3f}) flags it early for depot maintenance."
    else:
        status = "critical"
        status_label = "CRITICAL"
        alert_title = f"Critical Alert: Failure Probability Exceeds Cutoff (P = {p_failure:.2%})"
        alert_action = "Recommended action: Urgent workshop service required; prioritize high-voltage electrical and cooling inspection."
        alert_note = f"Failure probability ({p_failure:.2%}) exceeds critical operational boundaries, indicating severe internal degradation or thermal runaway potential."

    engineered_features = {
        'temperature_spread': float(feat_df['temperature_spread'].values[0]),
        'efficiency_gap': float(feat_df['efficiency_gap'].values[0]),
        'health_loss_interaction': float(feat_df['health_loss_interaction'].values[0]),
        'resistance_per_1000_cycles': float(feat_df['resistance_per_1000_cycles'].values[0]),
        'c_rate_proxy': float(feat_df['c_rate_proxy'].values[0]),
        'cell_voltage_spread': float(feat_df['cell_voltage_spread'].values[0]),
        'stress_index': float(feat_df['stress_index'].values[0]),
    }

    return {
        "rul_cycles": round(rul_cycles, 1),
        "p_failure": round(p_failure, 6),
        "flagged": flagged,
        "status": status,
        "status_label": status_label,
        "alert_title": alert_title,
        "alert_action": alert_action,
        "alert_note": alert_note,
        "engineered_features": engineered_features,
        "imputed_fields": imputed_fields,
        "warnings": warnings,
        "model_names": {
            "task1": "Tuned HistGradientBoostingRegressor (R² = 0.8972)",
            "task2": model_name
        },
        "version": "1.0.0-production",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
