import pytest
import numpy as np
import pandas as pd
from fastapi import HTTPException
from backend.app.pipeline import artifacts, featurize, predict_battery, validate_and_impute_inputs


def setup_module():
    artifacts.load()


def test_artifacts_loaded():
    """Verify all 8 artifacts are populated."""
    assert artifacts.ct is not None
    assert artifacts.model_rul is not None
    assert artifacts.model_lr is not None
    assert artifacts.model_xgb is not None
    assert artifacts.metadata is not None
    assert artifacts.scenarios is not None
    assert artifacts.feature_ranges is not None
    assert artifacts.test_scores_df is not None


def test_pipeline_parity_with_app():
    """Verify that backend pipeline produces identical predictions to app.py logic."""
    for sc_name, sc_obj in artifacts.scenarios.items():
        data = sc_obj['data'].copy()
        
        # Backend prediction
        res_backend = predict_battery("logistic", 0.193, data)
        
        # Direct computation matching app.py pipeline
        clean_data, _, _ = validate_and_impute_inputs(data)
        df_raw = pd.DataFrame([clean_data])
        df_feat = featurize(df_raw)
        Xt = artifacts.ct.transform(df_feat[artifacts.metadata['features']])
        direct_rul = float(artifacts.model_rul.predict(Xt)[0])
        direct_p_lr = float(artifacts.model_lr.predict_proba(Xt)[0, 1])
        
        assert np.isclose(res_backend['rul_cycles'], direct_rul, atol=0.1)
        assert np.isclose(res_backend['p_failure'], direct_p_lr, atol=1e-5)


def test_physical_validation_max_temp_lower_than_avg():
    """Ensure max_temp < avg_temp raises 422 HTTP exception."""
    bad_inputs = {
        "cell_temperature_max": 20.0,
        "cell_temperature_avg": 35.0
    }
    with pytest.raises(HTTPException) as exc_info:
        validate_and_impute_inputs(bad_inputs)
    assert exc_info.value.status_code == 422
    assert "Physical Anomaly" in exc_info.value.detail


def test_unknown_fields_rejected():
    """Ensure unexpected fields are rejected with clean 422."""
    bad_inputs = {
        "fake_telemetry_sensor_x": 999.0
    }
    with pytest.raises(HTTPException) as exc_info:
        validate_and_impute_inputs(bad_inputs)
    assert exc_info.value.status_code == 422
    assert "Unknown telemetry attributes" in exc_info.value.detail


def test_imputation_and_warning_reporting():
    """Ensure missing fields are imputed with defaults and warnings recorded."""
    partial_inputs = {
        "cell_temperature_max": 75.0,  # triggers outlier warning >72.98
        "cell_temperature_avg": 25.0,
        "battery_health_percent": 90.0,
        "capacity_loss_percent": 30.0   # triggers collinearity warning (sum=120)
    }
    full_data, warnings, imputed = validate_and_impute_inputs(partial_inputs)
    assert len(imputed) > 50
    assert any("72.98" in w for w in warnings)
    assert any("Collinearity Notice" in w for w in warnings)
    assert "cycle_count" in full_data
    assert full_data["cycle_count"] == artifacts.metadata['num_defaults']['cycle_count']
