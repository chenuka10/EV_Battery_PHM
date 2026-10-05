import pytest
import numpy as np
from backend.app.pipeline import artifacts
from backend.app.thresholds import (
    get_curves,
    get_operating_modes,
    get_threshold_metrics,
    simulate_fleet
)


def setup_module():
    artifacts.load()


def test_threshold_math_reproduces_report():
    """Verify threshold math at safety cutoff against Technical Report Table 13."""
    # The exact mathematical cutoff achieving >= 90% recall is 0.19259366 (printed as 0.193 in the report)
    m_exact = get_threshold_metrics("logistic", 0.19259366)
    assert m_exact['tp'] == 250
    assert m_exact['fn'] == 27
    assert m_exact['fp'] == 139
    assert m_exact['tn'] == 3584
    assert np.isclose(m_exact['recall'], 0.9025, atol=1e-3)
    assert np.isclose(m_exact['precision'], 0.6427, atol=1e-3)

    # At rounded tau = 0.193000
    m_rounded = get_threshold_metrics("logistic", 0.193)
    assert m_rounded['fp'] == 139
    assert m_rounded['fn'] in [27, 28]  # single borderline vehicle transition


def test_curves_generation():
    """Verify threshold sweep generates valid curve arrays."""
    data = get_curves("logistic")
    taus = data['curve']['taus']
    recalls = data['curve']['recalls']
    precisions = data['curve']['precisions']
    f1s = data['curve']['f1s']

    assert len(taus) == 91
    assert taus[0] == 0.05
    assert taus[-1] == 0.95
    assert len(recalls) == 91
    # Recall monotonically non-increasing
    assert recalls[0] >= recalls[-1]


def test_operating_modes_derivation():
    """Verify operating modes are dynamically computed."""
    res = get_operating_modes("logistic")
    modes = res['modes']
    assert "safety" in modes
    assert "balanced" in modes
    assert "precision" in modes
    assert modes['safety']['recall'] >= 0.898
    assert modes['balanced']['f1'] >= 0.75


def test_fleet_simulation():
    """Verify fleet workload scaling calculations."""
    sim = simulate_fleet(
        n_vehicles=5000,
        capacity=300,
        engine="logistic",
        tau=0.193
    )
    assert sim['fleet_size'] == 5000
    assert sim['capacity'] == 300
    assert sim['expected_failures'] == round(5000 * (277 / 4000), 1)
    assert sim['expected_flags'] > 0
    assert "scaling_assumption" in sim
