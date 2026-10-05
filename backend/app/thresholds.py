from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from backend.app.pipeline import artifacts


def get_threshold_metrics(engine: str, tau: float) -> Dict[str, Any]:
    """Compute live confusion matrix and metrics from held-out evaluation test set."""
    artifacts.load()
    if artifacts.test_scores_df is None:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "status": "data_file_not_found"
        }

    col = "p_lr" if engine.lower() == "logistic" else "p_xgb"
    y_true = artifacts.test_scores_df['y_true'].values
    p_scores = artifacts.test_scores_df[col].values

    preds = (p_scores >= tau).astype(int)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))

    recall = tp / (tp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return {
        "recall": round(float(recall), 4),
        "precision": round(float(precision), 4),
        "f1": round(float(f1), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


def get_curves(engine: str) -> Dict[str, Any]:
    """Sweep decision thresholds 0.05 to 0.95 to generate evaluation curves and markers."""
    artifacts.load()
    if artifacts.test_scores_df is None:
        return {
            "curve": {"taus": [], "recalls": [], "precisions": [], "f1s": []},
            "markers": {"safety_tau": 0.193, "f1_max_tau": 0.278}
        }

    col = "p_lr" if engine.lower() == "logistic" else "p_xgb"
    y_true = artifacts.test_scores_df['y_true'].values
    p_scores = artifacts.test_scores_df[col].values

    taus = np.linspace(0.05, 0.95, 91).round(3).tolist()
    recalls = []
    precisions = []
    f1s = []

    for t in taus:
        preds = (p_scores >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))

        rec = tp / (tp + fn + 1e-10)
        prec = tp / (tp + fp + 1e-10)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-10)

        recalls.append(round(float(rec), 4))
        precisions.append(round(float(prec), 4))
        f1s.append(round(float(f1), 4))

    # Identify F1-max
    best_f1_idx = int(np.argmax(f1s))
    best_f1_tau = taus[best_f1_idx]

    # Identify Safety cutoff (highest tau with recall >= 0.90)
    safety_cands = [taus[i] for i, r in enumerate(recalls) if r >= 0.90]
    safety_tau = max(safety_cands) if safety_cands else 0.193

    return {
        "curve": {
            "taus": taus,
            "recalls": recalls,
            "precisions": precisions,
            "f1s": f1s
        },
        "markers": {
            "safety_tau": safety_tau,
            "f1_max_tau": best_f1_tau,
            "default_tau": 0.500
        }
    }


def get_operating_modes(engine: str) -> Dict[str, Any]:
    """Derive operational decision modes dynamically from test set curves."""
    curves = get_curves(engine)
    markers = curves['markers']
    taus = curves['curve']['taus']
    recalls = curves['curve']['recalls']
    precisions = curves['curve']['precisions']
    f1s = curves['curve']['f1s']

    # 1. Safety Mode: target Recall >= 90%
    safety_tau = markers['safety_tau']
    safety_m = get_threshold_metrics(engine, safety_tau)

    # 2. Balanced Mode: F1-max
    f1_tau = markers['f1_max_tau']
    f1_m = get_threshold_metrics(engine, f1_tau)

    # 3. Precision Mode: highest recall with precision >= 0.80 (or highest attainable)
    prec_candidates = [(taus[i], recalls[i], precisions[i]) for i in range(len(taus)) if precisions[i] >= 0.80]
    if prec_candidates:
        prec_candidates.sort(key=lambda x: x[1], reverse=True)
        prec_tau = prec_candidates[0][0]
    else:
        prec_tau = 0.500
    prec_m = get_threshold_metrics(engine, prec_tau)

    return {
        "engine": engine,
        "modes": {
            "safety": {
                "name": "Safety-First Early Warning",
                "tau": safety_tau,
                "recall": safety_m['recall'],
                "precision": safety_m['precision'],
                "f1": safety_m['f1'],
                "missed_failures": safety_m['fn'],
                "false_alarms": safety_m['fp'],
                "description": "Maximizes battery failure interception (Recall ≥ 90%) to prevent on-road thermal stranding."
            },
            "balanced": {
                "name": "Balanced Operational Trade-off (F1-Optimal)",
                "tau": f1_tau,
                "recall": f1_m['recall'],
                "precision": f1_m['precision'],
                "f1": f1_m['f1'],
                "missed_failures": f1_m['fn'],
                "false_alarms": f1_m['fp'],
                "description": "Maximizes harmonic mean of Precision and Recall for balanced fleet maintenance workload."
            },
            "precision": {
                "name": "High-Precision Filter",
                "tau": prec_tau,
                "recall": prec_m['recall'],
                "precision": prec_m['precision'],
                "f1": prec_m['f1'],
                "missed_failures": prec_m['fn'],
                "false_alarms": prec_m['fp'],
                "description": "Throttles false alarms when depot workshop diagnostic bay capacity is severely limited."
            }
        }
    }


def simulate_fleet(
    n_vehicles: int,
    capacity: int,
    engine: str,
    tau: float = None,
    mode: str = None
) -> Dict[str, Any]:
    """Scale test-set failure and alarm rates across a user-defined fleet size."""
    modes_data = get_operating_modes(engine)

    if mode and mode.lower() in modes_data['modes']:
        tau_used = modes_data['modes'][mode.lower()]['tau']
        mode_used = modes_data['modes'][mode.lower()]['name']
    elif tau is not None:
        tau_used = tau
        mode_used = f"Custom (τ = {tau:.3f})"
    else:
        tau_used = 0.193
        mode_used = "Safety-First (Default)"

    m = get_threshold_metrics(engine, tau_used)
    test_total = 4000
    test_failures = 277
    test_normal = 3723

    # Scaled estimates
    failure_prevalence = test_failures / test_total  # 6.925%
    exp_failures = n_vehicles * failure_prevalence
    exp_normal = n_vehicles * (1.0 - failure_prevalence)

    exp_intercepted = exp_failures * m['recall']
    exp_missed = exp_failures * (1.0 - m['recall'])
    false_alarm_rate = m['fp'] / test_normal
    exp_false_alarms = exp_normal * false_alarm_rate
    exp_total_flags = exp_intercepted + exp_false_alarms

    ratio = exp_total_flags / (capacity + 1e-5)
    if exp_total_flags > capacity:
        status = f"Bottleneck: Expected monthly flags ({exp_total_flags:.0f}) exceed depot capacity ({capacity})."
    else:
        status = f"Adequate: Expected monthly flags ({exp_total_flags:.0f}) fit within depot capacity ({capacity})."

    return {
        "fleet_size": n_vehicles,
        "capacity": capacity,
        "tau_used": round(tau_used, 4),
        "mode_used": mode_used,
        "expected_failures": round(exp_failures, 1),
        "expected_normal": round(exp_normal, 1),
        "expected_flags": round(exp_total_flags, 1),
        "expected_intercepted": round(exp_intercepted, 1),
        "expected_missed": round(exp_missed, 1),
        "workload_ratio": round(ratio, 2),
        "capacity_status": status,
        "scaling_assumption": "Scaled proportionally from the 4,000-vehicle held-out stratified evaluation split (6.925% historical failure prevalence)."
    }
