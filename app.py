import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# Page Configuration & Aesthetics
# ==============================================================================
st.set_page_config(
    page_title="EV Battery PHM System | Dual-Task Diagnostics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern automotive styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.3rem;
    }
    .badge-safe {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-intercept {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Model & Metadata Loading (Cached)
# ==============================================================================
@st.cache_resource
def load_artifacts():
    ct = joblib.load('models/preprocessor.joblib')
    hgb = joblib.load('models/champion_rul_hgb.joblib')
    lr = joblib.load('models/champion_failure_lr.joblib')
    xgb = joblib.load('models/champion_failure_xgb.joblib')
    with open('models/metadata.json', 'r') as f:
        meta = json.load(f)
    with open('models/preset_scenarios.json', 'r') as f:
        scenarios = json.load(f)
    return ct, hgb, lr, xgb, meta, scenarios

try:
    ct, model_rul, model_lr, model_xgb, meta, scenarios = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# ==============================================================================
# Sidebar Controls
# ==============================================================================
st.sidebar.image("https://img.icons8.com/color/96/electric-vehicle.png", width=70)
st.sidebar.title("BMS Telemetry Control")
st.sidebar.markdown("**Group:** Necrons | **Dual-Task System**")
st.sidebar.markdown("---")

# 1. Preset Scenarios vs Custom
mode = st.sidebar.selectbox(
    "Vehicle Input Mode",
    ["Select a Preset Telemetry Scenario", "Manual Telemetry Tuning (Custom Sliders)"]
)

if mode == "Select a Preset Telemetry Scenario":
    selected_scenario_name = st.sidebar.selectbox(
        "Choose Real Telemetry Scenario:",
        list(scenarios.keys())
    )
    current_data = scenarios[selected_scenario_name]['data'].copy()
    st.sidebar.info(scenarios[selected_scenario_name]['description'])
else:
    current_data = {}
    st.sidebar.markdown("### Primary Diagnostic Sliders")
    # Provide intuitive sliders for the top physical drivers
    current_data['battery_health_percent'] = st.sidebar.slider("Battery Health (%)", 50.0, 100.0, 84.5, 0.5)
    current_data['capacity_loss_percent'] = st.sidebar.slider("Capacity Loss (%)", 0.0, 50.0, 100.0 - current_data['battery_health_percent'], 0.5)
    current_data['cell_temperature_max'] = st.sidebar.slider("Max Cell Temperature (°C)", 15.0, 65.0, 36.2, 0.5)
    current_data['cell_temperature_avg'] = st.sidebar.slider("Avg Cell Temperature (°C)", 15.0, 55.0, 31.8, 0.5)
    current_data['internal_resistance'] = st.sidebar.slider("Internal Resistance (mΩ)", 0.05, 0.80, 0.22, 0.01)
    current_data['cycle_count'] = st.sidebar.slider("Completed Cycle Count", 50, 4500, 1200, 50)
    current_data['thermal_runaway_risk'] = st.sidebar.slider("Thermal Runaway Risk Index", 0.0, 1.0, 0.18, 0.01)
    current_data['thermal_health_score'] = st.sidebar.slider("Thermal Health Score", 0.0, 100.0, 78.0, 1.0)
    current_data['average_charge_power_kw'] = st.sidebar.slider("Average Charge Power (kW)", 10.0, 150.0, 45.0, 1.0)
    current_data['battery_capacity_kwh'] = st.sidebar.slider("Pack Capacity (kWh)", 30.0, 120.0, 75.0, 1.0)
    
    # Fill remaining columns with verified medians
    for col in meta['features']:
        if col not in current_data:
            if col in meta['num_defaults']:
                current_data[col] = meta['num_defaults'][col]
            elif col in meta['cat_options']:
                current_data[col] = meta['cat_options'][col][0]

st.sidebar.markdown("---")

# 2. Decision Threshold Control (Your Signature Star Feature!)
st.sidebar.markdown("### 🎯 Decision Threshold ($\tau$) Calibration")
tau_slider = st.sidebar.slider(
    "Critical Failure Cutoff (tau)",
    min_value=0.05,
    max_value=0.95,
    value=0.193,  # The Calibrated Safety Threshold!
    step=0.01,
    help="Default ML threshold is 0.50. Automotive Safety threshold is calibrated at 0.193 (guaranteeing >= 90% failure recall)."
)

# Benchmark buttons
col_b1, col_b2 = st.sidebar.columns(2)
if col_b1.button("Default (0.50)"):
    tau_slider = 0.50
if col_b2.button("Safety (0.193)"):
    tau_slider = 0.193

# 3. Model Engine Toggle
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Diagnostic Engine")
engine_choice = st.sidebar.radio(
    "Inference Deployment Tier:",
    ["Edge BMS (Calibrated Logistic Regression)", "Cloud Fleet Analytics (Tuned XGBoost)"]
)

# ==============================================================================
# Main Dashboard Layout
# ==============================================================================
st.markdown('<div class="main-header">⚡ EV Battery Prognostics & Health Management (PHM)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dual-Task Edge Prognostics: Continuous RUL Cycles Regression & Calibrated Failure Risk Early Warning</div>', unsafe_allow_html=True)

# Build DataFrame for inference
input_df = pd.DataFrame([current_data])

# Recalculate 7 Domain Features to ensure consistency
input_df['temperature_spread'] = input_df['cell_temperature_max'] - input_df['cell_temperature_avg']
input_df['efficiency_gap'] = (input_df['charge_efficiency'] - input_df['discharge_efficiency']).abs()
input_df['health_loss_interaction'] = (input_df['battery_health_percent'] * input_df['capacity_loss_percent']) / 100.0
input_df['resistance_per_1000_cycles'] = (input_df['internal_resistance'] / (input_df['cycle_count'] + 1.0)) * 1000.0
input_df['c_rate_proxy'] = input_df['average_charge_power_kw'] / (input_df['battery_capacity_kwh'] + 1e-5)
input_df['cell_voltage_spread'] = input_df['cell_voltage_std'] / (input_df['cell_voltage_avg'] + 1e-5)
input_df['stress_index'] = input_df['aggressive_acceleration_score'] * input_df['hard_braking_score']

# Transform through Phase 1 ColumnTransformer
input_transformed = ct.transform(input_df[meta['features']])

# Predictions
pred_rul = float(model_rul.predict(input_transformed)[0])

if "Logistic" in engine_choice:
    prob_failure = float(model_lr.predict_proba(input_transformed)[0, 1])
    active_model_name = "Safety-Calibrated Logistic Regression"
else:
    prob_failure = float(model_xgb.predict_proba(input_transformed)[0, 1])
    active_model_name = "Tuned XGBoost (PR-AUC Champion)"

# Determine Hazard Status based on Tau
is_flagged = prob_failure >= tau_slider

# ==============================================================================
# Top Metric Overview Cards
# ==============================================================================
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Predicted Remaining Life (RUL)</div>
        <div class="metric-value">{pred_rul:,.0f} <span style="font-size: 1rem; color:#64748B;">cycles</span></div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Failure Probability P(Failure)</div>
        <div class="metric-value">{prob_failure:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Active Threshold (tau)</div>
        <div class="metric-value">{tau_slider:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    if not is_flagged:
        badge_html = '<span class="badge-safe">🟢 NOMINAL / SAFE</span>'
    elif prob_failure < 0.50 and is_flagged:
        badge_html = '<span class="badge-intercept">🟠 EARLY WARNING INTERCEPTED</span>'
    else:
        badge_html = '<span class="badge-critical">🔴 CRITICAL HAZARD</span>'
        
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Operational Safety Status</div>
        <div style="margin-top: 0.6rem;">{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# Operational Diagnostic Alert Banner
# ==============================================================================
if not is_flagged:
    st.success(f"✅ **BMS Telemetry Status: Normal Operation.** Failure probability ({prob_failure:.1%}) sits below the safety cutoff ($\tau = {tau_slider:.3f}$). Battery pack operating within thermal and chemical boundaries.")
elif prob_failure < 0.50 and is_flagged:
    st.warning(f"""
    ⚠️ **SAFETY HAZARD INTERCEPTED BY SAFETY CUTOFF ($\tau = {tau_slider:.3f}$)!**  
    * **Calculated Risk:** **{prob_failure:.1%}**  
    * **The Breakthrough:** Standard default machine learning cutoff ($0.50$) **MISSED** this battery because probability is below 50%!  
    * **Automotive Action:** Our calibrated safety threshold ($\tau = 0.193$) flags this vehicle for early dealer inspection, preventing catastrophic thermal runaway on the highway.
    """)
else:
    st.error(f"""
    🚨 **CRITICAL HAZARD DETECTED! IMMEDIATE BMS SHUTDOWN RECOMMENDED!**  
    * **Calculated Risk:** **{prob_failure:.1%}** (Exceeds both safety cutoff $\tau = {tau_slider:.3f}$ and default $0.50$).  
    * **Automotive Action:** Severe internal degradation or thermal runaway alert. Trigger active pack cooling and restrict fast-charging power.
    """)

# ==============================================================================
# Dual-Task Deep Dive Tabs
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "📈 Dual-Task Real-Time Telemetry",
    "🔬 Key Degradation Drivers & Attribution",
    "🏆 Group Master Leaderboard & Architecture"
])

with tab1:
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.subheader("Task 1: Battery Lifecycle Prognostics")
        st.markdown(f"**Champion Engine:** Tuned HistGradientBoosting ($R^2 = 0.8972$, RMSE = $528.64$ cycles)")
        
        # Lifecycle Progress Bar (based on nominal 4,500 cycle lifespan)
        cycle_fraction = min(max(pred_rul / 4500.0, 0.0), 1.0)
        st.progress(cycle_fraction, text=f"Estimated Health Fraction: {cycle_fraction:.1%}")
        
        # Operational Retirement Timeline
        est_years = pred_rul / 350.0  # assuming ~350 cycles per commuter year
        st.write(f"• **Estimated Useful Fleet Horizon:** **~{est_years:.1f} years** (assuming 350 charge cycles/year)")
        st.write(f"• **Current Health Percent:** `{input_df['battery_health_percent'].values[0]:.1f}%`")
        st.write(f"• **Internal Resistance:** `{input_df['internal_resistance'].values[0]:.3f} mΩ`")
        st.write(f"• **Resistance Rate:** `{input_df['resistance_per_1000_cycles'].values[0]:.2f} mΩ / 1,000 cycles`")

    with col_t2:
        st.subheader("Task 2: Critical Failure Risk Calibration")
        st.markdown(f"**Active Engine:** {active_model_name}")
        
        # Gauge chart using progress
        st.progress(min(max(prob_failure, 0.0), 1.0), text=f"Failure Probability: {prob_failure:.1%}")
        
        st.write(f"• **Threshold Decision Rule:** `Flag = 1 if P(Failure) >= {tau_slider:.3f}`")
        st.write(f"• **Cost Matrix Ratio ($C_{{FN}} / C_{{FP}}$):** `13.45 : 1` (Missed fire penalized 13.5x harder)")
        st.write(f"• **Automotive False Alarm Reduction:** Safety threshold achieves **90.25% Recall** with only **139 false alarms** (eliminating 175 unnecessary alarms vs uncalibrated boosting).")

with tab2:
    st.subheader("Physical Feature Attribution")
    st.markdown("Top electro-chemical and thermal variables influencing the current vehicle diagnosis:")
    
    # Feature breakdown cards
    f1, f2, f3 = st.columns(3)
    f1.metric("Thermal Spread (Max - Avg)", f"{input_df['temperature_spread'].values[0]:.2f} °C", help="Elevated spread indicates localized hot-spotting.")
    f2.metric("Health-Loss Interaction", f"{input_df['health_loss_interaction'].values[0]:.2f}", help="Composite product of health and capacity loss.")
    f3.metric("Effective C-Rate Proxy", f"{input_df['c_rate_proxy'].values[0]:.3f} C", help="Charging power divided by pack capacity.")

    st.markdown("---")
    st.markdown("### Primary Degradation Indicators in This Vehicle:")
    raw_telemetry_display = input_df[[
        'battery_health_percent', 'capacity_loss_percent', 'cell_temperature_max',
        'cell_temperature_avg', 'internal_resistance', 'cycle_count',
        'thermal_runaway_risk', 'thermal_health_score', 'charge_efficiency'
    ]].T
    raw_telemetry_display.columns = ["Observed Telemetry Value"]
    st.dataframe(raw_telemetry_display, use_container_width=True)

with tab3:
    st.subheader("Group Necrons — Master Evaluation Leaderboard")
    st.markdown("Consolidated benchmarks from **Viva 2 / Phase 2 submission** across all 4 teammates:")
    
    st.markdown("#### Task 1: Remaining Useful Life (RUL) Cycles Regression")
    t1_table = pd.DataFrame([
        {'Model': 'Dummy Regressor (Mean Floor)', 'Test R2': -0.0015, 'Test RMSE': 1649.79, 'Test MAE': 1349.30, 'Status': 'Baseline Floor'},
        {'Model': 'Ordinary Least Squares (OLS)', 'Test R2': 0.8963, 'Test RMSE': 530.95, 'Test MAE': 423.45, 'Status': 'Linear Benchmark'},
        {'Model': 'Ridge Regression (Tuned alpha=31.62)', 'Test R2': 0.8963, 'Test RMSE': 530.83, 'Test MAE': 423.56, 'Status': 'L2 Regularized'},
        {'Model': 'Lasso Regression (Tuned alpha=1.00)', 'Test R2': 0.8966, 'Test RMSE': 530.16, 'Test MAE': 423.02, 'Status': 'L1 Sparse'},
        {'Model': 'Decision Tree (max_depth=10)', 'Test R2': 0.8444, 'Test RMSE': 650.27, 'Test MAE': 512.39, 'Status': 'Tree Baseline'},
        {'Model': 'Random Forest (100 Trees)', 'Test R2': 0.8939, 'Test RMSE': 537.00, 'Test MAE': 430.62, 'Status': 'Bagging Ensemble'},
        {'Model': 'HistGradientBoosting (Tuned CV)', 'Test R2': 0.8972, 'Test RMSE': 528.64, 'Test MAE': 423.11, 'Status': '★ Task 1 Champion'}
    ])
    st.dataframe(t1_table, use_container_width=True)

    st.markdown("#### Task 2: Critical Battery Failure Classification (13.45:1 Imbalance)")
    t2_table = pd.DataFrame([
        {'Model': 'Dummy Classifier (Majority)', 'Recall': '0.00%', 'Precision': '0.00%', 'PR-AUC': 0.0693, 'ROC-AUC': 0.5000, 'Status': 'Naive Floor'},
        {'Model': 'Unweighted Decision Tree', 'Recall': '54.00%', 'Precision': '54.00%', 'PR-AUC': 0.3261, 'ROC-AUC': 0.7538, 'Status': 'Tree Baseline'},
        {'Model': 'Balanced Random Forest', 'Recall': '51.26%', 'Precision': '62.56%', 'PR-AUC': 0.6015, 'ROC-AUC': 0.9583, 'Status': 'Cost-Sensitive Bagging'},
        {'Model': 'Original Cost-Sensitive XGBoost', 'Recall': '91.34%', 'Precision': '44.62%', 'PR-AUC': 0.7420, 'ROC-AUC': 0.9756, 'Status': 'High-Recall Boosting'},
        {'Model': 'Tuned XGBoost (RandomizedSearch)', 'Recall': '89.53%', 'Precision': '62.63%', 'PR-AUC': 0.7798, 'ROC-AUC': 0.9811, 'Status': '★ Champion Tree Model'},
        {'Model': 'Safety-Calibrated Logistic (tau=0.193)', 'Recall': '90.25%', 'Precision': '64.27%', 'PR-AUC': 0.7896, 'ROC-AUC': 0.9830, 'Status': '★ Champion Linear Model'}
    ])
    st.dataframe(t2_table, use_container_width=True)

st.markdown("---")
st.caption("⚡ SLIIT IT3051 Data Mining Project | Group: Necrons | Dual-Task EV Battery PHM Deployment System")
