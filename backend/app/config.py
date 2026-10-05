"""Configuration schemas, slider definitions, and constants for the EV Battery PHM Backend."""

SLIDER_CONFIG = {
    # 1. Primary Degradation Indices
    'battery_health_percent': {
        'name': 'State of Health (SOH)',
        'min': 45.0, 'max': 100.0, 'step': 0.5, 'default': 85.34, 'unit': '%',
        'desc': 'Percentage of usable charge capacity relative to factory fresh rating.',
        'safe_range': 'Observed Dataset: 47.0% – 99.8% · EOL Threshold: 70.0% – 80.0%',
        'physics': 'Capacity fade occurs via active lithium trapping and SEI layer thickening.'
    },
    'capacity_loss_percent': {
        'name': 'Capacity Fade Loss',
        'min': 0.0, 'max': 55.0, 'step': 0.5, 'default': 14.66, 'unit': '%',
        'desc': 'Cumulative irreversible capacity reduction from the original pack rating.',
        'safe_range': 'Observed Dataset: 0.2% – 52.9% · Collinear with SOH (|r|=1.0)',
        'physics': 'Direct loss of active cathode material due to micro-cracking and transition metal dissolution.'
    },
    'internal_resistance': {
        'name': 'Internal Cell Resistance (ESR)',
        'min': 0.03, 'max': 1.15, 'step': 0.01, 'default': 0.22, 'unit': 'Ω',
        'desc': 'Ohmic and charge-transfer resistance to ionic transport within cells.',
        'safe_range': 'Observed: 0.03 – 1.12 Ω · Elevated Degradation: > 0.45 Ω',
        'physics': 'High resistance causes severe I²R Joule heating during acceleration and rapid voltage sag.'
    },
    'cycle_count': {
        'name': 'Completed Full Cycles',
        'min': 100, 'max': 3500, 'step': 50, 'default': 1315, 'unit': 'cycles',
        'desc': 'Equivalent 100% Depth-of-Discharge (DoD) energy cycles delivered.',
        'safe_range': 'Observed Dataset: 173 – 3,369 cycles (Mean: 1,770 cycles)',
        'physics': 'Repeated expansion/contraction induces mechanical fatigue and delamination of electrode coating.'
    },

    # 2. Thermal Dynamics
    'cell_temperature_max': {
        'name': 'Max Cell Hotspot Temp',
        'min': 15.0, 'max': 85.0, 'step': 0.5, 'default': 40.18, 'unit': '°C',
        'desc': 'Peak localized temperature recorded across pack thermal sensors.',
        'safe_range': 'Observed: -4.9°C – 86.9°C · Elevated Thermal Boundary: > 52°C',
        'physics': 'Temperatures > 55°C accelerate SEI decomposition and initiate exothermic self-heating reactions.'
    },
    'cell_temperature_avg': {
        'name': 'Mean Pack Temperature',
        'min': 15.0, 'max': 60.0, 'step': 0.5, 'default': 19.70, 'unit': '°C',
        'desc': 'Volumetric average temperature across all monitored battery modules.',
        'safe_range': 'Observed Dataset: -9.8°C – 62.1°C',
        'physics': 'Large temperature spread (Max - Avg > 12°C) causes non-uniform module aging.'
    },
    'thermal_runaway_risk': {
        'name': 'Thermal Runaway Hazard Index',
        'min': 0.0, 'max': 100.0, 'step': 1.0, 'default': 12.63, 'unit': '/100',
        'desc': 'Composite BMS probability index of uncontrolled self-accelerating heating.',
        'safe_range': 'Observed: 0.0 – 100.0 · Correlated with Thermal Health (|r|=0.987)',
        'physics': 'Scores likelihood of self-heating cascading faster than the liquid cooling loop can extract heat.'
    },
    'thermal_health_score': {
        'name': 'BMS Thermal Loop Health',
        'min': 0.0, 'max': 100.0, 'step': 1.0, 'default': 91.65, 'unit': '/100',
        'desc': 'Health indicator of coolant pumps, radiator valves, and chiller plate performance.',
        'safe_range': 'Observed Dataset: 0.0 – 100.0 (Higher indicates healthier heat rejection)',
        'physics': 'Low scores signify pump cavitation, restricted coolant passages, or aging thermal interface material.'
    },

    # 3. Charging Power & Stress
    'average_charge_power_kw': {
        'name': 'Average Charging Power',
        'min': 5.0, 'max': 150.0, 'step': 1.0, 'default': 29.39, 'unit': 'kW',
        'desc': 'Mean electrical power accepted during typical charging sessions.',
        'safe_range': 'AC Level 2: 7 – 22 kW · DC Fast Charge: 50 – 150 kW',
        'physics': 'High DC charging currents induce lithium plating at the anode when cells are cold or aged.'
    },
    'battery_capacity_kwh': {
        'name': 'Pack Energy Capacity',
        'min': 30.0, 'max': 150.0, 'step': 1.0, 'default': 83.84, 'unit': 'kWh',
        'desc': 'Total nominal nameplate energy storage capacity of the traction pack.',
        'safe_range': 'Observed Dataset: 30.0 – 149.8 kWh (Denominator for C-Rate proxy)',
        'physics': 'Used to compute the effective charging C-Rate proxy (Power / Capacity).'
    },
    'aggressive_acceleration_score': {
        'name': 'Aggressive Acceleration Score',
        'min': 0.0, 'max': 100.0, 'step': 1.0, 'default': 12.94, 'unit': '/100',
        'desc': 'Frequency of rapid throttle pedal tips and high-current discharge draws.',
        'safe_range': 'Observed Dataset: 0.0 – 100.0',
        'physics': 'High discharge current spikes create electro-mechanical shear strain on current collector tabs.'
    },
    'hard_braking_score': {
        'name': 'Hard Braking / Inrush Score',
        'min': 0.0, 'max': 100.0, 'step': 1.0, 'default': 10.40, 'unit': '/100',
        'desc': 'Frequency of sudden regenerative braking inrush currents into the cells.',
        'safe_range': 'Observed Dataset: 0.0 – 100.0',
        'physics': 'Abrupt high-rate charge acceptance spikes stress cell separators and increase voltage imbalance.'
    }
}
