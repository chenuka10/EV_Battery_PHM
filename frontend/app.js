/**
 * EV Battery Prognostics & Health Management (PHM) Dashboard
 * Frontend Application Logic (Vanilla JS, 100% Offline)
 * SLIIT IT3051 Mini Project 2026 | Group Necrons
 */

const STATE = {
  meta: null,
  activeVehicleId: "VAN-042",
  activeScenarioKey: null,
  engine: "logistic",
  tau: 0.193,
  inputs: {},
  prediction: null,
  curvesData: null,
  rankedBatchCsv: null,
  debounceTimer: null,
  vehicleRegistry: {
    "VAN-042": {
      name: "Delivery Van #042",
      model: "Mahindra BE 6 · 83.8 kWh LTO",
      chemistry: "Lithium Titanate (LTO)",
      route: "Urban Delivery Express",
      scenarioIndex: 0,
      icon: "🚚"
    },
    "VAN-017": {
      name: "Commuter Van #017",
      model: "Tesla Model S · 139.3 kWh LFP",
      chemistry: "Lithium Iron Phosphate (LFP)",
      route: "Suburban Shuttle Route",
      scenarioIndex: 1,
      icon: "🚙"
    },
    "VAN-089": {
      name: "Logistics Sprint #089",
      model: "Mahindra XUV400 · 113.4 kWh NMC",
      chemistry: "Nickel Manganese Cobalt (NMC)",
      route: "High-Speed Highway Freight",
      scenarioIndex: 2,
      icon: "⚡"
    },
    "VAN-104": {
      name: "Heavy Cargo #104",
      model: "BYD Seal · 124.7 kWh NMC",
      chemistry: "Nickel Manganese Cobalt (NMC)",
      route: "Intercity Heavy Duty",
      scenarioIndex: 3,
      icon: "🚨"
    }
  }
};

// ==============================================================================
// 1. Initialization
// ==============================================================================
document.addEventListener("DOMContentLoaded", async () => {
  try {
    await loadMetadata();
    setupNavigation();
    setupEventListeners();
    await loadThresholdData(STATE.tau);
    updateFleetSimulation();
    selectVehicle("VAN-042");
    await loadLeaderboards();
  } catch (err) {
    console.error("Initialization error:", err);
  }
});

async function loadMetadata() {
  const res = await fetch("/api/meta");
  if (!res.ok) throw new Error("Failed to load metadata from backend");
  STATE.meta = await res.json();

  renderFleetRegistry();
  renderEngineeredFormulas();
  updateFleetRibbonStats();
}

function updateFleetRibbonStats() {
  if (!STATE.meta?.scenarios) return;
  const scValues = Object.values(STATE.meta.scenarios);
  let totalSoh = 0;
  let countSoh = 0;
  let alertCount = 0;

  scValues.forEach(sc => {
    if (sc.data?.battery_health_percent) {
      totalSoh += sc.data.battery_health_percent;
      countSoh++;
    }
    if (sc.prob >= 0.193) {
      alertCount++;
    }
  });

  const avgSoh = countSoh > 0 ? (totalSoh / countSoh).toFixed(1) : "82.4";
  const sohEl = document.getElementById("kpi-fleet-soh");
  if (sohEl) sohEl.innerHTML = `${avgSoh}% <span class="kpi-sub">SOH</span>`;

  const alertEl = document.getElementById("kpi-critical-alerts");
  if (alertEl) alertEl.innerHTML = `${alertCount} <span class="kpi-sub">Vehicles</span>`;
}

// ==============================================================================
// 2. Navigation & Drawer Management
// ==============================================================================
function setupNavigation() {
  // Main view navigation tabs
  const navButtons = document.querySelectorAll(".nav-tab-btn");
  navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetView = btn.dataset.view;
      switchView(targetView);
    });
  });

  // Lab sub-tabs
  const subButtons = document.querySelectorAll(".lab-sub-tab");
  subButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      subButtons.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".lab-pane").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.subtab);
      if (target) target.classList.add("active");

      if (btn.dataset.subtab === "subtab-curves" && STATE.curvesData) {
        renderPRCurve(STATE.curvesData.curve, STATE.tau, STATE.curvesData.metrics);
        renderROCCurve(STATE.curvesData.metrics);
      }
    });
  });

  // Quick jump buttons
  document.getElementById("btn-header-switch-vehicle")?.addEventListener("click", () => switchView("view-fleet"));
  document.getElementById("btn-goto-diagnosis")?.addEventListener("click", () => switchView("view-diagnosis"));
  document.getElementById("btn-goto-prognosis")?.addEventListener("click", () => switchView("view-prognosis"));

  // Drawer triggers
  const drawer = document.getElementById("engineering-drawer");
  const backdrop = document.getElementById("drawer-backdrop");
  const btnOpen = document.getElementById("btn-open-drawer");
  const btnClose = document.getElementById("btn-close-drawer");

  const openDrawer = () => {
    drawer.classList.add("active");
    backdrop.classList.add("active");
  };

  const closeDrawer = () => {
    drawer.classList.remove("active");
    backdrop.classList.remove("active");
  };

  btnOpen?.addEventListener("click", openDrawer);
  btnClose?.addEventListener("click", closeDrawer);
  backdrop?.addEventListener("click", closeDrawer);
}

function switchView(viewId) {
  document.querySelectorAll(".nav-tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.view === viewId);
  });
  document.querySelectorAll(".view-panel").forEach(panel => {
    panel.classList.toggle("active", panel.id === viewId);
  });

  // Re-render responsive SVGs if their parent panel became visible
  if (viewId === "view-lab" && STATE.curvesData) {
    renderPRCurve(STATE.curvesData.curve, STATE.tau, STATE.curvesData.metrics);
    renderROCCurve(STATE.curvesData.metrics);
  } else if (viewId === "view-prognosis" && STATE.prediction) {
    renderDegradationCurve();
  }
}

// ==============================================================================
// 3. Fleet Assets & Vehicle Selection
// ==============================================================================
function renderFleetRegistry() {
  const container = document.getElementById("fleet-cards-grid");
  if (!container || !STATE.meta?.scenarios) return;
  container.innerHTML = "";

  const scKeys = Object.keys(STATE.meta.scenarios);

  for (const [vehId, info] of Object.entries(STATE.vehicleRegistry)) {
    const scKey = scKeys[info.scenarioIndex] || scKeys[0];
    const scObj = STATE.meta.scenarios[scKey];
    if (!scObj) continue;

    const data = scObj.data;
    const soh = data.battery_health_percent ? data.battery_health_percent.toFixed(1) + "%" : "85.0%";
    const cycles = data.cycle_count ? Math.round(data.cycle_count).toLocaleString() : "1,315";
    
    let probText = "< 0.1%";
    if (scObj.prob >= 0.001) {
      probText = (scObj.prob * 100).toFixed(1) + "%";
    }

    let badgeClass = "badge-green";
    let statusText = "HEALTHY";
    if (scObj.prob >= 0.50) {
      badgeClass = "badge-red";
      statusText = "CRITICAL";
    } else if (scObj.prob >= 0.193) {
      badgeClass = "badge-amber";
      statusText = "HAZARD (Caught τ=0.193)";
    }

    const card = document.createElement("div");
    card.className = `fleet-card ${vehId === STATE.activeVehicleId ? "active" : ""}`;
    card.id = `fleet-card-${vehId}`;
    card.onclick = () => {
      selectVehicle(vehId);
      switchView("view-overview");
    };

    card.innerHTML = `
      <div>
        <div class="fc-header">
          <div>
            <div class="fc-id">${info.icon} ${vehId}</div>
            <div class="fc-route">${info.name} · ${info.route}</div>
          </div>
          <span class="badge ${badgeClass}">${statusText}</span>
        </div>
        <div style="font-size:11px; color:var(--text-sub); margin-bottom:6px;">${info.model}</div>
        <div class="fc-metrics-row">
          <div>
            <div class="fc-m-label">State of Health</div>
            <div class="fc-m-val">${soh}</div>
          </div>
          <div>
            <div class="fc-m-label">Failure Risk</div>
            <div class="fc-m-val">${probText}</div>
          </div>
          <div>
            <div class="fc-m-label">Cycles Done</div>
            <div class="fc-m-val">${cycles}</div>
          </div>
        </div>
      </div>
      <button class="btn btn-outline btn-sm" style="width:100%;" type="button">
        ${vehId === STATE.activeVehicleId ? "★ Active Asset" : "Inspect Vehicle"}
      </button>
    `;
    container.appendChild(card);
  }
}

function selectVehicle(vehId) {
  STATE.activeVehicleId = vehId;
  const vInfo = STATE.vehicleRegistry[vehId];
  if (!vInfo) return;

  const scKeys = Object.keys(STATE.meta.scenarios);
  const scKey = scKeys[vInfo.scenarioIndex] || scKeys[0];
  STATE.activeScenarioKey = scKey;
  const scObj = STATE.meta.scenarios[scKey];
  if (!scObj) return;

  // IMPORTANT: Clone the FULL scenario telemetry dictionary so all 70+ model attributes are retained!
  STATE.inputs = { ...scObj.data };

  // Synchronize the 12 slider inputs with boundary clamps
  const cfg = STATE.meta.slider_config;
  for (const [feat, conf] of Object.entries(cfg)) {
    let val = STATE.inputs[feat];
    if (val === null || val === undefined || isNaN(val)) {
      val = conf.default;
    }
    val = Math.max(conf.min, Math.min(conf.max, parseFloat(val)));
    STATE.inputs[feat] = val;
  }

  // Update active fleet cards styling and button labels
  document.querySelectorAll(".fleet-card").forEach(c => {
    const isThis = c.id === `fleet-card-${vehId}`;
    c.classList.toggle("active", isThis);
    const btn = c.querySelector("button");
    if (btn) btn.textContent = isThis ? "★ Active Asset" : "Inspect Vehicle";
  });

  // Update vehicle headers and labels across all screens
  updateVehicleLabels(vehId, vInfo);

  // Rebuild slider controls in drawer and trigger dual-task prediction
  renderStudioSliders();
  triggerPrediction();
}

function updateVehicleLabels(vehId, vInfo) {
  document.getElementById("header-vehicle-id").textContent = vehId;
  document.getElementById("overview-vehicle-badge").textContent = `${vInfo.icon} ${vInfo.name}`;
  document.getElementById("overview-vehicle-model").textContent = vInfo.model;
  document.getElementById("diag-vehicle-title").textContent = vehId;
  document.getElementById("prog-vehicle-title").textContent = vehId;
  document.getElementById("pass-veh-id").textContent = `${vehId} (${vInfo.model})`;
  document.getElementById("pass-chemistry").textContent = vInfo.chemistry;
}

// ==============================================================================
// 4. 12-Feature Interactive BMS Engineering Drawer
// ==============================================================================
function renderStudioSliders() {
  const cfg = STATE.meta.slider_config;
  const groups = {
    degradation: ['battery_health_percent', 'capacity_loss_percent', 'internal_resistance', 'cycle_count'],
    thermal: ['cell_temperature_max', 'cell_temperature_avg', 'thermal_runaway_risk', 'thermal_health_score'],
    stress: ['average_charge_power_kw', 'battery_capacity_kwh', 'aggressive_acceleration_score', 'hard_braking_score']
  };

  for (const [groupName, feats] of Object.entries(groups)) {
    const container = document.getElementById(`group-${groupName}`);
    if (!container) continue;
    container.innerHTML = "";

    feats.forEach(feat => {
      const conf = cfg[feat];
      const val = STATE.inputs[feat] !== undefined ? STATE.inputs[feat] : conf.default;

      const ctrl = document.createElement("div");
      ctrl.className = "feature-control";
      ctrl.innerHTML = `
        <div class="feature-header">
          <label class="feature-name" for="slider-${feat}">
            ${conf.name}
            <span class="feature-info-icon" title="${conf.desc} (${conf.safe_range})">ⓘ</span>
          </label>
          <span style="font-size:11px; color:var(--text-muted);">${conf.unit}</span>
        </div>
        <div class="feature-input-row">
          <input type="range" id="slider-${feat}" class="feature-slider"
            min="${conf.min}" max="${conf.max}" step="${conf.step}" value="${val}">
          <input type="number" id="num-${feat}" class="feature-number-input"
            min="${conf.min}" max="${conf.max}" step="${conf.step}" value="${val}">
        </div>
        <div class="feature-meta">
          <span>Min: ${conf.min}</span>
          <span>Max: ${conf.max}</span>
        </div>
      `;

      const slider = ctrl.querySelector(`#slider-${feat}`);
      const numInput = ctrl.querySelector(`#num-${feat}`);

      slider.addEventListener("input", (e) => {
        const v = parseFloat(e.target.value);
        numInput.value = v;
        STATE.inputs[feat] = v;
        schedulePrediction();
      });

      numInput.addEventListener("change", (e) => {
        let v = parseFloat(e.target.value);
        if (isNaN(v)) v = conf.default;
        v = Math.max(conf.min, Math.min(conf.max, v));
        slider.value = v;
        numInput.value = v;
        STATE.inputs[feat] = v;
        schedulePrediction();
      });

      container.appendChild(ctrl);
    });
  }
}

function schedulePrediction() {
  clearTimeout(STATE.debounceTimer);
  STATE.debounceTimer = setTimeout(() => {
    triggerPrediction();
  }, 100);
}

// ==============================================================================
// 5. Dual-Task Inference & View Synchronization
// ==============================================================================
async function triggerPrediction() {
  // Physical anomaly check
  const tMax = STATE.inputs['cell_temperature_max'];
  const tAvg = STATE.inputs['cell_temperature_avg'];
  if (tMax !== undefined && tAvg !== undefined && tMax < tAvg) {
    renderAnomalyWarning(`Physical Anomaly: Maximum Hotspot Temp (${tMax.toFixed(1)}°C) cannot be lower than Average Pack Temp (${tAvg.toFixed(1)}°C).`);
    return;
  }

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        engine: STATE.engine,
        tau: STATE.tau,
        inputs: STATE.inputs
      })
    });

    if (!res.ok) {
      const err = await res.json();
      renderAnomalyWarning(err.detail || "Validation Error");
      return;
    }

    const data = await res.json();
    STATE.prediction = data;
    clearAnomalyWarning();

    updateOverviewUI(data);
    updateDiagnosisUI(data);
    updatePrognosisUI(data);
    updatePassportUI(data);
    updateAttributionUI();
  } catch (err) {
    console.error("Prediction error:", err);
  }
}

// ==============================================================================
// 6. UI Renderers
// ==============================================================================

/* View 1: Overview Screen */
function updateOverviewUI(data) {
  const soh = STATE.inputs['battery_health_percent'] !== undefined ? STATE.inputs['battery_health_percent'] : 85.0;
  const rulNum = data.rul_cycles;
  const pFail = data.p_failure;

  // Battery Graphic
  document.getElementById("hero-soh-val").textContent = `${soh.toFixed(1)}%`;
  const fill = document.getElementById("battery-fill-level");
  fill.style.height = `${Math.min(100, Math.max(12, soh))}%`;

  if (data.status === "nominal") {
    fill.style.background = "linear-gradient(180deg, #34D399 0%, #10B981 100%)";
  } else if (data.status === "early_warning") {
    fill.style.background = "linear-gradient(180deg, #FBBF24 0%, #D97706 100%)";
  } else {
    fill.style.background = "linear-gradient(180deg, #F87171 0%, #DC2626 100%)";
  }

  // Hero Status Pill
  const pill = document.getElementById("hero-status-pill");
  pill.className = `hero-status-pill ${data.status}`;
  const dot = document.getElementById("hero-status-dot");
  const text = document.getElementById("hero-status-text");

  if (data.status === "nominal") {
    dot.textContent = "🟢";
    text.textContent = "HEALTHY · Battery Operating Normally";
    document.getElementById("header-vehicle-status").textContent = `Healthy · ${soh.toFixed(1)}% SOH`;
    document.getElementById("header-vehicle-status").style.color = "var(--green-dark)";
  } else if (data.status === "early_warning") {
    dot.textContent = "⚠️";
    text.textContent = "HAZARD DETECTED · Intercepted by Safety Cutoff (τ=0.193)";
    document.getElementById("header-vehicle-status").textContent = `Warning · ${soh.toFixed(1)}% SOH`;
    document.getElementById("header-vehicle-status").style.color = "var(--amber-dark)";
  } else {
    dot.textContent = "🔴";
    text.textContent = "CRITICAL ALERT · High Failure Probability";
    document.getElementById("header-vehicle-status").textContent = `Critical Alert · ${soh.toFixed(1)}% SOH`;
    document.getElementById("header-vehicle-status").style.color = "var(--red-dark)";
  }

  // Dual Metrics
  document.getElementById("hero-risk-val").textContent = `${(pFail * 100).toFixed(2)}%`;
  document.getElementById("hero-risk-val").style.color = pFail > STATE.tau ? (pFail > 0.5 ? "var(--red)" : "var(--amber-dark)") : "var(--green-dark)";
  document.getElementById("hero-risk-sub").textContent = pFail > STATE.tau ? "Elevated Failure Risk" : "Low Risk · Safe";

  document.getElementById("hero-rul-val").textContent = Math.round(rulNum).toLocaleString();
  const yrs = (rulNum / 350.0).toFixed(1);
  document.getElementById("hero-rul-sub").textContent = `Cycles (~${yrs} Yrs*)`;

  // Condition Matrix
  const tMax = STATE.inputs['cell_temperature_max'] !== undefined ? STATE.inputs['cell_temperature_max'] : 40.2;
  const res = STATE.inputs['internal_resistance'] !== undefined ? STATE.inputs['internal_resistance'] : 0.22;
  const power = STATE.inputs['average_charge_power_kw'] !== undefined ? STATE.inputs['average_charge_power_kw'] : 29.4;
  const thermHealth = STATE.inputs['thermal_health_score'] !== undefined ? STATE.inputs['thermal_health_score'] : 91.7;

  document.getElementById("cond-temp-val").textContent = `${tMax.toFixed(1)}°C`;
  setCondBadge("cond-temp-status", tMax < 45 ? "✓ Normal" : (tMax < 55 ? "⚠ Warm" : "🛑 Hot"), tMax < 45 ? "status-good" : (tMax < 55 ? "status-warn" : "status-crit"));

  document.getElementById("cond-res-val").textContent = `${res.toFixed(3)} Ω`;
  setCondBadge("cond-res-status", res < 0.35 ? "✓ Normal" : (res < 0.60 ? "⚠ Elevated" : "🛑 High"), res < 0.35 ? "status-good" : (res < 0.60 ? "status-warn" : "status-crit"));

  document.getElementById("cond-charge-val").textContent = `${power.toFixed(1)} kW`;
  setCondBadge("cond-charge-status", power < 50 ? "✓ Normal" : "⚠ Fast Charge", power < 50 ? "status-good" : "status-warn");

  document.getElementById("cond-therm-val").textContent = `${thermHealth.toFixed(1)} / 100`;
  setCondBadge("cond-therm-status", thermHealth > 75 ? "✓ Good" : (thermHealth > 50 ? "⚠ Fair" : "🛑 Degraded"), thermHealth > 75 ? "status-good" : (thermHealth > 50 ? "status-warn" : "status-crit"));

  // Prescriptive Action
  const actionHeadline = document.getElementById("overview-action-headline");
  const actionSub = document.getElementById("overview-action-sub");
  const actionBadge = document.getElementById("action-badge");
  const quarantine = document.getElementById("overview-quarantine-status");
  const inspDays = document.getElementById("overview-inspection-days");

  if (data.status === "nominal") {
    actionHeadline.textContent = "✓ Continue normal operation";
    actionSub.textContent = "No active intervention needed. Pack operating safely within nominal electro-thermal boundaries.";
    actionBadge.textContent = "Verified Safe";
    actionBadge.className = "badge badge-green";
    quarantine.textContent = "No (Clear)";
    quarantine.style.color = "var(--green-dark)";
    inspDays.textContent = "~180 Days";
  } else if (data.status === "early_warning") {
    actionHeadline.textContent = "⚠ Schedule Coolant Loop & Cell Rebalancing";
    actionSub.textContent = "Telemetry exceeds calibrated safety boundary (τ=0.193). Schedule preventative depot inspection within 14 days to prevent on-road failure.";
    actionBadge.textContent = "Inspection Advised";
    actionBadge.className = "badge badge-amber";
    quarantine.textContent = "Preventative Bay";
    quarantine.style.color = "var(--amber-dark)";
    inspDays.textContent = "< 14 Days";
  } else {
    actionHeadline.textContent = "🛑 Immediate Vehicle Quarantine & Teardown";
    actionSub.textContent = "Severe thermal divergence or internal resistance degradation detected. Remove vehicle from service immediately.";
    actionBadge.textContent = "Critical Quarantine";
    actionBadge.className = "badge badge-red";
    quarantine.textContent = "URGENT (Lockout)";
    quarantine.style.color = "var(--red-dark)";
    inspDays.textContent = "Immediate (0 Days)";
  }
}

function setCondBadge(elId, text, cls) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.textContent = text;
  el.className = `cond-badge ${cls}`;
}

/* View 3: Diagnosis Screen */
function updateDiagnosisUI(data) {
  const hero = document.getElementById("diag-status-hero");
  const icon = document.getElementById("diag-hero-icon");
  const level = document.getElementById("diag-hero-level");
  const prob = document.getElementById("diag-hero-prob");
  const text = document.getElementById("diag-hero-text");
  const pFail = data.p_failure;

  hero.className = `diag-status-hero ${data.status}`;
  prob.textContent = `${(pFail * 100).toFixed(2)}% Failure Probability (Active τ = ${STATE.tau.toFixed(3)})`;

  if (data.status === "nominal") {
    icon.textContent = "🟢";
    level.textContent = "LOW RISK · NOMINAL HEALTH";
    text.textContent = "Pack operating safely below the critical safety cutoff (τ = 0.193). No acute thermal runaway or cell voltage divergence detected.";
  } else if (data.status === "early_warning") {
    icon.textContent = "⚠️";
    level.textContent = "EARLY WARNING · SAFETY HAZARD DETECTED";
    text.textContent = "Subtle degradation captured! While standard 0.50 threshold would mark this battery as safe, our calibrated τ = 0.193 cutoff intercepts the incipient hazard.";
  } else {
    icon.textContent = "🔴";
    level.textContent = "CRITICAL ALERT · IMMEDIATE FAILURE RISK";
    text.textContent = "Acute thermal and electrochemical breakdown. High probability of thermal runaway or irreversible cell string failure.";
  }

  // Work Order Details
  const woTitle = document.getElementById("wo-task-title");
  const woDetails = document.getElementById("wo-task-details");
  const woBadge = document.getElementById("wo-priority-badge");

  if (data.status === "nominal") {
    woTitle.textContent = "Routine Fleet Operation Clearance";
    woDetails.textContent = "Vehicle is cleared for continued commercial high-demand duty. Standard 6-month preventive maintenance logging scheduled.";
    woBadge.textContent = "Routine Priority";
    woBadge.className = "badge badge-green";
  } else if (data.status === "early_warning") {
    woTitle.textContent = "Diagnostic Teardown: Thermal & Resistance Imbalance";
    woDetails.textContent = "Dispatch vehicle to depot inspection bay within 14 days. Perform full pack impedance sweep and coolant pressure test.";
    woBadge.textContent = "High Priority";
    woBadge.className = "badge badge-amber";
  } else {
    woTitle.textContent = "EMERGENCY: Immediate Pack Quarantine";
    woDetails.textContent = "Isolate battery immediately. Disable fast charging. Perform thermal imaging sweep and prepare pack for replacement.";
    woBadge.textContent = "CRITICAL LOCKOUT";
    woBadge.className = "badge badge-red";
  }

  // Dynamic Subsystem Physical Integrity Check
  updateIntegrityChecks(data);
}

function updateIntegrityChecks(data) {
  const tMax = STATE.inputs['cell_temperature_max'] || 40.2;
  const tAvg = STATE.inputs['cell_temperature_avg'] || 20.0;
  const tSpread = data.engineered_features?.temperature_spread || (tMax - tAvg);
  const vStd = STATE.inputs['cell_voltage_std'] || 0.002;
  const vImbalance = STATE.inputs['voltage_imbalance'] || 5.0;
  const res = STATE.inputs['internal_resistance'] || 0.22;
  const cRate = data.engineered_features?.c_rate_proxy || 0.35;
  const power = STATE.inputs['average_charge_power_kw'] || 29.4;

  // 1. Thermal Check
  const tIcon = document.getElementById("icon-thermal-chk");
  const tSub = document.getElementById("chk-thermal-sub");
  const tBadge = document.getElementById("badge-thermal-chk");
  if (tSpread > 15 || tMax > 52) {
    if (tIcon) { tIcon.textContent = "🛑"; tIcon.className = "chk-icon crit"; }
    if (tBadge) { tBadge.textContent = "Fail"; tBadge.className = "cond-badge status-crit"; }
    if (tSub) tSub.textContent = `Severe hotspot temperature (${tMax.toFixed(1)}°C, spread: ${tSpread.toFixed(1)}°C) exceeding safe cooling bounds.`;
  } else if (tSpread > 10 || tMax > 45) {
    if (tIcon) { tIcon.textContent = "⚠"; tIcon.className = "chk-icon warn"; }
    if (tBadge) { tBadge.textContent = "Warning"; tBadge.className = "cond-badge status-warn"; }
    if (tSub) tSub.textContent = `Elevated thermal spread (${tSpread.toFixed(1)}°C); coolant loop operating under stress.`;
  } else {
    if (tIcon) { tIcon.textContent = "✓"; tIcon.className = "chk-icon good"; }
    if (tBadge) { tBadge.textContent = "Pass"; tBadge.className = "cond-badge status-good"; }
    if (tSub) tSub.textContent = `Max temp (${tMax.toFixed(1)}°C) vs average spread within safe boundaries.`;
  }

  // 2. Voltage Check
  const vIcon = document.getElementById("icon-voltage-chk");
  const vSub = document.getElementById("chk-voltage-sub");
  const vBadge = document.getElementById("badge-voltage-chk");
  if (vStd > 0.02 || vImbalance > 25) {
    if (vIcon) { vIcon.textContent = "🛑"; vIcon.className = "chk-icon crit"; }
    if (vBadge) { vBadge.textContent = "Imbalance"; vBadge.className = "cond-badge status-crit"; }
    if (vSub) vSub.textContent = `Severe cell voltage divergence (std: ${vStd.toFixed(4)}); string requires rebalancing.`;
  } else if (vStd > 0.01 || vImbalance > 10) {
    if (vIcon) { vIcon.textContent = "⚠"; vIcon.className = "chk-icon warn"; }
    if (vBadge) { vBadge.textContent = "Warning"; vBadge.className = "cond-badge status-warn"; }
    if (vSub) vSub.textContent = `Moderate voltage dispersion (std: ${vStd.toFixed(4)}) across internal series strings.`;
  } else {
    if (vIcon) { vIcon.textContent = "✓"; vIcon.className = "chk-icon good"; }
    if (vBadge) { vBadge.textContent = "Pass"; vBadge.className = "cond-badge status-good"; }
    if (vSub) vSub.textContent = `Dispersion std-dev is nominal (${vStd.toFixed(4)}) across internal serial strings.`;
  }

  // 3. Resistance Check
  const rIcon = document.getElementById("icon-res-chk");
  const rSub = document.getElementById("chk-res-sub");
  const rBadge = document.getElementById("badge-res-chk");
  if (res > 0.60) {
    if (rIcon) { rIcon.textContent = "🛑"; rIcon.className = "chk-icon crit"; }
    if (rBadge) { rBadge.textContent = "Degraded"; rBadge.className = "cond-badge status-crit"; }
    if (rSub) rSub.textContent = `High internal resistance (${res.toFixed(3)} Ω); severe Joule heating and voltage sag risk.`;
  } else if (res > 0.35) {
    if (rIcon) { rIcon.textContent = "⚠"; rIcon.className = "chk-icon warn"; }
    if (rBadge) { rBadge.textContent = "Warning"; rBadge.className = "cond-badge status-warn"; }
    if (rSub) rSub.textContent = `Moderate impedance growth (${res.toFixed(3)} Ω); monitor electrolyte health.`;
  } else {
    if (rIcon) { rIcon.textContent = "✓"; rIcon.className = "chk-icon good"; }
    if (rBadge) { rBadge.textContent = "Pass"; rBadge.className = "cond-badge status-good"; }
    if (rSub) rSub.textContent = `Internal resistance (${res.toFixed(3)} Ω) shows healthy electrochemical conductivity.`;
  }

  // 4. Charging Check
  const cIcon = document.getElementById("icon-charge-chk");
  const cSub = document.getElementById("chk-charge-sub");
  const cBadge = document.getElementById("badge-charge-chk");
  if (cRate > 0.7 || power > 80) {
    if (cIcon) { cIcon.textContent = "⚠"; cIcon.className = "chk-icon warn"; }
    if (cBadge) { cBadge.textContent = "High Strain"; cBadge.className = "cond-badge status-warn"; }
    if (cSub) cSub.textContent = `High C-rate fast charging (${power.toFixed(1)} kW) accelerates anode lithium plating.`;
  } else {
    if (cIcon) { cIcon.textContent = "✓"; cIcon.className = "chk-icon good"; }
    if (cBadge) { cBadge.textContent = "Pass"; cBadge.className = "cond-badge status-good"; }
    if (cSub) cSub.textContent = `Fast-charging thermal load (${power.toFixed(1)} kW) absorbed within coolant loop capacity.`;
  }
}

/* View 4: Prognosis Screen */
function updatePrognosisUI(data) {
  const rulNum = data.rul_cycles;
  document.getElementById("prog-rul-cycles").innerHTML = `${Math.round(rulNum).toLocaleString()} <span class="rul-unit">Cycles</span>`;
  
  const yrs = (rulNum / 350.0).toFixed(1);
  const km = Math.round(rulNum * 30).toLocaleString();
  const retirementYear = 2026 + Math.round(rulNum / 350.0);

  document.getElementById("prog-service-years").textContent = `~${yrs} Years*`;
  document.getElementById("prog-service-km").textContent = `~${km} km*`;
  document.getElementById("prog-retirement-date").textContent = `~${retirementYear} EOL`;

  // Circular gauge (reference 10,000 cycles)
  const circle = document.getElementById("prog-gauge-bar");
  const maxRef = 10000;
  const pct = Math.min(1.0, Math.max(0.0, rulNum / maxRef));
  const circumference = 2 * Math.PI * 42;
  const offset = circumference * (1 - pct);
  if (circle) {
    circle.style.strokeDashoffset = offset;
    circle.style.stroke = pct > 0.6 ? "var(--blue)" : (pct > 0.3 ? "var(--amber)" : "var(--red)");
  }
  document.getElementById("prog-gauge-val").textContent = `${Math.round(pct * 100)}%`;

  // Lifecycle progress fill
  const soh = STATE.inputs['battery_health_percent'] !== undefined ? STATE.inputs['battery_health_percent'] : 85.0;
  const lifeFill = document.getElementById("lifecycle-progress-fill");
  if (lifeFill) lifeFill.style.width = `${Math.min(100, Math.max(0, soh))}%`;

  const stageBadge = document.getElementById("prog-stage-badge");
  if (soh >= 80) {
    stageBadge.textContent = "Stage 1: Primary Automotive (>80% SOH)";
    stageBadge.className = "badge badge-green";
  } else if (soh >= 70) {
    stageBadge.textContent = "Stage 2: Second-Life Transition (70-80% SOH)";
    stageBadge.className = "badge badge-amber";
  } else {
    stageBadge.textContent = "Stage 3: End-of-Life Decommissioning (<70% SOH)";
    stageBadge.className = "badge badge-red";
  }

  // Draw degradation trajectory curve SVG
  renderDegradationCurve();
}

/* View 5: Battery Passport Report */
function updatePassportUI(data) {
  const soh = STATE.inputs['battery_health_percent'] !== undefined ? STATE.inputs['battery_health_percent'] : 85.0;
  const rulNum = data.rul_cycles;
  const pFail = data.p_failure;

  document.getElementById("pass-soh").textContent = `${soh.toFixed(1)}%`;
  document.getElementById("pass-cycles").textContent = `${Math.round(STATE.inputs['cycle_count'] || 1315).toLocaleString()} Cycles`;
  document.getElementById("pass-capacity").textContent = `${(STATE.inputs['battery_capacity_kwh'] || 83.8).toFixed(1)} kWh`;
  document.getElementById("pass-rul").textContent = `${Math.round(rulNum).toLocaleString()} Cycles (~${(rulNum/350).toFixed(1)} Yrs*)`;
  document.getElementById("pass-prob").textContent = `${(pFail * 100).toFixed(2)}% (${data.status_label})`;
  document.getElementById("pass-cutoff").textContent = `τ = ${STATE.tau.toFixed(3)} (Safety)`;

  // Stamp and Grade calculation
  const stamp = document.getElementById("passport-grade-stamp");
  const stampGrade = document.getElementById("passport-stamp-grade");
  const stampSub = document.getElementById("passport-stamp-sub");
  const triageText = document.getElementById("second-life-triage-text");

  let grade = "GRADE A";
  let gradeSub = "Automotive Tier-1";

  if (pFail > STATE.tau || soh < 70) {
    grade = "GRADE C";
    gradeSub = "Direct Recycling Required";
    stamp.style.borderColor = "var(--red)";
    stampGrade.style.color = "var(--red-dark)";
    stampSub.style.color = "var(--red)";
    triageText.textContent = "🛑 Critical Decommission: Direct Material Recovery";
    triageText.style.color = "var(--red-dark)";
  } else if (soh < 82 || rulNum < 2500) {
    grade = "GRADE B";
    gradeSub = "Certified Second-Life BESS";
    stamp.style.borderColor = "var(--amber)";
    stampGrade.style.color = "var(--amber-dark)";
    stampSub.style.color = "var(--amber)";
    triageText.textContent = "✓ Certified for Stationary Battery Energy Storage (BESS)";
    triageText.style.color = "var(--amber-dark)";
  } else {
    grade = "GRADE A";
    gradeSub = "Automotive Tier-1";
    stamp.style.borderColor = "var(--green)";
    stampGrade.style.color = "var(--green-dark)";
    stampSub.style.color = "var(--green)";
    triageText.textContent = "✓ Approved for Continued Primary Automotive Traction";
    triageText.style.color = "var(--green-dark)";
  }

  stampGrade.textContent = grade;
  stampSub.textContent = gradeSub;

  // Residual Valuation Model
  const cap = STATE.inputs['battery_capacity_kwh'] || 83.8;
  const activeKwh = (cap * (soh / 100.0));
  const baseResale = Math.round(activeKwh * 75);
  const salvage = Math.round(activeKwh * 25);

  document.getElementById("val-resale-price").textContent = `$${baseResale.toLocaleString()}`;
  document.getElementById("val-salvage-price").textContent = `$${salvage.toLocaleString()}`;
}

// ==============================================================================
// 7. Degradation Curve SVG Renderer
// ==============================================================================
function renderDegradationCurve() {
  const svg = document.getElementById("svg-degradation-curve");
  if (!svg || !STATE.prediction) return;

  const w = 440, h = 250;
  const padL = 45, padR = 25, padT = 20, padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const currentSOH = Math.min(100, Math.max(45, STATE.inputs['battery_health_percent'] !== undefined ? STATE.inputs['battery_health_percent'] : 85.0));
  const rulCycles = STATE.prediction.rul_cycles || 7825;
  const remainingYears = Math.min(25, Math.max(0.5, rulCycles / 350.0));

  const startYear = 2022;
  const currentYear = 2026;
  const eolYear = currentYear + remainingYears;
  const maxYear = Math.max(2036, Math.min(2052, Math.round(eolYear + 2)));

  const yearToX = (yr) => padL + ((yr - startYear) / (maxYear - startYear)) * plotW;
  const sohToY = (soh) => padT + ((100 - soh) / (100 - 55)) * plotH;

  const xStart = yearToX(startYear);
  const yStart = sohToY(100);

  const xNow = yearToX(currentYear);
  const yNow = sohToY(currentSOH);

  const xEol = yearToX(eolYear);
  const yEol = sohToY(70);
  const y70 = sohToY(70);

  let pathD = "";
  if (currentSOH >= 70) {
    const c1x = xStart + (xNow - xStart) * 0.5;
    const c1y = yStart;
    const c2x = xNow - (xNow - xStart) * 0.2;
    const c2y = yNow;

    const c3x = xNow + (xEol - xNow) * 0.3;
    const c3y = yNow;
    const c4x = xEol - (xEol - xNow) * 0.3;
    const c4y = yEol;

    pathD = `M ${xStart.toFixed(1)} ${yStart.toFixed(1)} C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${xNow.toFixed(1)} ${yNow.toFixed(1)} C ${c3x.toFixed(1)} ${c3y.toFixed(1)}, ${c4x.toFixed(1)} ${c4y.toFixed(1)}, ${xEol.toFixed(1)} ${yEol.toFixed(1)}`;
  } else {
    // Already below 70% EOL
    pathD = `M ${xStart.toFixed(1)} ${yStart.toFixed(1)} C ${(xStart + 30).toFixed(1)} ${yStart.toFixed(1)}, ${(xNow - 30).toFixed(1)} ${yNow.toFixed(1)}, ${xNow.toFixed(1)} ${yNow.toFixed(1)}`;
  }

  svg.innerHTML = `
    <!-- Axes -->
    <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}" stroke="#E2E8F0" stroke-width="1.5" />
    <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" stroke="#E2E8F0" stroke-width="1.5" />

    <!-- Y Labels (SOH %) -->
    <text x="${padL - 8}" y="${sohToY(100) + 4}" text-anchor="end" font-size="10" fill="#64748B">100%</text>
    <text x="${padL - 8}" y="${sohToY(85) + 4}" text-anchor="end" font-size="10" fill="#64748B">85%</text>
    <text x="${padL - 8}" y="${y70 + 4}" text-anchor="end" font-size="10" fill="#EF4444" font-weight="700">70%</text>
    <text x="14" y="${h / 2}" font-size="10" fill="#475569" transform="rotate(-90 14 ${h / 2})" text-anchor="middle" font-weight="600">State of Health (SOH)</text>

    <!-- X Labels (Years) -->
    <text x="${xStart}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#64748B">${startYear}</text>
    <text x="${xNow}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#2563EB" font-weight="700">${currentYear}</text>
    ${currentSOH >= 70 ? `
      <text x="${Math.min(xEol, w - padR)}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#EF4444" font-weight="700">~${Math.round(eolYear)}</text>
    ` : ''}
    <text x="${padL + plotW / 2}" y="${h - 8}" text-anchor="middle" font-size="10" fill="#475569" font-weight="600">Operating Timeline (Calendar Years)</text>

    <!-- 70% EOL Threshold Line -->
    <line x1="${padL}" y1="${y70}" x2="${w - padR}" y2="${y70}" stroke="#FECACA" stroke-dasharray="4" stroke-width="1.5" />
    <text x="${w - padR}" y="${y70 - 6}" text-anchor="end" font-size="9" fill="#DC2626" font-weight="700">70% Critical Automotive EOL</text>

    <!-- Trajectory Path -->
    <path d="${pathD}" fill="none" stroke="#2563EB" stroke-width="2.5" />

    <!-- Current Point Marker (NOW) -->
    <circle cx="${xNow.toFixed(1)}" cy="${yNow.toFixed(1)}" r="6" fill="#2563EB" stroke="#FFF" stroke-width="2" />
    <rect x="${xNow - 25}" y="${yNow - 28}" width="50" height="20" rx="4" fill="#0F172A" />
    <text x="${xNow}" y="${yNow - 14}" text-anchor="middle" font-size="10" fill="#FFF" font-weight="700">NOW</text>

    <!-- EOL Intercept Marker -->
    ${currentSOH >= 70 ? `
      <circle cx="${xEol.toFixed(1)}" cy="${yEol.toFixed(1)}" r="6" fill="#EF4444" stroke="#FFF" stroke-width="2" />
      <rect x="${Math.min(xEol - 25, w - padR - 55)}" y="${yEol - 28}" width="55" height="20" rx="4" fill="#DC2626" />
      <text x="${Math.min(xEol + 2, w - padR - 27)}" y="${yEol - 14}" text-anchor="middle" font-size="10" fill="#FFF" font-weight="700">EOL ~${Math.round(eolYear)}</text>
    ` : ''}
  `;
}

// ==============================================================================
// 8. Feature Attribution & Engineering
// ==============================================================================
async function updateAttributionUI() {
  const container = document.getElementById("attribution-bars");
  if (!container) return;

  try {
    const res = await fetch("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        engine: STATE.engine,
        inputs: STATE.inputs
      })
    });

    if (res.ok) {
      const explainData = await res.json();
      container.innerHTML = "";

      // Render top risk factors
      if (explainData.top_risk_factors && explainData.top_risk_factors.length > 0) {
        const titleEl = document.createElement("div");
        titleEl.style.fontSize = "11px";
        titleEl.style.fontWeight = "700";
        titleEl.style.color = "var(--red-dark)";
        titleEl.style.textTransform = "uppercase";
        titleEl.style.marginBottom = "8px";
        titleEl.textContent = "▲ Factors Elevating Failure Risk";
        container.appendChild(titleEl);

        explainData.top_risk_factors.forEach(item => {
          const row = document.createElement("div");
          row.className = "bar-chart-row";
          const score = Math.min(100, Math.max(15, Math.abs(item.weight) * 60));
          const cleanName = item.feature.replace(/_/g, " ").replace("num__", "").replace("cat__", "");
          row.innerHTML = `
            <div class="bar-label">
              <span style="text-transform:capitalize;">${cleanName}</span>
              <span style="color:var(--red-dark); font-weight:700;">+${item.weight.toFixed(2)}</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill danger" style="width: ${score.toFixed(1)}%;"></div>
            </div>
          `;
          container.appendChild(row);
        });
      }

      // Render top protective factors
      if (explainData.top_protective_factors && explainData.top_protective_factors.length > 0) {
        const protTitle = document.createElement("div");
        protTitle.style.fontSize = "11px";
        protTitle.style.fontWeight = "700";
        protTitle.style.color = "var(--green-dark)";
        protTitle.style.textTransform = "uppercase";
        protTitle.style.margin = "14px 0 8px";
        protTitle.textContent = "▼ Protective Factors (Preserving Health)";
        container.appendChild(protTitle);

        explainData.top_protective_factors.forEach(item => {
          const row = document.createElement("div");
          row.className = "bar-chart-row";
          const score = Math.min(100, Math.max(15, Math.abs(item.weight) * 50));
          const cleanName = item.feature.replace(/_/g, " ").replace("num__", "").replace("cat__", "");
          row.innerHTML = `
            <div class="bar-label">
              <span style="text-transform:capitalize;">${cleanName}</span>
              <span style="color:var(--green-dark); font-weight:700;">${item.weight.toFixed(2)}</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill safe" style="width: ${score.toFixed(1)}%;"></div>
            </div>
          `;
          container.appendChild(row);
        });
      }
      return;
    }
  } catch (err) {
    console.warn("Could not fetch attribution from /api/explain, falling back to local indices", err);
  }

  // Fallback if network or error
  renderLocalAttributionFallback(container);
}

function renderLocalAttributionFallback(container) {
  const tMax = STATE.inputs['cell_temperature_max'] || 40.0;
  const res = STATE.inputs['internal_resistance'] || 0.22;
  const cycles = STATE.inputs['cycle_count'] || 1315;
  const loss = STATE.inputs['capacity_loss_percent'] || 14.6;
  const power = STATE.inputs['average_charge_power_kw'] || 29.4;

  const factors = [
    { name: "Max Cell Hotspot Temp", val: tMax, unit: "°C", impact: tMax > 52 ? "danger" : (tMax > 38 ? "warning" : "safe"), score: Math.min(100, (tMax / 85) * 100) },
    { name: "Internal Cell Resistance", val: res, unit: "Ω", impact: res > 0.45 ? "danger" : (res > 0.25 ? "warning" : "safe"), score: Math.min(100, (res / 1.1) * 100) },
    { name: "Completed Cycles Wear", val: cycles, unit: "cyc", impact: cycles > 2500 ? "danger" : (cycles > 1500 ? "warning" : "safe"), score: Math.min(100, (cycles / 3500) * 100) },
    { name: "Capacity Fade Loss", val: loss, unit: "%", impact: loss > 30 ? "danger" : (loss > 15 ? "warning" : "safe"), score: Math.min(100, (loss / 50) * 100) },
    { name: "Average Charging C-Rate", val: power, unit: "kW", impact: power > 80 ? "warning" : "neutral", score: Math.min(100, (power / 150) * 100) }
  ];

  container.innerHTML = "";
  factors.forEach(f => {
    const row = document.createElement("div");
    row.className = "bar-chart-row";
    row.innerHTML = `
      <div class="bar-label">
        <span>${f.name}</span>
        <span style="color:var(--text-muted);">${f.val.toFixed(2)} ${f.unit}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill ${f.impact}" style="width: ${f.score.toFixed(1)}%;"></div>
      </div>
    `;
    container.appendChild(row);
  });
}

function renderEngineeredFormulas() {
  const container = document.getElementById("engineered-list");
  if (!container || !STATE.meta?.engineered_formulas) return;

  container.innerHTML = "";
  for (const [name, formula] of Object.entries(STATE.meta.engineered_formulas)) {
    const box = document.createElement("div");
    box.style.background = "var(--surface-alt)";
    box.style.padding = "8px 12px";
    box.style.borderRadius = "var(--radius-sm)";
    box.innerHTML = `
      <strong style="color:var(--blue);">${name}:</strong>
      <code style="display:block; font-size:10px; color:var(--text-sub); margin-top:2px;">${formula}</code>
    `;
    container.appendChild(box);
  }
}

// ==============================================================================
// 9. PHM Lab: Thresholds, Curves, Leaderboards, & Simulation
// ==============================================================================
async function loadThresholdData(tau) {
  try {
    const res = await fetch(`/api/threshold?engine=${STATE.engine}&tau=${tau}`);
    if (!res.ok) return;
    const data = await res.json();
    STATE.curvesData = data;

    // Update confusion matrix
    document.getElementById("cm-tp").textContent = data.metrics.tp.toLocaleString();
    document.getElementById("cm-fn").textContent = data.metrics.fn.toLocaleString();
    document.getElementById("cm-fp").textContent = data.metrics.fp.toLocaleString();
    document.getElementById("cm-prec").textContent = `${(data.metrics.precision * 100).toFixed(2)}%`;

    const recPct = (data.metrics.recall * 100).toFixed(2);
    const recBadge = document.getElementById("recall-badge");
    if (recBadge) recBadge.textContent = `${recPct}% Recall`;

    renderPRCurve(data.curve, tau, data.metrics);
    renderROCCurve(data.metrics);
  } catch (err) {
    console.error("Threshold fetch error:", err);
  }
}

function renderPRCurve(curve, currentTau, metrics) {
  const svg = document.getElementById("svg-pr-curve");
  if (!svg || !curve || !curve.recalls || curve.recalls.length === 0) return;

  const w = 400, h = 240;
  const padL = 45, padR = 25, padT = 20, padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  let pathD = "";
  let currentPt = null;

  for (let i = 0; i < curve.taus.length; i++) {
    const rec = curve.recalls[i];
    const prec = curve.precisions[i];
    const x = padL + rec * plotW;
    const y = padT + (1 - prec) * plotH;

    if (i === 0) pathD += `M ${x.toFixed(1)} ${y.toFixed(1)}`;
    else pathD += ` L ${x.toFixed(1)} ${y.toFixed(1)}`;

    if (Math.abs(curve.taus[i] - currentTau) < 0.015 && !currentPt) {
      currentPt = { x, y, rec, prec };
    }
  }

  const floorY = padT + (1 - 0.0693) * plotH;

  svg.innerHTML = `
    <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}" stroke="#E2E8F0" stroke-width="1.5" />
    <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" stroke="#E2E8F0" stroke-width="1.5" />
    
    <text x="${padL - 8}" y="${padT + 5}" text-anchor="end" font-size="10" fill="#64748B">1.0</text>
    <text x="${padL - 8}" y="${padT + plotH * 0.5 + 4}" text-anchor="end" font-size="10" fill="#64748B">0.5</text>
    <text x="${padL - 8}" y="${h - padB}" text-anchor="end" font-size="10" fill="#64748B">0.0</text>
    <text x="14" y="${h / 2}" font-size="10" fill="#475569" transform="rotate(-90 14 ${h / 2})" text-anchor="middle" font-weight="600">Precision</text>

    <text x="${padL}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#64748B">0.0</text>
    <text x="${padL + plotW * 0.5}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#64748B">0.5</text>
    <text x="${w - padR}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#64748B">1.0</text>
    <text x="${padL + plotW / 2}" y="${h - 8}" text-anchor="middle" font-size="10" fill="#475569" font-weight="600">Recall (Sensitivity)</text>

    <line x1="${padL}" y1="${floorY}" x2="${w - padR}" y2="${floorY}" stroke="#CBD5E1" stroke-dasharray="4" stroke-width="1" />
    <text x="${w - padR - 5}" y="${floorY - 4}" text-anchor="end" font-size="9" fill="#94A3B8">Naive Floor (6.9%)</text>

    <path d="${pathD}" fill="none" stroke="#2563EB" stroke-width="2.5" />

    ${currentPt ? `
      <line x1="${currentPt.x}" y1="${padT}" x2="${currentPt.x}" y2="${h - padB}" stroke="#10B981" stroke-dasharray="3" stroke-width="1" />
      <circle cx="${currentPt.x}" cy="${currentPt.y}" r="6" fill="#10B981" stroke="#FFF" stroke-width="2" />
      <rect x="${Math.min(currentPt.x + 8, w - 110)}" y="${Math.max(currentPt.y - 28, padT)}" width="96" height="24" rx="4" fill="#0F172A" />
      <text x="${Math.min(currentPt.x + 14, w - 104)}" y="${Math.max(currentPt.y - 12, padT + 16)}" font-size="10" fill="#FFF" font-weight="700">τ=${currentTau.toFixed(3)}: Rec ${(currentPt.rec*100).toFixed(1)}%</text>
    ` : ''}
  `;
}

function renderROCCurve(metrics) {
  const svg = document.getElementById("svg-roc-curve");
  if (!svg) return;

  const w = 400, h = 240;
  const padL = 45, padR = 25, padT = 20, padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  let rocD = `M ${padL} ${h - padB}`;
  const points = [
    [0.0, 0.0], [0.01, 0.70], [0.038, 0.9025], [0.08, 0.96], [0.15, 0.98], [0.4, 0.995], [1.0, 1.0]
  ];

  points.forEach((pt, i) => {
    const x = padL + pt[0] * plotW;
    const y = padT + (1 - pt[1]) * plotH;
    if (i === 0) rocD = `M ${x.toFixed(1)} ${y.toFixed(1)}`;
    else rocD += ` L ${x.toFixed(1)} ${y.toFixed(1)}`;
  });

  const markerX = padL + (139 / 3723) * plotW;
  const markerY = padT + (1 - (250 / 277)) * plotH;

  svg.innerHTML = `
    <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}" stroke="#E2E8F0" stroke-width="1.5" />
    <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" stroke="#E2E8F0" stroke-width="1.5" />
    <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${padT}" stroke="#CBD5E1" stroke-dasharray="4" stroke-width="1.5" />

    <text x="${padL - 8}" y="${padT + 5}" text-anchor="end" font-size="10" fill="#64748B">1.0</text>
    <text x="${padL - 8}" y="${padT + plotH * 0.5 + 4}" text-anchor="end" font-size="10" fill="#64748B">0.5</text>
    <text x="${padL - 8}" y="${h - padB}" text-anchor="end" font-size="10" fill="#64748B">0.0</text>
    <text x="14" y="${h / 2}" font-size="10" fill="#475569" transform="rotate(-90 14 ${h / 2})" text-anchor="middle" font-weight="600">True Positive Rate</text>

    <text x="${padL}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#64748B">0.0</text>
    <text x="${padL + plotW * 0.5}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#64748B">0.5</text>
    <text x="${w - padR}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#64748B">1.0</text>
    <text x="${padL + plotW / 2}" y="${h - 8}" text-anchor="middle" font-size="10" fill="#475569" font-weight="600">False Positive Rate</text>

    <path d="${rocD}" fill="none" stroke="#10B981" stroke-width="2.5" />

    <circle cx="${markerX.toFixed(1)}" cy="${markerY.toFixed(1)}" r="6" fill="#10B981" stroke="#FFF" stroke-width="2" />
    <rect x="${markerX + 8}" y="${markerY - 20}" width="95" height="24" rx="4" fill="#0F172A" />
    <text x="${markerX + 14}" y="${markerY - 4}" font-size="10" fill="#FFF" font-weight="700">TPR 90.3% · FPR 3.7%</text>
  `;
}

async function loadLeaderboards() {
  try {
    const res = await fetch("/api/leaderboard");
    if (!res.ok) return;
    const data = await res.json();

    // Task 1
    const t1Body = document.getElementById("leaderboard-t1-body");
    if (t1Body) {
      t1Body.innerHTML = "";
      data.task1.forEach(row => {
        const tr = document.createElement("tr");
        if (row.status.includes("★")) tr.className = "champion-row";
        tr.innerHTML = `
          <td><strong>${row.model}</strong></td>
          <td>${row.paradigm}</td>
          <td>${row.r2.toFixed(4)}</td>
          <td>${row.rmse.toFixed(2)}</td>
          <td>${row.mae.toFixed(2)}</td>
          <td><span class="badge ${row.status.includes('★') ? 'badge-blue' : 'badge-gray'}">${row.status}</span></td>
        `;
        t1Body.appendChild(tr);
      });
    }

    // Task 2
    const t2Body = document.getElementById("leaderboard-t2-body");
    if (t2Body) {
      t2Body.innerHTML = "";
      data.task2.forEach(row => {
        const tr = document.createElement("tr");
        if (row.status.includes("★")) tr.className = "champion-row";
        tr.innerHTML = `
          <td><strong>${row.model}</strong></td>
          <td>${(row.accuracy * 100).toFixed(2)}%</td>
          <td>${row.precision}</td>
          <td>${row.recall}</td>
          <td>${row.f1}</td>
          <td>${row.pr_auc.toFixed(4)}</td>
          <td>${row.roc_auc.toFixed(4)}</td>
          <td style="color:${row.missed > 100 ? 'var(--red)' : 'var(--text-main)'}; font-weight:600;">${row.missed}</td>
          <td>${row.false_alarms}</td>
          <td><span class="badge ${row.status.includes('★') ? 'badge-green' : 'badge-gray'}">${row.status}</span></td>
        `;
        t2Body.appendChild(tr);
      });
    }
  } catch (err) {
    console.error("Leaderboard fetch error:", err);
  }
}

function updateFleetSimulation() {
  const nVehicles = parseInt(document.getElementById("sim-fleet-size")?.value || 10000);
  const techHours = parseInt(document.getElementById("sim-hours")?.value || 1200);

  const elFleet = document.getElementById("sim-fleet-size-val");
  if (elFleet) elFleet.textContent = nVehicles.toLocaleString();

  const elHours = document.getElementById("sim-hours-val");
  if (elHours) elHours.textContent = `${techHours.toLocaleString()} hrs`;

  const totalFailures = Math.round(nVehicles * (277.0 / 4000.0));
  const recall = STATE.curvesData?.metrics?.recall || 0.9025;
  const fpRate = (STATE.curvesData?.metrics?.fp || 139) / 3723.0;

  const intercepted = Math.round(totalFailures * recall);
  const falseAlarms = Math.round((nVehicles - totalFailures) * fpRate);
  const totalInspections = intercepted + falseAlarms;

  const elInt = document.getElementById("sim-intercepted");
  if (elInt) elInt.textContent = intercepted.toLocaleString();

  const elWork = document.getElementById("sim-workload");
  if (elWork) elWork.textContent = `${totalInspections.toLocaleString()} teardowns`;
}

// ==============================================================================
// 10. Batch CSV Upload
// ==============================================================================
async function handleBatchUpload(file) {
  const formData = new FormData();
  formData.append("file", file);

  const container = document.getElementById("batch-results-container");
  const summary = document.getElementById("batch-summary-text");
  const tbody = document.getElementById("batch-table-body");

  summary.textContent = "Processing telemetry records...";
  container.style.display = "block";

  try {
    const res = await fetch(`/api/batch?engine=${STATE.engine}&tau=${STATE.tau}`, {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Batch processing error: ${err.detail || 'Invalid CSV'}`);
      return;
    }

    const data = await res.json();
    STATE.rankedBatchCsv = data.csv_download;

    summary.innerHTML = `
      Total Analyzed: <strong>${data.total_records.toLocaleString()}</strong> ·
      Flagged for Depot Inspection: <strong style="color:var(--amber-dark);">${data.total_flagged} vehicles</strong>
      (τ = ${data.tau.toFixed(3)})
    `;

    tbody.innerHTML = "";
    data.ranked_records.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>Vehicle #${r.row_index}</td>
        <td><strong>${Math.round(r.rul_cycles).toLocaleString()} cycles</strong></td>
        <td><strong style="color:${r.p_failure > STATE.tau ? 'var(--red)' : 'var(--green-dark)'};">${(r.p_failure * 100).toFixed(2)}%</strong></td>
        <td><span class="badge ${r.flagged ? (r.p_failure > 0.5 ? 'badge-red' : 'badge-amber') : 'badge-green'}">${r.status}</span></td>
        <td>${r.flagged ? "YES (DEPOT)" : "NO (OK)"}</td>
        <td>${r.imputed_count > 0 ? `${r.imputed_count} fields imputed` : "Complete"}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Batch upload failed:", err);
    alert("Batch upload failed. Verify backend service is running.");
  }
}

// ==============================================================================
// 11. Event Listeners
// ==============================================================================
function setupEventListeners() {
  // Engine select
  document.getElementById("engine-select")?.addEventListener("change", (e) => {
    STATE.engine = e.target.value;
    loadThresholdData(STATE.tau);
    triggerPrediction();
  });

  // Threshold slider
  const tauSlider = document.getElementById("tau-slider");
  tauSlider?.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    STATE.tau = val;
    document.getElementById("slider-tau-indicator").innerHTML = `<strong>Active τ = ${val.toFixed(3)}</strong>`;
    document.querySelectorAll(".mode-pill").forEach(p => p.classList.remove("active"));
    loadThresholdData(val);
    triggerPrediction();
  });

  // Mode pills
  document.querySelectorAll(".mode-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".mode-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      const tau = parseFloat(pill.dataset.tau);
      STATE.tau = tau;
      if (tauSlider) tauSlider.value = tau;
      document.getElementById("slider-tau-indicator").innerHTML = `<strong>Active τ = ${tau.toFixed(3)}</strong>`;
      loadThresholdData(tau);
      triggerPrediction();
    });
  });

  // Reset studio button
  document.getElementById("btn-reset-studio")?.addEventListener("click", () => {
    if (STATE.activeVehicleId) selectVehicle(STATE.activeVehicleId);
  });

  // Export passport (Markdown)
  document.getElementById("btn-export-passport")?.addEventListener("click", exportHealthPassport);
  document.getElementById("btn-download-passport-md")?.addEventListener("click", exportHealthPassport);

  // Print passport (Native Browser Print to PDF)
  document.getElementById("btn-print-passport")?.addEventListener("click", () => {
    switchView("view-report");
    setTimeout(() => {
      window.print();
    }, 200);
  });

  // Fleet simulator sliders
  document.getElementById("sim-fleet-size")?.addEventListener("input", updateFleetSimulation);
  document.getElementById("sim-hours")?.addEventListener("input", updateFleetSimulation);

  // Batch CSV file dropzone
  const dropzone = document.getElementById("batch-dropzone");
  const fileInput = document.getElementById("batch-file-input");

  dropzone?.addEventListener("click", () => fileInput.click());
  fileInput?.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleBatchUpload(e.target.files[0]);
  });

  dropzone?.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.style.borderColor = "var(--blue)"; });
  dropzone?.addEventListener("dragleave", () => { dropzone.style.borderColor = "var(--border)"; });
  dropzone?.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--border)";
    if (e.dataTransfer.files.length > 0) handleBatchUpload(e.dataTransfer.files[0]);
  });

  // Batch download ranked CSV
  document.getElementById("btn-download-ranked-csv")?.addEventListener("click", () => {
    if (!STATE.rankedBatchCsv) return;
    const blob = new Blob([STATE.rankedBatchCsv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `EV_Fleet_Ranked_Triage_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // Self-test diagnostic button
  document.getElementById("btn-run-selftest")?.addEventListener("click", async () => {
    const box = document.getElementById("selftest-output");
    box.textContent = "Running live /api/selftest diagnostic across all presets...";
    try {
      const res = await fetch("/api/selftest");
      const data = await res.json();
      box.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      box.textContent = `Diagnostic failed: ${err.message}`;
    }
  });
}

function exportHealthPassport() {
  if (!STATE.prediction) return;
  const p = STATE.prediction;
  const vInfo = STATE.vehicleRegistry[STATE.activeVehicleId] || { name: "Vehicle #042", model: "Mahindra BE 6", chemistry: "LTO" };
  const d = new Date().toISOString();

  const md = `# EV Battery Health & Valuation Passport
Generated: ${d}
System: SLIIT IT3051 Fundamentals of Data Mining · Group Necrons
Asset: ${vInfo.name} (${vInfo.model}) · ${vInfo.chemistry}

---
## 1. Executive Health & Safety Diagnosis (Task 2)
- **Status:** ${p.status_label}
- **Failure Probability:** ${(p.p_failure * 100).toFixed(2)}%
- **Active Safety Cutoff (τ):** ${STATE.tau.toFixed(3)} (Safety standard: 90.25% test recall)
- **Immediate Action:** ${p.alert_action}

---
## 2. Longevity Prognostics (Task 1)
- **Remaining Useful Life (RUL):** ${Math.round(p.rul_cycles).toLocaleString()} cycles
- **Projected Service Life:** ~${(p.rul_cycles / 350.0).toFixed(1)} operating years*
- **Projected Road Longevity:** ~${Math.round(p.rul_cycles * 30).toLocaleString()} km*
- *Note: Calendar and distance conversions assume 350 full cycles/year and ~30 km/cycle profile.*
- **Regression Engine:** ${p.model_names.task1} (Test R²: 0.8972, RMSE: 528.64)

---
## 3. Second-Life Circular Economy Valuation
- **State of Health (SOH):** ${(STATE.inputs['battery_health_percent'] !== undefined ? STATE.inputs['battery_health_percent'] : 85.0).toFixed(1)}%
- **Asset Triage Grade:** ${p.p_failure > STATE.tau ? "GRADE C (Direct Recycling)" : (STATE.inputs['battery_health_percent'] < 82 ? "GRADE B (Stationary BESS)" : "GRADE A (Primary Automotive)")}
- **Estimated Resale Valuation:** $${Math.round((STATE.inputs['battery_capacity_kwh'] || 83.8) * ((STATE.inputs['battery_health_percent'] !== undefined ? STATE.inputs['battery_health_percent'] : 85.0)/100) * 75).toLocaleString()}
- **Stationary BESS Salvage Value:** $${Math.round((STATE.inputs['battery_capacity_kwh'] || 83.8) * ((STATE.inputs['battery_health_percent'] !== undefined ? STATE.inputs['battery_health_percent'] : 85.0)/100) * 25).toLocaleString()}

---
## 4. Key Physical Telemetry & Engineered Features
- **Max Hotspot Temp:** ${(STATE.inputs['cell_temperature_max'] !== undefined ? STATE.inputs['cell_temperature_max'] : 40.2).toFixed(1)} °C
- **Internal Cell Resistance:** ${(STATE.inputs['internal_resistance'] !== undefined ? STATE.inputs['internal_resistance'] : 0.22).toFixed(3)} Ω
- **Temperature Spread:** ${p.engineered_features.temperature_spread.toFixed(2)} °C
- **Efficiency Gap:** ${p.engineered_features.efficiency_gap.toFixed(2)} %
- **C-Rate Proxy:** ${p.engineered_features.c_rate_proxy.toFixed(4)}

---
*Notice: Dual-task machine learning edge inference prototype for research and evaluation purposes only.*
`;

  const blob = new Blob([md], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `EV_Battery_Passport_${STATE.activeVehicleId}_${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(url);
}

function renderAnomalyWarning(msg) {
  const warnContainer = document.getElementById("warnings-box");
  if (!warnContainer) return;
  warnContainer.innerHTML = `
    <div class="val-item warning" style="background:var(--red-bg); border-color:var(--red-border); color:var(--red-dark);">
      <span>🛑</span><span>${msg}</span>
    </div>
  `;
}

function clearAnomalyWarning() {
  const warnContainer = document.getElementById("warnings-box");
  if (warnContainer) warnContainer.innerHTML = "";
}
