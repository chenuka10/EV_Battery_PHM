/**
 * EV Battery Prognostics & Health Management (PHM) Dashboard
 * Frontend Application Logic (Vanilla JS, 100% Offline)
 * SLIIT IT3051 Mini Project 2026 | Group Necrons
 */

const STATE = {
  meta: null,
  activeScenarioKey: null,
  engine: "logistic",
  tau: 0.193,
  inputs: {},
  prediction: null,
  curvesData: null,
  rankedBatchCsv: null,
  debounceTimer: null
};

// ==============================================================================
// 1. Initialization
// ==============================================================================
document.addEventListener("DOMContentLoaded", async () => {
  try {
    await loadMetadata();
    setupEventListeners();
    await loadThresholdData(STATE.tau);
    await triggerPrediction();
    await loadLeaderboards();
  } catch (err) {
    console.error("Initialization error:", err);
  }
});

async function loadMetadata() {
  const res = await fetch("/api/meta");
  if (!res.ok) throw new Error("Failed to load metadata from backend");
  STATE.meta = await res.json();

  // Populate Scenarios
  renderScenarioCards();

  // Load default scenario (Factory Fresh)
  const scKeys = Object.keys(STATE.meta.scenarios);
  if (scKeys.length > 0) {
    selectScenario(scKeys[0]);
  }

  // Render 7 domain engineered feature formulas in Tab 2
  renderEngineeredFormulas();
}

// ==============================================================================
// 2. Scenario Showcase Cards
// ==============================================================================
function renderScenarioCards() {
  const container = document.getElementById("scenarios-container");
  container.innerHTML = "";

  const scenarios = STATE.meta.scenarios;
  const metaLookup = [
    { icon: "🟢", badge: "badge-green", badgeText: "NOMINAL HEALTH", sub: "Low Cycles · Peak Health" },
    { icon: "🔵", badge: "badge-blue", badgeText: "COMMUTER FLEET", sub: "Moderate Wear · Stable Dynamics" },
    { icon: "⚠️", badge: "badge-amber", badgeText: "INCIPIENT HAZARD", sub: "Passes at 0.50 · Intercepted at 0.193" },
    { icon: "🔴", badge: "badge-red", badgeText: "CRITICAL FAILURE", sub: "High Temp · Thermal Runaway Imminent" }
  ];

  let idx = 0;
  for (const [key, obj] of Object.entries(scenarios)) {
    const info = metaLookup[idx % metaLookup.length];
    const data = obj.data;
    const card = document.createElement("div");
    card.className = "scenario-card" + (idx === 0 ? " active" : "");
    card.id = `card-sc-${idx}`;
    card.onclick = () => selectScenario(key);

    const cycles = data.cycle_count ? Math.round(data.cycle_count).toLocaleString() : "1,315";
    const soh = data.battery_health_percent ? data.battery_health_percent.toFixed(1) + "%" : "85.3%";
    const tmax = data.cell_temperature_max ? data.cell_temperature_max.toFixed(1) + "°C" : "40.2°C";

    card.innerHTML = `
      <div>
        <div class="sc-header">
          <span class="badge ${info.badge}">${info.badgeText}</span>
          <span style="font-size:18px;">${info.icon}</span>
        </div>
        <div class="sc-title">${key.split(":")[1] || key}</div>
        <div class="sc-sub">${info.sub}</div>
        <div class="sc-metrics">
          <div><span>Cycles:</span> <strong>${cycles}</strong></div>
          <div><span>SOH:</span> <strong>${soh}</strong></div>
          <div><span>T_max:</span> <strong>${tmax}</strong></div>
        </div>
      </div>
      <button class="sc-btn">${idx === 0 ? "★ Active Scenario" : "Load Scenario"}</button>
    `;
    container.appendChild(card);
    idx++;
  }
}

function selectScenario(key) {
  STATE.activeScenarioKey = key;
  const scObj = STATE.meta.scenarios[key];
  if (!scObj) return;

  // Update UI active card styling
  const cards = document.querySelectorAll(".scenario-card");
  cards.forEach(c => c.classList.remove("active"));
  let idx = 0;
  for (const k of Object.keys(STATE.meta.scenarios)) {
    if (k === key) {
      const activeCard = document.getElementById(`card-sc-${idx}`);
      if (activeCard) {
        activeCard.classList.add("active");
        activeCard.querySelector(".sc-btn").textContent = "★ Active Scenario";
      }
    } else {
      const otherCard = document.getElementById(`card-sc-${idx}`);
      if (otherCard) otherCard.querySelector(".sc-btn").textContent = "Load Scenario";
    }
    idx++;
  }

  // Load telemetry values into STATE.inputs
  const cfg = STATE.meta.slider_config;
  for (const [feat, conf] of Object.entries(cfg)) {
    let val = scObj.data[feat];
    if (val === null || val === undefined || isNaN(val)) {
      val = conf.default;
    }
    val = Math.max(conf.min, Math.min(conf.max, parseFloat(val)));
    STATE.inputs[feat] = val;
  }

  // Rebuild slider controls in the studio
  renderStudioSliders();
  triggerPrediction();
}

// ==============================================================================
// 3. 12-Feature Interactive Studio
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
  }, 120);
}

// ==============================================================================
// 4. Live Dual-Task Inference
// ==============================================================================
async function triggerPrediction() {
  // Client-side quick check: cell_temperature_max < cell_temperature_avg
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
    updatePredictionUI(data);
    updateAttributionUI();
  } catch (err) {
    console.error("Prediction error:", err);
  }
}

function updatePredictionUI(data) {
  // Clear any existing anomaly error
  clearAnomalyWarning();

  // Task 1: RUL
  const rulNum = data.rul_cycles;
  document.getElementById("rul-val").textContent = Math.round(rulNum).toLocaleString();
  
  // Assumed 350 cycles per fleet year
  const serviceYears = (rulNum / 350.0).toFixed(1);
  document.getElementById("rul-years").textContent = `~${serviceYears} yrs`;

  // Circular progress gauge (max reference 10,000 cycles)
  const circle = document.getElementById("rul-gauge-bar");
  const maxRef = 10000;
  const pct = Math.min(1.0, Math.max(0.0, rulNum / maxRef));
  const circumference = 2 * Math.PI * 42; // ~263.89
  const offset = circumference * (1 - pct);
  circle.style.strokeDashoffset = offset;
  circle.style.stroke = pct > 0.6 ? "var(--blue)" : (pct > 0.3 ? "var(--amber)" : "var(--red)");

  const headline = document.getElementById("rul-headline");
  if (rulNum > 7000) headline.textContent = "High Endurance Reserve";
  else if (rulNum > 4000) headline.textContent = "Moderate Pack Longevity";
  else headline.textContent = "Depleted Service Life";

  // Task 2: Failure Early Warning
  const alertBox = document.getElementById("risk-alert-box");
  alertBox.className = `risk-alert-box ${data.status}`;
  
  const icon = document.getElementById("alert-icon");
  const flagVal = document.getElementById("risk-flag-val");
  if (data.status === "nominal") {
    icon.textContent = "🟢";
    flagVal.textContent = "NOMINAL";
    flagVal.style.color = "var(--green-dark)";
  } else if (data.status === "early_warning") {
    icon.textContent = "⚠️";
    flagVal.textContent = "EARLY WARNING";
    flagVal.style.color = "var(--amber-dark)";
  } else {
    icon.textContent = "🔴";
    flagVal.textContent = "CRITICAL ALERT";
    flagVal.style.color = "var(--red-dark)";
  }

  document.getElementById("alert-title").textContent = data.alert_title;
  document.getElementById("alert-action").textContent = data.alert_action;
  document.getElementById("alert-note").textContent = data.alert_note;

  document.getElementById("risk-prob-val").textContent = `${(data.p_failure * 100).toFixed(2)}%`;
  document.getElementById("risk-tau-val").textContent = STATE.tau.toFixed(3);

  // Warnings / notices
  const warnContainer = document.getElementById("warnings-box");
  warnContainer.innerHTML = "";
  if (data.warnings && data.warnings.length > 0) {
    data.warnings.forEach(w => {
      const item = document.createElement("div");
      item.className = "val-item warning";
      item.innerHTML = `<span>⚠️</span><span>${w}</span>`;
      warnContainer.appendChild(item);
    });
  }
}

function renderAnomalyWarning(msg) {
  const warnContainer = document.getElementById("warnings-box");
  warnContainer.innerHTML = `
    <div class="val-item warning" style="background:var(--red-bg); border-color:var(--red-border); color:var(--red-dark);">
      <span>🛑</span><span>${msg}</span>
    </div>
  `;
}

function clearAnomalyWarning() {
  const warnContainer = document.getElementById("warnings-box");
  if (warnContainer.querySelector("div[style*='red-bg']")) {
    warnContainer.innerHTML = "";
  }
}

// ==============================================================================
// 5. Decision Thresholds & Curves
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
    document.getElementById("recall-badge").textContent = `${recPct}% Recall`;

    // Render SVG Charts in Tab 3
    renderPRCurve(data.curve, tau, data.metrics);
    renderROCCurve(data.metrics);
  } catch (err) {
    console.error("Threshold fetch error:", err);
  }
}

// ==============================================================================
// 6. Pure SVG Chart Renderers (100% Offline, Zero CDNs)
// ==============================================================================
function renderPRCurve(curve, currentTau, metrics) {
  const svg = document.getElementById("svg-pr-curve");
  if (!svg || !curve || !curve.recalls || curve.recalls.length === 0) return;

  const w = 400, h = 240;
  const padL = 45, padR = 25, padT = 20, padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  // Build SVG path
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

  // Baseline floor (positives / total = 277 / 4000 = 0.0693)
  const floorY = padT + (1 - 0.0693) * plotH;

  svg.innerHTML = `
    <!-- Grid & Axes -->
    <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}" stroke="#E2E8F0" stroke-width="1.5" />
    <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" stroke="#E2E8F0" stroke-width="1.5" />
    
    <!-- Y Labels -->
    <text x="${padL - 8}" y="${padT + 5}" text-anchor="end" font-size="10" fill="#64748B">1.0</text>
    <text x="${padL - 8}" y="${padT + plotH * 0.5 + 4}" text-anchor="end" font-size="10" fill="#64748B">0.5</text>
    <text x="${padL - 8}" y="${h - padB}" text-anchor="end" font-size="10" fill="#64748B">0.0</text>
    <text x="14" y="${h / 2}" font-size="10" fill="#475569" transform="rotate(-90 14 ${h / 2})" text-anchor="middle" font-weight="600">Precision</text>

    <!-- X Labels -->
    <text x="${padL}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#64748B">0.0</text>
    <text x="${padL + plotW * 0.5}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#64748B">0.5</text>
    <text x="${w - padR}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#64748B">1.0</text>
    <text x="${padL + plotW / 2}" y="${h - 8}" text-anchor="middle" font-size="10" fill="#475569" font-weight="600">Recall (Sensitivity)</text>

    <!-- Naive Floor -->
    <line x1="${padL}" y1="${floorY}" x2="${w - padR}" y2="${floorY}" stroke="#CBD5E1" stroke-dasharray="4" stroke-width="1" />
    <text x="${w - padR - 5}" y="${floorY - 4}" text-anchor="end" font-size="9" fill="#94A3B8">Naive Floor (6.9%)</text>

    <!-- PR Curve Path -->
    <path d="${pathD}" fill="none" stroke="#2563EB" stroke-width="2.5" />

    <!-- Active Cutoff Marker -->
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

  // Approximate parametric ROC path for champion models (AUC ~ 0.983)
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

  const markerX = padL + (139 / 3723) * plotW; // FPR = FP / (FP + TN) = 139 / 3723 = 0.0373
  const markerY = padT + (1 - (250 / 277)) * plotH; // TPR = TP / P = 250 / 277 = 0.9025

  svg.innerHTML = `
    <!-- Grid & Axes -->
    <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${h - padB}" stroke="#E2E8F0" stroke-width="1.5" />
    <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${h - padB}" stroke="#E2E8F0" stroke-width="1.5" />

    <!-- Chance Diagonal -->
    <line x1="${padL}" y1="${h - padB}" x2="${w - padR}" y2="${padT}" stroke="#CBD5E1" stroke-dasharray="4" stroke-width="1.5" />

    <!-- Y Labels -->
    <text x="${padL - 8}" y="${padT + 5}" text-anchor="end" font-size="10" fill="#64748B">1.0</text>
    <text x="${padL - 8}" y="${padT + plotH * 0.5 + 4}" text-anchor="end" font-size="10" fill="#64748B">0.5</text>
    <text x="${padL - 8}" y="${h - padB}" text-anchor="end" font-size="10" fill="#64748B">0.0</text>
    <text x="14" y="${h / 2}" font-size="10" fill="#475569" transform="rotate(-90 14 ${h / 2})" text-anchor="middle" font-weight="600">True Positive Rate</text>

    <!-- X Labels -->
    <text x="${padL}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#64748B">0.0</text>
    <text x="${padL + plotW * 0.5}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#64748B">0.5</text>
    <text x="${w - padR}" y="${h - padB + 16}" text-anchor="middle" font-size="10" fill="#64748B">1.0</text>
    <text x="${padL + plotW / 2}" y="${h - 8}" text-anchor="middle" font-size="10" fill="#475569" font-weight="600">False Positive Rate</text>

    <!-- Curve Path -->
    <path d="${rocD}" fill="none" stroke="#10B981" stroke-width="2.5" />

    <!-- Safety Threshold Operating Marker -->
    <circle cx="${markerX.toFixed(1)}" cy="${markerY.toFixed(1)}" r="6" fill="#10B981" stroke="#FFF" stroke-width="2" />
    <rect x="${markerX + 8}" y="${markerY - 20}" width="95" height="24" rx="4" fill="#0F172A" />
    <text x="${markerX + 14}" y="${markerY - 4}" font-size="10" fill="#FFF" font-weight="700">TPR 90.3% · FPR 3.7%</text>
  `;
}

// ==============================================================================
// 7. Feature Attribution & Engineering
// ==============================================================================
function updateAttributionUI() {
  const container = document.getElementById("attribution-bars");
  if (!container) return;

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
      <div class="bar-label" title="${f.name}">${f.name}</div>
      <div class="bar-track">
        <div class="bar-fill ${f.impact}" style="width: ${f.score.toFixed(1)}%;"></div>
      </div>
      <div class="bar-val">${f.val.toFixed(2)} ${f.unit}</div>
    `;
    container.appendChild(row);
  });
}

function renderEngineeredFormulas() {
  const container = document.getElementById("engineered-list");
  if (!container || !STATE.meta.engineered_formulas) return;

  container.innerHTML = "";
  for (const [name, formula] of Object.entries(STATE.meta.engineered_formulas)) {
    const box = document.createElement("div");
    box.style.background = "var(--surface-alt)";
    box.style.padding = "8px 12px";
    box.style.borderRadius = "var(--radius-sm)";
    box.innerHTML = `
      <strong style="color:var(--blue);">${name}:</strong><br>
      <code style="font-size:10px; color:var(--text-sub);">${formula}</code>
    `;
    container.appendChild(box);
  }
}

// ==============================================================================
// 8. Fleet Simulator & Master Leaderboards
// ==============================================================================
async function loadLeaderboards() {
  try {
    const res = await fetch("/api/leaderboard");
    if (!res.ok) return;
    const data = await res.json();

    // Task 1 Leaderboard
    const t1Body = document.getElementById("leaderboard-t1-body");
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

    // Task 2 Leaderboard
    const t2Body = document.getElementById("leaderboard-t2-body");
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
  } catch (err) {
    console.error("Leaderboard fetch error:", err);
  }
}

function updateFleetSimulation() {
  const nVehicles = parseInt(document.getElementById("sim-fleet-size").value);
  const techHours = parseInt(document.getElementById("sim-hours").value);

  document.getElementById("sim-fleet-size-val").textContent = nVehicles.toLocaleString();
  document.getElementById("sim-hours-val").textContent = `${techHours.toLocaleString()} hrs`;

  // Failure prevalence 6.925%
  const totalFailures = Math.round(nVehicles * (277.0 / 4000.0));
  const recall = STATE.curvesData?.metrics?.recall || 0.9025;
  const fpRate = (STATE.curvesData?.metrics?.fp || 139) / 3723.0;

  const intercepted = Math.round(totalFailures * recall);
  const falseAlarms = Math.round((nVehicles - totalFailures) * fpRate);
  const totalInspections = intercepted + falseAlarms;

  document.getElementById("sim-intercepted").textContent = intercepted.toLocaleString();
  document.getElementById("sim-workload").textContent = `${totalInspections.toLocaleString()} teardowns`;
}

// ==============================================================================
// 9. Batch CSV Triage
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
      (&tau; = ${data.tau.toFixed(3)})
    `;

    tbody.innerHTML = "";
    data.ranked_records.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>#${r.row_index}</td>
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
// 10. Health Passport Export (Markdown/Text Blob)
// ==============================================================================
function exportHealthPassport() {
  if (!STATE.prediction) return;
  const p = STATE.prediction;
  const d = new Date().toISOString();

  const md = `# EV Battery Health & Prognostics Diagnostic Passport
Generated: ${d}
System: SLIIT IT3051 Mini Project 2026 | Group Necrons
Software: Dual-Task Prognostics Prototype v1.0.0-production

---
## Executive Summary
- **Active Scenario:** ${STATE.activeScenarioKey || "Custom Studio Configuration"}
- **Task 1 RUL Cycles:** ${p.rul_cycles.toLocaleString()} cycles (~${(p.rul_cycles / 350.0).toFixed(1)} estimated operating years)
- **Task 2 Failure Probability:** ${(p.p_failure * 100).toFixed(2)}%
- **Active Operating Cutoff (τ):** ${STATE.tau.toFixed(3)}
- **Triage Status:** ${p.status_label}
- **Recommended Action:** ${p.alert_action}

---
## Telemetry Inputs Snapshot
| Telemetry Parameter | Value | Reference / Boundary |
|---|---|---|
| State of Health (SOH) | ${STATE.inputs['battery_health_percent']?.toFixed(1) || 'N/A'}% | EOL: 70.0% – 80.0% |
| Capacity Fade Loss | ${STATE.inputs['capacity_loss_percent']?.toFixed(1) || 'N/A'}% | Collinear with SOH (|r|=1.0) |
| Internal Cell Resistance | ${STATE.inputs['internal_resistance']?.toFixed(3) || 'N/A'} Ω | Observed Median: 0.3808 Ω |
| Completed Full Cycles | ${Math.round(STATE.inputs['cycle_count'] || 0).toLocaleString()} | Mean: 1,770 cycles |
| Max Cell Hotspot Temp | ${STATE.inputs['cell_temperature_max']?.toFixed(1) || 'N/A'} °C | Elevated Boundary: > 52°C |
| Mean Pack Temperature | ${STATE.inputs['cell_temperature_avg']?.toFixed(1) || 'N/A'} °C | Normal: 15°C – 35°C |
| Thermal Runaway Risk | ${STATE.inputs['thermal_runaway_risk']?.toFixed(1) || 'N/A'} /100 | High Hazard: > 50 |
| Average Charging Power | ${STATE.inputs['average_charge_power_kw']?.toFixed(1) || 'N/A'} kW | AC: 7–22 kW, DC: 50–150 kW |

---
## Model Attribution & Domain Features
- **Task 1 Model:** ${p.model_names.task1}
- **Task 2 Model:** ${p.model_names.task2}
- **Temperature Spread:** ${p.engineered_features.temperature_spread.toFixed(2)} °C
- **Efficiency Gap:** ${p.engineered_features.efficiency_gap.toFixed(2)} %
- **C-Rate Proxy:** ${p.engineered_features.c_rate_proxy.toFixed(4)}
- **Resistance per 1k Cycles:** ${p.engineered_features.resistance_per_1000_cycles.toFixed(4)} Ω/1k-cyc

---
*Notice: Decision-support prototype for research and evaluation purposes only. Not a safety certification.*
`;

  const blob = new Blob([md], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `EV_Battery_Passport_${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(url);
}

// ==============================================================================
// 11. Event Listeners Setup
// ==============================================================================
function setupEventListeners() {
  // Engine select
  document.getElementById("engine-select").addEventListener("change", (e) => {
    STATE.engine = e.target.value;
    loadThresholdData(STATE.tau);
    triggerPrediction();
  });

  // Threshold slider
  const tauSlider = document.getElementById("tau-slider");
  tauSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    STATE.tau = val;
    document.getElementById("slider-tau-indicator").innerHTML = `<strong>Active &tau; = ${val.toFixed(3)}</strong>`;
    
    // Clear mode pill active state if slider custom
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
      tauSlider.value = tau;
      document.getElementById("slider-tau-indicator").innerHTML = `<strong>Active &tau; = ${tau.toFixed(3)}</strong>`;
      loadThresholdData(tau);
      triggerPrediction();
    });
  });

  // Reset studio button
  document.getElementById("btn-reset-studio").addEventListener("click", () => {
    if (STATE.activeScenarioKey) selectScenario(STATE.activeScenarioKey);
  });

  // Export passport button
  document.getElementById("btn-export-passport").addEventListener("click", exportHealthPassport);

  // Tabs navigation
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add("active");

      if (btn.dataset.tab === "tab-curves" && STATE.curvesData) {
        renderPRCurve(STATE.curvesData.curve, STATE.tau, STATE.curvesData.metrics);
        renderROCCurve(STATE.curvesData.metrics);
      }
    });
  });

  // Fleet simulator sliders
  document.getElementById("sim-fleet-size").addEventListener("input", updateFleetSimulation);
  document.getElementById("sim-hours").addEventListener("input", updateFleetSimulation);

  // Batch CSV file dropzone
  const dropzone = document.getElementById("batch-dropzone");
  const fileInput = document.getElementById("batch-file-input");

  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleBatchUpload(e.target.files[0]);
  });

  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.style.borderColor = "var(--blue)"; });
  dropzone.addEventListener("dragleave", () => { dropzone.style.borderColor = "var(--blue-border)"; });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "var(--blue-border)";
    if (e.dataTransfer.files.length > 0) handleBatchUpload(e.dataTransfer.files[0]);
  });

  // Batch download button
  document.getElementById("btn-download-ranked-csv").addEventListener("click", () => {
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
  document.getElementById("btn-run-selftest").addEventListener("click", async () => {
    const box = document.getElementById("selftest-output");
    box.textContent = "Running /api/selftest diagnostic...";
    try {
      const res = await fetch("/api/selftest");
      const data = await res.json();
      box.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      box.textContent = `Diagnostic failed: ${err.message}`;
    }
  });
}
