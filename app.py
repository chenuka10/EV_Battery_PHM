import os
import json
import html as _html
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="EV Battery PHM System | Dual-Task Diagnostics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


def compact(s: str) -> str:
    """Collapse multi-line HTML into one line so Markdown never treats
    indented HTML as a code block."""
    return "".join(line.strip() for line in s.strip().splitlines())


# ==============================================================================
# Design System (White + Electric Blue EV Technology Theme)
# ==============================================================================
THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
:root {
    --bg: #F7FAFC;
    --card: #FFFFFF;
    --blue: #1677FF;
    --navy: #0B2A4A;
    --text: #102A43;
    --muted: #60758A;
    --surface: #EAF3FF;
    --sidebar: #F0F6FF;
    --border: #D9E7F5;
    --ok: #16A673;
    --warn: #E6A700;
    --crit: #E5484D;
    --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea, .stApp [data-baseweb="select"] div {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
}
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"] { display: none; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }

/* ---------- Header ---------- */
.main-header { display: flex; align-items: center; gap: 0.7rem; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: var(--navy); line-height: 1.15; margin: 0; }
.main-header svg { flex: none; }
.sub-header { font-size: 0.98rem; color: var(--muted); margin: 0.45rem 0 1.1rem 0; line-height: 1.5; max-width: 62rem; }
.hdr-rule { height: 1px; background: var(--border); position: relative; margin-bottom: 1.6rem; }
.hdr-rule::before { content: ""; position: absolute; left: 0; top: -1px; width: 72px; height: 3px; border-radius: 2px; background: var(--blue); }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top: 1.3rem; }
[data-testid="stSidebar"] label p { font-size: 0.78rem; font-weight: 600; color: var(--navy); }
[data-testid="stSidebar"] [data-testid="stSlider"] { padding-bottom: 0.15rem; }
.sb-brand { display: flex; align-items: center; gap: 0.75rem; padding-bottom: 0.9rem; border-bottom: 1px solid var(--border); }
.sb-brand-icon { width: 42px; height: 42px; border-radius: 10px; background: #fff; border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; }
.sb-brand-title { font-size: 1.02rem; font-weight: 700; color: var(--navy); letter-spacing: -0.01em; line-height: 1.2; }
.sb-brand-sub { font-size: 0.72rem; color: var(--muted); margin-top: 0.15rem; letter-spacing: 0.02em; }
.sb-section { display: flex; align-items: center; gap: 0.5rem; margin: 1.35rem 0 0.6rem 0; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--navy); }
.sb-section .n { font-family: var(--mono); color: var(--blue); }
.sb-section::after { content: ""; flex: 1; height: 1px; background: var(--border); }
.sb-sub { font-size: 0.72rem; color: var(--muted); font-weight: 600; letter-spacing: 0.04em; margin: 0.1rem 0 0.4rem 0; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #fff; border: 1px solid var(--border); border-radius: 8px; min-height: 40px; }
[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within { border-color: var(--blue); box-shadow: 0 0 0 3px rgba(22,119,255,0.14); }
[data-testid="stSidebar"] [data-testid="stAlert"] { background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--blue); border-radius: 8px; color: var(--navy); }
[data-testid="stSidebar"] [data-testid="stAlert"] p { font-size: 0.8rem; line-height: 1.5; color: var(--navy); }
[data-testid="stSidebar"] [data-testid="stAlert"] svg { color: var(--blue); fill: var(--blue); }

/* Sliders */
div[data-baseweb="slider"] [role="slider"] { background-color: var(--blue) !important; border-color: #fff !important; box-shadow: 0 0 0 3px rgba(22,119,255,0.18) !important; }
[data-testid="stSliderThumbValue"] { color: var(--blue) !important; font-weight: 600; font-family: var(--mono) !important; font-size: 0.75rem; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: var(--muted); font-size: 0.68rem; }

/* Radio as selectable rows */
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.4rem; }
[data-testid="stSidebar"] div[role="radiogroup"] label { background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 0.55rem 0.7rem; width: 100%; transition: border-color 0.15s, background 0.15s; }
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { border-color: var(--blue); }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background: var(--surface); border-color: var(--blue); }
[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 0.8rem; font-weight: 600; color: var(--navy); }

/* Buttons */
.stButton > button { width: 100%; background: #fff; color: var(--navy); border: 1px solid var(--border); border-radius: 8px; font-weight: 600; font-size: 0.78rem; padding: 0.4rem 0.5rem; transition: all 0.15s; }
.stButton > button:hover { border-color: var(--blue); color: var(--blue); background: var(--surface); }
.stButton > button:focus:not(:active) { border-color: var(--blue); color: var(--blue); box-shadow: 0 0 0 3px rgba(22,119,255,0.14); }
.stButton > button:active { background: var(--blue); color: #fff; border-color: var(--blue); }

/* ---------- Threshold calibration panel ---------- */
.tau-panel { background: #fff; border: 1px solid var(--border); border-radius: 10px; padding: 0.95rem 1rem 0.9rem 1rem; margin-bottom: 0.7rem; position: relative; overflow: hidden; }
.tau-panel::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--blue); }
.tau-top { display: flex; justify-content: space-between; align-items: center; }
.tau-label { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.12em; color: var(--muted); text-transform: uppercase; }
.tau-state { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.1em; color: var(--blue); background: var(--surface); border-radius: 4px; padding: 0.15rem 0.45rem; }
.tau-value { font-family: var(--mono); font-size: 1.9rem; font-weight: 600; color: var(--navy); letter-spacing: -0.02em; margin: 0.35rem 0 0.7rem 0; }
.tau-value .tau-sym { color: var(--blue); }
.tau-refs { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
.tau-ref { border: 1px solid var(--border); border-radius: 8px; padding: 0.4rem 0.6rem; background: #FAFCFF; }
.tau-ref span { display: block; font-size: 0.6rem; letter-spacing: 0.1em; font-weight: 600; color: var(--muted); }
.tau-ref b { font-family: var(--mono); font-size: 0.95rem; color: var(--navy); font-weight: 600; }
.tau-ref.active { border-color: var(--blue); background: var(--surface); }
.tau-ref.active span { color: var(--blue); }

/* ---------- Metric cards ---------- */
.metric-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.1rem 1.2rem; min-height: 138px; box-shadow: 0 1px 2px rgba(11,42,74,0.04); display: flex; flex-direction: column; justify-content: space-between; }
.metric-title { font-size: 0.68rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.09em; line-height: 1.35; }
.metric-value { font-size: 2.1rem; font-weight: 700; color: var(--navy); letter-spacing: -0.03em; line-height: 1.1; margin: 0.55rem 0 0.4rem 0; font-variant-numeric: tabular-nums; display: flex; align-items: center; min-height: 2.3rem; }
.metric-value .unit { font-size: 0.85rem; font-weight: 500; color: var(--muted); letter-spacing: 0; margin-left: 0.4rem; }
.metric-detail { font-size: 0.74rem; color: var(--blue); font-weight: 500; line-height: 1.4; }

/* ---------- Status badges ---------- */
.status-badge { display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.28rem 0.7rem; border-radius: 6px; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.08em; white-space: nowrap; }
.status-badge i { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.sb-safe { background: #E8F7F1; color: #0B5E43; }
.sb-safe i { background: var(--ok); }
.sb-warn { background: #FFF5D6; color: #7A5600; }
.sb-warn i { background: var(--warn); }
.sb-crit { background: #FDECEC; color: #9B1C20; }
.sb-crit i { background: var(--crit); }

/* ---------- Diagnostic alert ---------- */
.alert { border: 1px solid var(--border); border-left: 4px solid var(--c); border-radius: 10px; padding: 1.15rem 1.4rem 1.1rem 1.4rem; margin: 1.3rem 0 1.6rem 0; background: var(--bg-t); }
.alert-safe { --c: var(--ok); --bg-t: #F6FBF9; }
.alert-warn { --c: var(--warn); --bg-t: #FFFCF2; }
.alert-crit { --c: var(--crit); --bg-t: #FFF7F7; }
.alert-head { display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 0.95rem; }
.alert-title { font-size: 1.02rem; font-weight: 700; color: var(--navy); letter-spacing: -0.01em; }
.alert-grid { display: grid; grid-template-columns: 1fr 1fr 2.2fr; gap: 1.4rem; padding: 0.85rem 0; border-top: 1px solid rgba(11,42,74,0.07); border-bottom: 1px solid rgba(11,42,74,0.07); }
.alert-k { font-size: 0.64rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 0.3rem; }
.alert-v { font-size: 1.5rem; font-weight: 700; color: var(--navy); font-variant-numeric: tabular-nums; letter-spacing: -0.02em; line-height: 1.2; }
.alert-v .sym { color: var(--blue); }
.alert-v.act { font-size: 0.9rem; font-weight: 600; line-height: 1.45; letter-spacing: 0; }
.alert-note { margin-top: 0.8rem; font-size: 0.84rem; color: var(--muted); line-height: 1.55; }
.alert-note b { color: var(--navy); font-weight: 600; }
@media (max-width: 900px) { .alert-grid { grid-template-columns: 1fr; gap: 0.8rem; } }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 0.3rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { height: auto; padding: 0.75rem 1.2rem; background: transparent; border-radius: 8px 8px 0 0; }
.stTabs [data-baseweb="tab"] p { font-size: 0.74rem; font-weight: 600; letter-spacing: 0.1em; color: var(--muted); }
.stTabs [data-baseweb="tab"]:hover { background: var(--surface); }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background: var(--surface); }
.stTabs [data-baseweb="tab"][aria-selected="true"] p { color: var(--blue); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--blue); height: 2px; }
.stTabs [data-baseweb="tab-border"] { background-color: transparent; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.4rem; }

/* ---------- Panels ---------- */
.panel { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.3rem 1.5rem; min-height: 340px; box-shadow: 0 1px 2px rgba(11,42,74,0.04); }
.panel-eyebrow { font-family: var(--mono); font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em; color: var(--blue); text-transform: uppercase; }
.panel-title { font-size: 1.12rem; font-weight: 700; color: var(--navy); letter-spacing: -0.01em; margin: 0.2rem 0 0.5rem 0; }
.panel-sub { font-size: 0.84rem; color: var(--muted); line-height: 1.5; margin-bottom: 1.1rem; }
.panel-sub b { color: var(--navy); font-weight: 600; }
.kv { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; padding: 0.6rem 0; border-top: 1px solid #EDF3FA; font-size: 0.84rem; }
.kv span { color: var(--muted); }
.kv span small { color: #8CA0B3; font-size: 0.74rem; }
.kv b { color: var(--navy); font-family: var(--mono); font-weight: 600; font-size: 0.82rem; text-align: right; }
.kv.wrap { display: block; line-height: 1.55; }
.kv.wrap b { font-family: inherit; }

.section-title { font-size: 1.15rem; font-weight: 700; color: var(--navy); letter-spacing: -0.01em; margin: 0 0 0.25rem 0; }
.section-sub { font-size: 0.86rem; color: var(--muted); margin-bottom: 1.1rem; line-height: 1.5; }
.section-sub b { color: var(--navy); font-weight: 600; }
.table-title { display: flex; align-items: center; gap: 0.6rem; font-size: 0.95rem; font-weight: 700; color: var(--navy); margin: 1.6rem 0 0.7rem 0; }
.table-title::before { content: ""; width: 3px; height: 1.05rem; background: var(--blue); border-radius: 2px; }

/* ---------- Progress bars ---------- */
.pbar { margin: 0.2rem 0 1.15rem 0; }
.pbar-head { display: flex; justify-content: space-between; align-items: baseline; font-size: 0.8rem; margin-bottom: 0.45rem; }
.pbar-head span { color: var(--muted); font-weight: 500; }
.pbar-head b { color: var(--navy); font-family: var(--mono); font-weight: 600; }
.pbar-track { position: relative; height: 10px; background: var(--surface); border-radius: 5px; border: 1px solid var(--border); }
.pbar-fill { height: 100%; background: var(--blue); border-radius: 5px; }
.pbar-tick { position: absolute; top: -5px; width: 2px; height: 18px; background: var(--navy); border-radius: 1px; }
.pbar-legend { display: flex; align-items: center; gap: 0.4rem; margin-top: 0.45rem; font-size: 0.7rem; color: var(--muted); }
.pbar-legend i { width: 2px; height: 10px; background: var(--navy); display: inline-block; }

/* ---------- Native st.metric (Degradation tab) ---------- */
[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.2rem; box-shadow: 0 1px 2px rgba(11,42,74,0.04); }
[data-testid="stMetricLabel"] p { font-size: 0.68rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.09em; color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--navy); font-weight: 700; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }

/* ---------- Data tables ---------- */
.tbl-wrap { background: var(--card); border: 1px solid var(--border); border-radius: 10px; overflow-x: auto; box-shadow: 0 1px 2px rgba(11,42,74,0.04); }
table.data-tbl { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
table.data-tbl thead th { background: #F3F8FE; color: var(--navy); text-align: left; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; padding: 0.8rem 1rem; border-bottom: 1px solid var(--border); white-space: nowrap; }
table.data-tbl tbody td { padding: 0.7rem 1rem; color: var(--text); border-bottom: 1px solid #EDF3FA; white-space: nowrap; }
table.data-tbl tbody tr:last-child td { border-bottom: none; }
table.data-tbl tbody tr:nth-child(even) td { background: #FAFCFE; }
table.data-tbl tbody tr:hover td { background: #F3F8FE; }
table.data-tbl .num { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 0.8rem; }
table.data-tbl .mdl { font-weight: 600; color: var(--navy); }
table.data-tbl .mono { font-family: var(--mono); font-size: 0.78rem; color: var(--navy); }
table.data-tbl tbody tr.champ td { background: var(--surface) !important; }
table.data-tbl tbody tr.champ td:first-child { box-shadow: inset 3px 0 0 var(--blue); }
.chip { display: inline-block; font-size: 0.72rem; font-weight: 600; color: var(--muted); }
.chip.champ { color: var(--blue); background: #fff; border: 1px solid #BBD7FB; border-radius: 5px; padding: 0.12rem 0.5rem; }

/* ---------- Footer ---------- */
.app-footer { margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.74rem; color: var(--muted); letter-spacing: 0.02em; }
.app-footer b { color: var(--blue); font-weight: 600; }
"""

st.markdown("<style>" + compact(THEME_CSS) + "</style>", unsafe_allow_html=True)

# ==============================================================================
# UI Helper Functions (presentation only — no model logic)
# ==============================================================================
BOLT_SVG = (
    '<svg width="30" height="30" viewBox="0 0 24 24" fill="#1677FF" aria-hidden="true">'
    '<path d="M13 2 4 14h6l-1 8 9-12h-6l1-8z"/></svg>'
)
BATTERY_SVG = (
    '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#1677FF" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<rect x="2" y="7" width="17" height="10" rx="2"/><path d="M22 11v2"/>'
    '<path d="m11.5 9.2-2.6 3.3h3.2l-2.6 3.3"/></svg>'
)


def sb_section(num: str, title: str):
    st.sidebar.markdown(
        compact(f'<div class="sb-section"><span class="n">{num}</span>{title}</div>'),
        unsafe_allow_html=True,
    )


def progress_bar(frac: float, label: str, value: str, marker: float = None, marker_label: str = ""):
    frac = min(max(frac, 0.0), 1.0)
    tick = ""
    legend = ""
    if marker is not None:
        tick = f'<div class="pbar-tick" style="left:calc({min(max(marker, 0.0), 1.0) * 100:.1f}% - 1px)"></div>'
        legend = f'<div class="pbar-legend"><i></i>{marker_label}</div>'
    return compact(f"""
    <div class="pbar">
        <div class="pbar-head"><span>{label}</span><b>{value}</b></div>
        <div class="pbar-track"><div class="pbar-fill" style="width:{frac * 100:.1f}%"></div>{tick}</div>
        {legend}
    </div>
    """)


def fmt_value(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _html.escape(str(v))
    if f.is_integer():
        return f"{int(f):,}"
    return f"{f:.4f}".rstrip("0").rstrip(".")


def render_table(df: pd.DataFrame, num_cols=(), decimals=None, model_col="Model", status_col="Status"):
    """Render a DataFrame as a styled HTML table. Rows whose status contains
    'Champion' get a subtle light-blue highlight."""
    decimals = decimals or {}
    head = "".join(
        f'<th class="{"num" if c in num_cols else ""}">{_html.escape(str(c))}</th>' for c in df.columns
    )
    rows = []
    for _, row in df.iterrows():
        is_champ = status_col in df.columns and "Champion" in str(row[status_col])
        cells = []
        for c in df.columns:
            v = row[c]
            if c == model_col:
                cells.append(f'<td class="mdl">{_html.escape(str(v))}</td>')
            elif c == status_col:
                cls = "chip champ" if is_champ else "chip"
                cells.append(f'<td><span class="{cls}">{_html.escape(str(v))}</span></td>')
            elif c in num_cols:
                txt = f"{v:.{decimals[c]}f}" if c in decimals and isinstance(v, (int, float, np.floating)) else _html.escape(str(v))
                cells.append(f'<td class="num">{txt}</td>')
            else:
                cells.append(f"<td>{_html.escape(str(v))}</td>")
        rows.append(f'<tr class="{"champ" if is_champ else ""}">{"".join(cells)}</tr>')
    return compact(
        f'<div class="tbl-wrap"><table class="data-tbl"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


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
st.sidebar.markdown(
    compact(f"""
    <div class="sb-brand">
        <div class="sb-brand-icon">{BATTERY_SVG}</div>
        <div>
            <div class="sb-brand-title">BMS Telemetry Control</div>
            <div class="sb-brand-sub">Group: Necrons &nbsp;·&nbsp; Dual-Task System</div>
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)

# 1. Preset Scenarios vs Custom
sb_section("01", "Vehicle Input Mode")
mode = st.sidebar.selectbox(
    "Vehicle Input Mode",
    ["Select a Preset Telemetry Scenario", "Manual Telemetry Tuning (Custom Sliders)"],
    label_visibility="collapsed"
)

sb_section("02", "Telemetry Controls")
if mode == "Select a Preset Telemetry Scenario":
    selected_scenario_name = st.sidebar.selectbox(
        "Choose Real Telemetry Scenario:",
        list(scenarios.keys())
    )
    current_data = scenarios[selected_scenario_name]['data'].copy()
    st.sidebar.info(scenarios[selected_scenario_name]['description'])
else:
    current_data = {}
    st.sidebar.markdown('<div class="sb-sub">PRIMARY DIAGNOSTIC SLIDERS</div>', unsafe_allow_html=True)
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

# 2. Decision Threshold Control (Safety Threshold Calibration)
sb_section("03", "Safety Threshold Calibration")
tau_panel = st.sidebar.empty()  # filled once the final tau value is known
tau_slider = st.sidebar.slider(
    "Critical Failure Cutoff (τ)",
    min_value=0.05,
    max_value=0.95,
    value=0.193,  # The Calibrated Safety Threshold
    step=0.01,
    help="Default ML threshold is 0.50. The calibrated safety threshold is 0.193, which achieved 90.25% recall on the evaluation set."
)

# Benchmark buttons
col_b1, col_b2 = st.sidebar.columns(2)
if col_b1.button("Default (0.50)"):
    tau_slider = 0.50
if col_b2.button("Safety (0.193)"):
    tau_slider = 0.193

# Threshold calibration panel (reflects the final tau value)
if abs(tau_slider - 0.193) < 1e-9:
    tau_state = "CALIBRATED"
elif abs(tau_slider - 0.50) < 1e-9:
    tau_state = "DEFAULT"
else:
    tau_state = "CUSTOM"

tau_panel.markdown(
    compact(f"""
    <div class="tau-panel">
        <div class="tau-top">
            <div class="tau-label">Safety Threshold</div>
            <div class="tau-state">{tau_state}</div>
        </div>
        <div class="tau-value"><span class="tau-sym">τ</span> = {tau_slider:.3f}</div>
        <div class="tau-refs">
            <div class="tau-ref {'active' if tau_state == 'DEFAULT' else ''}"><span>DEFAULT</span><b>0.50</b></div>
            <div class="tau-ref {'active' if tau_state == 'CALIBRATED' else ''}"><span>SAFETY</span><b>0.193</b></div>
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)

# 3. Model Engine Toggle
sb_section("04", "Diagnostic Engine")
engine_choice = st.sidebar.radio(
    "Inference Deployment Tier:",
    ["Edge BMS (Calibrated Logistic Regression)", "Cloud Fleet Analytics (Tuned XGBoost)"]
)

# ==============================================================================
# Main Dashboard Layout
# ==============================================================================
st.markdown(
    compact(f'<div class="main-header">{BOLT_SVG}<span>EV Battery Prognostics &amp; Health Management (PHM)</span></div>'),
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Dual-Task Edge Prognostics: Continuous RUL Cycles Regression &amp; Calibrated Failure Risk Early Warning</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="hdr-rule"></div>', unsafe_allow_html=True)

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

# Display-only derived values
cycle_fraction = min(max(pred_rul / 4500.0, 0.0), 1.0)   # nominal 4,500 cycle lifespan
est_years = pred_rul / 350.0                              # ~350 cycles per commuter year

if not is_flagged:
    status_key, status_label = "safe", "NOMINAL"
elif prob_failure < 0.50 and is_flagged:
    status_key, status_label = "warn", "EARLY WARNING"
else:
    status_key, status_label = "crit", "CRITICAL"

badge_html = f'<span class="status-badge sb-{status_key}"><i></i>{status_label}</span>'

# ==============================================================================
# Top Metric Overview Cards
# ==============================================================================
def metric_card(title, value_html, detail_html):
    return compact(f"""
    <div class="metric-card">
        <div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value_html}</div>
        </div>
        <div class="metric-detail">{detail_html}</div>
    </div>
    """)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(metric_card(
        "Predicted Remaining Life (RUL)",
        f'{pred_rul:,.0f}<span class="unit">cycles</span>',
        f"≈ {est_years:.1f} yr horizon at 350 cycles/yr"
    ), unsafe_allow_html=True)

with m2:
    st.markdown(metric_card(
        "Failure Probability P(Failure)",
        f"{prob_failure:.1%}",
        _html.escape(active_model_name)
    ), unsafe_allow_html=True)

with m3:
    st.markdown(metric_card(
        "Active Threshold (τ)",
        f"{tau_slider:.3f}",
        "Default 0.50 &nbsp;·&nbsp; Safety 0.193"
    ), unsafe_allow_html=True)

with m4:
    st.markdown(metric_card(
        "Operational Safety Status",
        badge_html,
        f"P = {prob_failure:.1%} {'≥' if is_flagged else '&lt;'} τ = {tau_slider:.3f}"
    ), unsafe_allow_html=True)

# ==============================================================================
# Operational Diagnostic Alert Banner
# ==============================================================================
if status_key == "safe":
    alert_title = "BMS Telemetry Status: Normal Operation"
    alert_action = "No elevated-risk flag at the current threshold. Continue routine telemetry monitoring."
    alert_note = (
        f"Failure probability (<b>{prob_failure:.1%}</b>) sits below the safety cutoff "
        f"(τ = {tau_slider:.3f}). No elevated-risk flag was raised for this telemetry snapshot."
    )
elif status_key == "warn":
    alert_title = "Early Warning: Flagged by the Safety Cutoff"
    alert_action = "Recommended diagnostic action: inspect battery pack telemetry. Supports earlier inspection."
    alert_note = (
        f"The default machine-learning cutoff (0.50) would <b>not</b> flag this battery because the probability "
        f"is below 50%. The calibrated safety threshold (τ = {tau_slider:.3f}) flags it for early inspection."
    )
else:
    alert_title = "Critical: Failure Probability Above Both Thresholds"
    alert_action = "Recommended diagnostic action: inspect battery pack telemetry; review pack cooling and fast-charging power limits."
    alert_note = (
        f"Failure probability (<b>{prob_failure:.1%}</b>) exceeds both the safety cutoff (τ = {tau_slider:.3f}) "
        f"and the default 0.50 cutoff, indicating a pattern consistent with severe degradation or a thermal anomaly."
    )

st.markdown(compact(f"""
<div class="alert alert-{status_key}">
    <div class="alert-head">{badge_html}<span class="alert-title">{alert_title}</span></div>
    <div class="alert-grid">
        <div><div class="alert-k">Failure Probability</div><div class="alert-v">{prob_failure:.1%}</div></div>
        <div><div class="alert-k">Active Threshold</div><div class="alert-v"><span class="sym">τ</span> = {tau_slider:.3f}</div></div>
        <div><div class="alert-k">Recommended Action</div><div class="alert-v act">{alert_action}</div></div>
    </div>
    <div class="alert-note">{alert_note}</div>
</div>
"""), unsafe_allow_html=True)

# ==============================================================================
# Dual-Task Deep Dive Tabs
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "01 · LIVE DIAGNOSTICS",
    "02 · DEGRADATION ANALYSIS",
    "03 · MODEL BENCHMARKS"
])

with tab1:
    col_t1, col_t2 = st.columns(2, gap="large")

    with col_t1:
        rul_bar = progress_bar(
            cycle_fraction, "Estimated Health Fraction", f"{cycle_fraction:.1%}"
        )
        st.markdown(compact(f"""
        <div class="panel">
            <div class="panel-eyebrow">Task 1</div>
            <div class="panel-title">Battery Lifecycle Prognostics</div>
            <div class="panel-sub"><b>Champion Engine:</b> Tuned HistGradientBoosting &nbsp;·&nbsp; R² = 0.8972, RMSE = 528.64 cycles</div>
            {rul_bar}
            <div class="kv"><span>Estimated Useful Fleet Horizon <small>(350 charge cycles/year)</small></span><b>~{est_years:.1f} years</b></div>
            <div class="kv"><span>Current Health Percent</span><b>{input_df['battery_health_percent'].values[0]:.1f}%</b></div>
            <div class="kv"><span>Internal Resistance</span><b>{input_df['internal_resistance'].values[0]:.3f} mΩ</b></div>
            <div class="kv"><span>Resistance Rate</span><b>{input_df['resistance_per_1000_cycles'].values[0]:.2f} mΩ / 1,000 cycles</b></div>
        </div>
        """), unsafe_allow_html=True)

    with col_t2:
        fail_bar = progress_bar(
            prob_failure, "Failure Probability", f"{prob_failure:.1%}",
            marker=tau_slider, marker_label=f"Safety cutoff τ = {tau_slider:.3f}"
        )
        st.markdown(compact(f"""
        <div class="panel">
            <div class="panel-eyebrow">Task 2</div>
            <div class="panel-title">Critical Failure Risk Calibration</div>
            <div class="panel-sub"><b>Active Engine:</b> {_html.escape(active_model_name)}</div>
            {fail_bar}
            <div class="kv"><span>Threshold Decision Rule</span><b>Flag = 1 if P(Failure) ≥ {tau_slider:.3f}</b></div>
            <div class="kv"><span>Cost Matrix Ratio (C<sub>FN</sub> / C<sub>FP</sub>)</span><b>13.45 : 1</b></div>
            <div class="kv wrap"><span>False Alarm Reduction:</span> <b>Safety threshold achieved 90.25% recall on the evaluation set with 139 false alarms (175 fewer than the uncalibrated boosting baseline).</b></div>
        </div>
        """), unsafe_allow_html=True)

with tab2:
    st.markdown(compact("""
    <div class="section-title">Physical Feature Attribution</div>
    <div class="section-sub">Top electro-chemical and thermal variables influencing the current vehicle diagnosis:</div>
    """), unsafe_allow_html=True)

    # Feature breakdown cards
    f1, f2, f3 = st.columns(3, gap="medium")
    f1.metric("Thermal Spread (Max - Avg)", f"{input_df['temperature_spread'].values[0]:.2f} °C", help="Elevated spread indicates localized hot-spotting.")
    f2.metric("Health-Loss Interaction", f"{input_df['health_loss_interaction'].values[0]:.2f}", help="Composite product of health and capacity loss.")
    f3.metric("Effective C-Rate Proxy", f"{input_df['c_rate_proxy'].values[0]:.3f} C", help="Charging power divided by pack capacity.")

    st.markdown('<div class="table-title">Primary Degradation Indicators in This Vehicle</div>', unsafe_allow_html=True)
    raw_telemetry_display = input_df[[
        'battery_health_percent', 'capacity_loss_percent', 'cell_temperature_max',
        'cell_temperature_avg', 'internal_resistance', 'cycle_count',
        'thermal_runaway_risk', 'thermal_health_score', 'charge_efficiency'
    ]].T
    raw_telemetry_display.columns = ["Observed Telemetry Value"]

    telemetry_rows = "".join(
        f'<tr><td class="mono">{_html.escape(str(idx))}</td><td class="num">{fmt_value(row.iloc[0])}</td></tr>'
        for idx, row in raw_telemetry_display.iterrows()
    )
    st.markdown(compact(
        '<div class="tbl-wrap"><table class="data-tbl"><thead><tr>'
        '<th>Telemetry Variable</th><th class="num">Observed Telemetry Value</th>'
        f'</tr></thead><tbody>{telemetry_rows}</tbody></table></div>'
    ), unsafe_allow_html=True)

with tab3:
    st.markdown(compact("""
    <div class="section-title">Group Necrons — Master Evaluation Leaderboard</div>
    <div class="section-sub">Consolidated benchmarks from <b>Viva 2 / Phase 2 submission</b> across all 4 teammates.</div>
    """), unsafe_allow_html=True)

    st.markdown('<div class="table-title">Task 1: Remaining Useful Life (RUL) Cycles Regression</div>', unsafe_allow_html=True)
    t1_table = pd.DataFrame([
        {'Model': 'Dummy Regressor (Mean Floor)', 'Test R2': -0.0015, 'Test RMSE': 1649.79, 'Test MAE': 1349.30, 'Status': 'Baseline Floor'},
        {'Model': 'Ordinary Least Squares (OLS)', 'Test R2': 0.8963, 'Test RMSE': 530.95, 'Test MAE': 423.45, 'Status': 'Linear Benchmark'},
        {'Model': 'Ridge Regression (Tuned alpha=31.62)', 'Test R2': 0.8963, 'Test RMSE': 530.83, 'Test MAE': 423.56, 'Status': 'L2 Regularized'},
        {'Model': 'Lasso Regression (Tuned alpha=1.00)', 'Test R2': 0.8966, 'Test RMSE': 530.16, 'Test MAE': 423.02, 'Status': 'L1 Sparse'},
        {'Model': 'Decision Tree (max_depth=10)', 'Test R2': 0.8444, 'Test RMSE': 650.27, 'Test MAE': 512.39, 'Status': 'Tree Baseline'},
        {'Model': 'Random Forest (100 Trees)', 'Test R2': 0.8939, 'Test RMSE': 537.00, 'Test MAE': 430.62, 'Status': 'Bagging Ensemble'},
        {'Model': 'HistGradientBoosting (Tuned CV)', 'Test R2': 0.8972, 'Test RMSE': 528.64, 'Test MAE': 423.11, 'Status': '★ Task 1 Champion'}
    ])
    st.markdown(render_table(
        t1_table,
        num_cols=('Test R2', 'Test RMSE', 'Test MAE'),
        decimals={'Test R2': 4, 'Test RMSE': 2, 'Test MAE': 2}
    ), unsafe_allow_html=True)

    st.markdown('<div class="table-title">Task 2: Critical Battery Failure Classification (13.45:1 Imbalance)</div>', unsafe_allow_html=True)
    t2_table = pd.DataFrame([
        {'Model': 'Dummy Classifier (Majority)', 'Recall': '0.00%', 'Precision': '0.00%', 'PR-AUC': 0.0693, 'ROC-AUC': 0.5000, 'Status': 'Naive Floor'},
        {'Model': 'Unweighted Decision Tree', 'Recall': '54.00%', 'Precision': '54.00%', 'PR-AUC': 0.3261, 'ROC-AUC': 0.7538, 'Status': 'Tree Baseline'},
        {'Model': 'Balanced Random Forest', 'Recall': '51.26%', 'Precision': '62.56%', 'PR-AUC': 0.6015, 'ROC-AUC': 0.9583, 'Status': 'Cost-Sensitive Bagging'},
        {'Model': 'Original Cost-Sensitive XGBoost', 'Recall': '91.34%', 'Precision': '44.62%', 'PR-AUC': 0.7420, 'ROC-AUC': 0.9756, 'Status': 'High-Recall Boosting'},
        {'Model': 'Tuned XGBoost (RandomizedSearch)', 'Recall': '89.53%', 'Precision': '62.63%', 'PR-AUC': 0.7798, 'ROC-AUC': 0.9811, 'Status': '★ Champion Tree Model'},
        {'Model': 'Safety-Calibrated Logistic (tau=0.193)', 'Recall': '90.25%', 'Precision': '64.27%', 'PR-AUC': 0.7896, 'ROC-AUC': 0.9830, 'Status': '★ Champion Linear Model'}
    ])
    st.markdown(render_table(
        t2_table,
        num_cols=('Recall', 'Precision', 'PR-AUC', 'ROC-AUC'),
        decimals={'PR-AUC': 4, 'ROC-AUC': 4}
    ), unsafe_allow_html=True)

st.markdown(
    '<div class="app-footer"><b>⚡</b> SLIIT IT3051 Data Mining Project &nbsp;|&nbsp; Group: Necrons &nbsp;|&nbsp; Dual-Task EV Battery PHM Deployment System</div>',
    unsafe_allow_html=True,
)