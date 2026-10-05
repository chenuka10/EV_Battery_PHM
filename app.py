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
# Design System: Clean White + Electric Blue + Battery Green
# ==============================================================================
THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {
    --bg: #F8FAFC;
    --card: #FFFFFF;
    --card-hover: #FBFDFF;
    --blue: #2563EB;
    --blue-dark: #1E40AF;
    --blue-light: #EFF6FF;
    --blue-border: #BFDBFE;
    --green: #059669;
    --green-dark: #047857;
    --green-light: #ECFDF5;
    --green-border: #A7F3D0;
    --navy: #0F172A;
    --text: #1E293B;
    --muted: #64748B;
    --border: #E2E8F0;
    --sidebar: #FFFFFF;
    --warn: #D97706;
    --warn-light: #FFFBEB;
    --warn-border: #FDE68A;
    --crit: #DC2626;
    --crit-light: #FEF2F2;
    --crit-border: #FECACA;
    --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea, .stApp [data-baseweb="select"] div {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
}

.stApp {
    background-color: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stAppDeployButton"] {
    display: none;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
    max-width: 1400px;
}

/* ---------- Header ---------- */
.main-header {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: var(--navy);
    line-height: 1.2;
    margin: 0;
}

.main-header svg {
    flex: none;
}

.sub-header {
    font-size: 0.98rem;
    color: var(--muted);
    margin: 0.45rem 0 1.1rem 0;
    line-height: 1.55;
    max-width: 68rem;
}

.hdr-rule {
    height: 2px;
    background: linear-gradient(90deg, var(--blue) 0%, var(--green) 35%, var(--border) 100%);
    position: relative;
    margin-bottom: 1.5rem;
    border-radius: 2px;
}

/* ---------- How to use bar ---------- */
.how-to-use {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1.4rem;
    display: flex;
    gap: 1.5rem;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 3px rgba(15,23,42,0.03);
}

.how-step {
    font-size: 0.8rem;
    color: var(--navy);
}

.how-step b {
    color: var(--blue);
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] {
    padding-top: 1.5rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

.sb-brand {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}

.sb-brand-icon {
    width: 44px;
    height: 44px;
    border-radius: 10px;
    background: var(--green-light);
    border: 1px solid var(--green-border);
    display: flex;
    align-items: center;
    justify-content: center;
}

.sb-brand-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--navy);
    letter-spacing: -0.01em;
    line-height: 1.2;
}

.sb-brand-sub {
    font-size: 0.75rem;
    color: var(--green-dark);
    font-weight: 600;
    margin-top: 0.15rem;
    letter-spacing: 0.02em;
}

.sb-section {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin: 1.4rem 0 0.65rem 0;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--navy);
}

.sb-section .n {
    font-family: var(--mono);
    color: var(--blue);
    background: var(--blue-light);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.7rem;
}

.sb-section::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Threshold calibration panel */
.tau-panel {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(15,23,42,0.03);
}

.tau-panel::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: var(--blue);
}

.tau-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.tau-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
}

.tau-state {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--blue);
    background: var(--blue-light);
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
}

.tau-value {
    font-family: var(--mono);
    font-size: 1.95rem;
    font-weight: 700;
    color: var(--navy);
    letter-spacing: -0.02em;
    margin: 0.4rem 0 0.75rem 0;
}

.tau-value .tau-sym {
    color: var(--blue);
}

.tau-refs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
}

.tau-ref {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.45rem 0.6rem;
    background: #F8FAFC;
    text-align: center;
}

.tau-ref span {
    display: block;
    font-size: 0.62rem;
    letter-spacing: 0.08em;
    font-weight: 600;
    color: var(--muted);
}

.tau-ref b {
    font-family: var(--mono);
    font-size: 0.95rem;
    color: var(--navy);
    font-weight: 700;
}

.tau-ref.active {
    border-color: var(--blue);
    background: var(--blue-light);
}

.tau-ref.active span {
    color: var(--blue);
}

/* Radio buttons in sidebar */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.4rem;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.75rem;
    width: 100%;
    transition: all 0.15s ease;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    border-color: var(--blue);
    background: var(--blue-light);
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: var(--blue-light);
    border-color: var(--blue);
    box-shadow: 0 0 0 1px var(--blue);
}

[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--navy);
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: #FFFFFF;
    color: var(--navy);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.82rem;
    padding: 0.5rem 0.7rem;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    border-color: var(--blue);
    color: var(--blue);
    background: var(--blue-light);
    box-shadow: 0 2px 6px rgba(37,99,235,0.08);
}

.stButton > button:active {
    background: var(--blue);
    color: #FFFFFF;
    border-color: var(--blue);
}

/* Containers styled */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03) !important;
    padding: 0.4rem !important;
}

/* ---------- Metric Overview Cards ---------- */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.3rem;
    min-height: 145px;
    box-shadow: 0 2px 4px rgba(15,23,42,0.03), 0 1px 2px rgba(15,23,42,0.02);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.metric-card:hover {
    box-shadow: 0 4px 12px rgba(15,23,42,0.06);
}

.metric-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    line-height: 1.3;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--navy);
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0.5rem 0 0.35rem 0;
    font-variant-numeric: tabular-nums;
    display: flex;
    align-items: center;
}

.metric-value .unit {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--muted);
    letter-spacing: 0;
    margin-left: 0.45rem;
}

.metric-detail {
    font-size: 0.78rem;
    color: var(--blue);
    font-weight: 600;
    line-height: 1.4;
}

.metric-detail.green {
    color: var(--green);
}

/* ---------- Status Badges ---------- */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    white-space: nowrap;
}

.status-badge i {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
}

.sb-safe {
    background: var(--green-light);
    color: var(--green-dark);
    border: 1px solid var(--green-border);
}
.sb-safe i {
    background: var(--green);
}

.sb-warn {
    background: var(--warn-light);
    color: var(--warn);
    border: 1px solid var(--warn-border);
}
.sb-warn i {
    background: var(--warn);
}

.sb-crit {
    background: var(--crit-light);
    color: var(--crit);
    border: 1px solid var(--crit-border);
}
.sb-crit i {
    background: var(--crit);
}

/* ---------- Diagnostic Alert Banner ---------- */
.alert {
    border: 1px solid var(--border);
    border-left: 5px solid var(--c);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 1.4rem 0 1.8rem 0;
    background: var(--bg-t);
    box-shadow: 0 2px 6px rgba(15,23,42,0.03);
}

.alert-safe {
    --c: var(--green);
    --bg-t: #F0FDF4;
    border-color: var(--green-border);
}

.alert-warn {
    --c: var(--warn);
    --bg-t: #FFFBEB;
    border-color: var(--warn-border);
}

.alert-crit {
    --c: var(--crit);
    --bg-t: #FEF2F2;
    border-color: var(--crit-border);
}

.alert-head {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    flex-wrap: wrap;
    margin-bottom: 0.9rem;
}

.alert-title {
    font-size: 1.08rem;
    font-weight: 800;
    color: var(--navy);
    letter-spacing: -0.01em;
}

.alert-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 2.4fr;
    gap: 1.5rem;
    padding: 0.9rem 0;
    border-top: 1px solid rgba(15,23,42,0.06);
    border-bottom: 1px solid rgba(15,23,42,0.06);
}

.alert-k {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 0.25rem;
}

.alert-v {
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--navy);
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.alert-v .sym {
    color: var(--blue);
}

.alert-v.act {
    font-size: 0.92rem;
    font-weight: 600;
    line-height: 1.45;
    letter-spacing: 0;
}

.alert-note {
    margin-top: 0.85rem;
    font-size: 0.86rem;
    color: var(--muted);
    line-height: 1.55;
}

.alert-note b {
    color: var(--navy);
    font-weight: 700;
}

/* ---------- Scenario Showcase Cards ---------- */
.scenario-box {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.15rem 1.1rem 1rem 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 220px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.03);
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}

.scenario-box:hover {
    box-shadow: 0 6px 16px rgba(15,23,42,0.06);
    transform: translateY(-2px);
}

.scenario-box.active {
    border: 2px solid var(--blue);
    background: #FAFCFF;
    box-shadow: 0 6px 18px rgba(37,99,235,0.12);
}

.scenario-badge-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}

.sc-active-pill {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #FFFFFF;
    background: var(--blue);
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
}

.sc-title {
    font-size: 0.98rem;
    font-weight: 800;
    color: var(--navy);
    line-height: 1.25;
    margin-bottom: 0.25rem;
}

.sc-sub {
    font-size: 0.74rem;
    color: var(--muted);
    margin-bottom: 0.75rem;
    line-height: 1.35;
}

.sc-metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.4rem;
    background: #F8FAFC;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 0.6rem;
    margin-bottom: 0.6rem;
}

.sc-metric-item {
    font-size: 0.7rem;
    color: var(--muted);
}

.sc-metric-item b {
    display: block;
    color: var(--navy);
    font-family: var(--mono);
    font-size: 0.8rem;
    font-weight: 700;
}

/* ---------- Feature Studio Panel ---------- */
.studio-head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.1rem;
}

.studio-head-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}

.studio-head-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--navy);
    letter-spacing: -0.01em;
}

.feature-explainer {
    background: #F8FAFC;
    border-left: 3px solid var(--blue);
    border-radius: 0 6px 6px 0;
    padding: 0.5rem 0.7rem;
    font-size: 0.76rem;
    color: var(--muted);
    line-height: 1.45;
    margin-top: -0.3rem;
    margin-bottom: 1.1rem;
}

.feature-explainer.green-accent {
    border-left-color: var(--green);
}

.feature-explainer b {
    color: var(--navy);
    font-weight: 600;
}

.feature-explainer .limits {
    display: block;
    font-size: 0.7rem;
    color: var(--blue-dark);
    font-weight: 600;
    margin-top: 0.2rem;
}

/* Sliders */
div[data-baseweb="slider"] [role="slider"] {
    background-color: var(--blue) !important;
    border-color: #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.2) !important;
}

[data-testid="stSliderThumbValue"] {
    color: var(--blue) !important;
    font-weight: 700;
    font-family: var(--mono) !important;
    font-size: 0.78rem;
}

[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
    color: var(--muted);
    font-size: 0.7rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    height: auto;
    padding: 0.8rem 1.4rem;
    background: transparent;
    border-radius: 8px 8px 0 0;
}

.stTabs [data-baseweb="tab"] p {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--muted);
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--blue-light);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--blue-light);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] p {
    color: var(--blue);
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--blue);
    height: 3px;
    border-radius: 2px 2px 0 0;
}

.stTabs [data-baseweb="tab-border"] {
    background-color: transparent;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* Panels */
.panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    min-height: 350px;
    box-shadow: 0 2px 4px rgba(15,23,42,0.03);
}

.panel-eyebrow {
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: var(--blue);
    text-transform: uppercase;
}

.panel-title {
    font-size: 1.18rem;
    font-weight: 800;
    color: var(--navy);
    letter-spacing: -0.01em;
    margin: 0.25rem 0 0.5rem 0;
}

.panel-sub {
    font-size: 0.86rem;
    color: var(--muted);
    line-height: 1.5;
    margin-bottom: 1.2rem;
}

.panel-sub b {
    color: var(--navy);
    font-weight: 700;
}

.kv {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 1rem;
    padding: 0.65rem 0;
    border-top: 1px solid #F1F5F9;
    font-size: 0.86rem;
}

.kv span {
    color: var(--muted);
}

.kv span small {
    color: #94A3B8;
    font-size: 0.74rem;
}

.kv b {
    color: var(--navy);
    font-family: var(--mono);
    font-weight: 700;
    font-size: 0.85rem;
    text-align: right;
}

.kv.wrap {
    display: block;
    line-height: 1.55;
}

.kv.wrap b {
    font-family: inherit;
}

/* Progress bars */
.pbar {
    margin: 0.3rem 0 1.25rem 0;
}

.pbar-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.82rem;
    margin-bottom: 0.5rem;
}

.pbar-head span {
    color: var(--muted);
    font-weight: 600;
}

.pbar-head b {
    color: var(--navy);
    font-family: var(--mono);
    font-weight: 700;
}

.pbar-track {
    position: relative;
    height: 12px;
    background: #F1F5F9;
    border-radius: 6px;
    border: 1px solid var(--border);
}

.pbar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--green) 0%, var(--blue) 100%);
    border-radius: 6px;
}

.pbar-fill.risk {
    background: linear-gradient(90deg, var(--green) 0%, var(--warn) 60%, var(--crit) 100%);
}

.pbar-tick {
    position: absolute;
    top: -6px;
    width: 3px;
    height: 22px;
    background: var(--navy);
    border-radius: 2px;
}

.pbar-legend {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin-top: 0.5rem;
    font-size: 0.74rem;
    color: var(--muted);
    font-weight: 500;
}

.pbar-legend i {
    width: 3px;
    height: 12px;
    background: var(--navy);
    display: inline-block;
    border-radius: 1px;
}

/* Data Tables */
.tbl-wrap {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow-x: auto;
    box-shadow: 0 2px 4px rgba(15,23,42,0.03);
    margin-bottom: 1.2rem;
}

table.data-tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
}

table.data-tbl thead th {
    background: #F8FAFC;
    color: var(--navy);
    text-align: left;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.85rem 1.1rem;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}

table.data-tbl tbody td {
    padding: 0.75rem 1.1rem;
    color: var(--text);
    border-bottom: 1px solid #F1F5F9;
    white-space: nowrap;
}

table.data-tbl tbody tr:last-child td {
    border-bottom: none;
}

table.data-tbl tbody tr:nth-child(even) td {
    background: #FAFCFE;
}

table.data-tbl tbody tr:hover td {
    background: var(--blue-light);
}

table.data-tbl .num {
    text-align: right;
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    font-size: 0.82rem;
}

table.data-tbl .mdl {
    font-weight: 700;
    color: var(--navy);
}

table.data-tbl .mono {
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--navy);
}

table.data-tbl tbody tr.champ td {
    background: #F0F7FF !important;
}

table.data-tbl tbody tr.champ td:first-child {
    box-shadow: inset 4px 0 0 var(--blue);
}

.chip {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--muted);
}

.chip.champ {
    color: var(--blue-dark);
    background: #FFFFFF;
    border: 1px solid var(--blue-border);
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-weight: 700;
}

/* Headings */
.section-title {
    font-size: 1.22rem;
    font-weight: 800;
    color: var(--navy);
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem 0;
}

.section-sub {
    font-size: 0.88rem;
    color: var(--muted);
    margin-bottom: 1.3rem;
    line-height: 1.5;
}

.section-sub b {
    color: var(--navy);
    font-weight: 700;
}

.table-title {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1rem;
    font-weight: 800;
    color: var(--navy);
    margin: 1.6rem 0 0.8rem 0;
}

.table-title::before {
    content: "";
    width: 4px;
    height: 1.15rem;
    background: var(--green);
    border-radius: 2px;
}

.table-title.blue-accent::before {
    background: var(--blue);
}

/* Footer */
.app-footer {
    margin-top: 3.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    font-size: 0.76rem;
    color: var(--muted);
    letter-spacing: 0.02em;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.8rem;
}

.app-footer b {
    color: var(--blue);
    font-weight: 700;
}
"""

st.markdown("<style>" + compact(THEME_CSS) + "</style>", unsafe_allow_html=True)

# ==============================================================================
# UI SVG Assets & Helpers
# ==============================================================================
BOLT_SVG = (
    '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>'
)

BATTERY_SVG = (
    '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<rect x="2" y="7" width="16" height="10" rx="2" ry="2"/>'
    '<line x1="22" y1="11" x2="22" y2="13"/>'
    '<line x1="6" y1="12" x2="14" y2="12"/>'
    '<line x1="10" y1="9" x2="10" y2="15"/></svg>'
)


def sb_section(num: str, title: str):
    st.sidebar.markdown(
        compact(f'<div class="sb-section"><span class="n">{num}</span>{title}</div>'),
        unsafe_allow_html=True,
    )


def progress_bar(frac: float, label: str, value: str, marker: float = None, marker_label: str = "", is_risk: bool = False):
    frac = min(max(frac, 0.0), 1.0)
    tick = ""
    legend = ""
    fill_cls = "pbar-fill risk" if is_risk else "pbar-fill"
    if marker is not None:
        tick = f'<div class="pbar-tick" style="left:calc({min(max(marker, 0.0), 1.0) * 100:.1f}% - 1.5px)"></div>'
        legend = f'<div class="pbar-legend"><i></i>{marker_label}</div>'
    return compact(f"""
    <div class="pbar">
        <div class="pbar-head"><span>{label}</span><b>{value}</b></div>
        <div class="pbar-track"><div class="{fill_cls}" style="width:{frac * 100:.1f}%"></div>{tick}</div>
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
    decimals = decimals or {}
    head = "".join(
        f'<th class="{"num" if c in num_cols else ""}">{_html.escape(str(c))}</th>' for c in df.columns
    )
    rows = []
    for _, row in df.iterrows():
        is_champ = status_col and status_col in df.columns and "Champion" in str(row[status_col])
        cells = []
        for c in df.columns:
            v = row[c]
            if c == model_col:
                cells.append(f'<td class="mdl">{_html.escape(str(v))}</td>')
            elif status_col and c == status_col:
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
# Model Loading, Feature Engineering Helper & State Manager
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
    with open('models/feature_ranges.json', 'r') as f:
        ranges = json.load(f)
    test_scores = pd.read_csv('models/test_scores.csv')
    return ct, hgb, lr, xgb, meta, scenarios, ranges, test_scores


try:
    ct, model_rul, model_lr, model_xgb, meta, scenarios, feature_ranges, test_scores_df = load_artifacts()
except Exception as e:
    st.error(f"Error loading system artifacts: {e}")
    st.stop()


def featurize(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Calculate the 7 verified domain engineering features with epsilon protection."""
    df = df_raw.copy()
    df['temperature_spread'] = df['cell_temperature_max'] - df['cell_temperature_avg']
    df['efficiency_gap'] = (df['charge_efficiency'] - df['discharge_efficiency']).abs()
    df['health_loss_interaction'] = (df['battery_health_percent'] * df['capacity_loss_percent']) / 100.0
    df['resistance_per_1000_cycles'] = (df['internal_resistance'] / (df['cycle_count'] + 1.0)) * 1000.0
    df['c_rate_proxy'] = df['average_charge_power_kw'] / (df['battery_capacity_kwh'] + 1e-5)
    df['cell_voltage_spread'] = df['cell_voltage_std'] / (df['cell_voltage_avg'] + 1e-5)
    df['stress_index'] = df['aggressive_acceleration_score'] * df['hard_braking_score']
    return df


# ==============================================================================
# Parameter Definitions & Explanations Dictionary (Derived from Data & Report)
# ==============================================================================
SLIDER_CONFIG = {
    # 1. Electrochemical Wear
    'battery_health_percent': {
        'name': 'Battery State of Health (SOH)',
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
        'desc': 'Frequency of harsh deceleration causing high-current regenerative inrush surges.',
        'safe_range': 'Observed Dataset: 0.0 – 100.0',
        'physics': 'Severe regen inrush at high state-of-charge causes transient over-voltage stress on separator membranes.'
    }
}

SCENARIO_KEYS = list(scenarios.keys())

def load_into_state(scenario_data: dict, slider_configs: dict):
    """Safely clamp and type-cast telemetry values before storing in session_state."""
    for k, cfg in slider_configs.items():
        val = scenario_data.get(k, cfg['default'])
        if pd.isna(val) or val is None:
            val = cfg['default']
        val = max(min(float(val), float(cfg['max'])), float(cfg['min']))
        if isinstance(cfg['step'], int):
            val = int(round(val))
        st.session_state[f"input_{k}"] = val

# Initialize session state
if 'active_scenario' not in st.session_state:
    st.session_state['active_scenario'] = SCENARIO_KEYS[0]
    load_into_state(scenarios[SCENARIO_KEYS[0]]['data'], SLIDER_CONFIG)

if 'tau_cutoff' not in st.session_state:
    st.session_state['tau_cutoff'] = 0.193


# ==============================================================================
# Sidebar: Engine & Persistent Safety Cutoff Controls
# ==============================================================================
st.sidebar.markdown(
    compact(f"""
    <div class="sb-brand">
        <div class="sb-brand-icon">{BATTERY_SVG}</div>
        <div>
            <div class="sb-brand-title">BMS Edge PHM Control</div>
            <div class="sb-brand-sub">Group Necrons · Dual-Task Inference</div>
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)

# 1. Diagnostic Engine Toggle
sb_section("01", "Diagnostic Engine")
engine_choice = st.sidebar.radio(
    "Select Model Inference Tier:",
    ["Edge BMS (Calibrated Logistic Regression)", "Cloud Fleet Analytics (Tuned XGBoost)"],
    label_visibility="collapsed"
)

# 2. Decision Threshold Control (Safety Cutoff)
sb_section("02", "Safety Threshold Calibration")

col_b1, col_b2 = st.sidebar.columns(2)
if col_b1.button("Default (0.50)", use_container_width=True):
    st.session_state['tau_cutoff'] = 0.50
    st.rerun()
if col_b2.button("Safety (0.193)", use_container_width=True):
    st.session_state['tau_cutoff'] = 0.193
    st.rerun()

tau_slider = st.sidebar.slider(
    "Critical Failure Cutoff (τ)",
    min_value=0.05,
    max_value=0.95,
    value=float(st.session_state['tau_cutoff']),
    step=0.01,
    key="tau_slider_widget",
    help="Default threshold is 0.50. The calibrated safety threshold is 0.193, achieving 90.25% recall on the evaluation set."
)
st.session_state['tau_cutoff'] = tau_slider

if abs(tau_slider - 0.193) < 1e-4:
    tau_state = "CALIBRATED (SAFETY)"
elif abs(tau_slider - 0.50) < 1e-4:
    tau_state = "DEFAULT (0.50)"
else:
    tau_state = "CUSTOM"

st.sidebar.markdown(
    compact(f"""
    <div class="tau-panel">
        <div class="tau-top">
            <div class="tau-label">Operating Cutoff</div>
            <div class="tau-state">{tau_state}</div>
        </div>
        <div class="tau-value"><span class="tau-sym">τ</span> = {tau_slider:.3f}</div>
        <div class="tau-refs">
            <div class="tau-ref {'active' if abs(tau_slider-0.50)<1e-4 else ''}"><span>DEFAULT</span><b>0.500</b></div>
            <div class="tau-ref {'active' if abs(tau_slider-0.193)<1e-4 else ''}"><span>SAFETY</span><b>0.193</b></div>
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)

# 3. Model Architecture Specs in Sidebar
sb_section("03", "System Specifications")
st.sidebar.markdown(compact("""
<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:0.75rem; font-size:0.76rem; color:#64748B; line-height:1.55;">
    <div style="display:flex; justify-content:space-between; margin-bottom:0.35rem;">
        <span style="font-weight:600; color:#0F172A;">Task 1 Architecture:</span>
        <span style="font-family:'JetBrains Mono'; font-weight:700; color:#2563EB;">HistGB (R²=0.897)</span>
    </div>
    <div style="display:flex; justify-content:space-between; margin-bottom:0.35rem;">
        <span style="font-weight:600; color:#0F172A;">Task 2 Edge:</span>
        <span style="font-family:'JetBrains Mono'; font-weight:700; color:#059669;">Calibrated Logistic</span>
    </div>
    <div style="display:flex; justify-content:space-between; margin-bottom:0.35rem;">
        <span style="font-weight:600; color:#0F172A;">Imbalance Ratio:</span>
        <span style="font-family:'JetBrains Mono'; font-weight:700; color:#0F172A;">13.45 : 1</span>
    </div>
    <div style="display:flex; justify-content:space-between;">
        <span style="font-weight:600; color:#0F172A;">Evaluation Recall:</span>
        <span style="font-family:'JetBrains Mono'; font-weight:700; color:#059669;">90.25% (τ=0.193)</span>
    </div>
</div>
"""), unsafe_allow_html=True)


# ==============================================================================
# Main Dashboard: Header & Step-by-Step Guide
# ==============================================================================
st.markdown(
    compact(f'<div class="main-header">{BOLT_SVG}<span>EV Battery Prognostics &amp; Health Management (PHM)</span></div>'),
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Dual-Task Edge Prognostics: Continuous RUL Cycles Regression (Task 1) &amp; Calibrated Failure Risk Early Warning (Task 2)</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="hdr-rule"></div>', unsafe_allow_html=True)

# 3-Step Guide for Non-Technical Users
st.markdown(compact("""
<div class="how-to-use">
    <div class="how-step"><b style="color:#2563EB;">Step 1:</b> Select a vehicle scenario below or adjust sliders in the studio</div>
    <div class="how-step"><b style="color:#2563EB;">Step 2:</b> Choose your operating cutoff (τ) in the sidebar (Default 0.50 vs Safety 0.193)</div>
    <div class="how-step"><b style="color:#2563EB;">Step 3:</b> Review Remaining Life (Task 1) and Early Warning Risk (Task 2)</div>
</div>
"""), unsafe_allow_html=True)


# ==============================================================================
# Main Section: Attractive Dynamic Scenario Showcase Cards
# ==============================================================================
st.markdown('<div class="section-title">Verified Vehicle Telemetry Scenarios</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Select any real vehicle profile below to populate telemetry data, or customize physical parameters directly in the studio.</div>',
    unsafe_allow_html=True
)

sc_cols = st.columns(4)

scenario_meta = [
    {
        'key': SCENARIO_KEYS[0],
        'icon': '🟢',
        'badge': 'sb-safe',
        'badge_text': 'NOMINAL HEALTH',
        'title': 'Factory-Fresh Battery',
        'sub': 'Low Cycles · Peak Health',
        'desc': 'Nominal operation with low resistance and stable thermal dynamics.'
    },
    {
        'key': SCENARIO_KEYS[1],
        'icon': '🔵',
        'badge': 'sb-safe',
        'badge_text': 'NORMAL WEAR',
        'title': 'Aged Fleet Commuter',
        'sub': 'Moderate Degradation',
        'desc': 'Expected capacity fade with stable thermal envelope.'
    },
    {
        'key': SCENARIO_KEYS[2],
        'icon': '🟠',
        'badge': 'sb-warn',
        'badge_text': '⚡ INTERCEPTED AT τ=0.193',
        'title': 'Incipient Hazard',
        'sub': 'Early Warning Showcase',
        'desc': 'Passes default 0.50 cutoff, but caught early by calibrated safety threshold!'
    },
    {
        'key': SCENARIO_KEYS[3],
        'icon': '🔴',
        'badge': 'sb-crit',
        'badge_text': 'CRITICAL RUNAWAY',
        'title': 'Active Runaway Risk',
        'sub': 'Imminent Failure State',
        'desc': 'Severe thermal divergence and capacity collapse requiring immediate inspection.'
    }
]

for idx, col in enumerate(sc_cols):
    info = scenario_meta[idx]
    is_active = (st.session_state['active_scenario'] == info['key'])
    active_cls = "active" if is_active else ""
    pill_html = '<span class="sc-active-pill">★ ACTIVE</span>' if is_active else f'<span class="status-badge {info["badge"]}">{info["badge_text"]}</span>'
    
    # Compute quick telemetry stats dynamically from scenario data
    s_data = scenarios[info['key']]['data']
    s_cycles = f"{int(s_data.get('cycle_count', 0)):,}"
    s_soh = f"{s_data.get('battery_health_percent', 0.0):.1f}%"
    s_temp = f"{s_data.get('cell_temperature_max', 0.0):.1f}°C"
    
    # Run dynamic inference for card probability badge
    try:
        s_df = featurize(pd.DataFrame([s_data]))
        s_trans = ct.transform(s_df[meta['features']])
        s_prob = float(model_lr.predict_proba(s_trans)[0, 1])
        s_risk_str = f"{s_prob:.1%}" if s_prob >= 0.001 else "< 0.1%"
    except Exception:
        s_risk_str = "N/A"

    with col:
        st.markdown(compact(f"""
        <div class="scenario-box {active_cls}">
            <div>
                <div class="scenario-badge-top">{pill_html}<span style="font-size:1.15rem;">{info['icon']}</span></div>
                <div class="sc-title">{info['title']}</div>
                <div class="sc-sub">{info['sub']}</div>
                <div class="sc-metrics">
                    <div class="sc-metric-item">Cycles<b>{s_cycles}</b></div>
                    <div class="sc-metric-item">SOH<b>{s_soh}</b></div>
                    <div class="sc-metric-item">Max Temp<b>{s_temp}</b></div>
                    <div class="sc-metric-item">Failure Risk<b>{s_risk_str}</b></div>
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

        btn_label = f"✓ Loaded ({info['title']})" if is_active else f"Load {info['title']}"
        if st.button(btn_label, key=f"btn_sc_{idx}", use_container_width=True):
            st.session_state['active_scenario'] = info['key']
            load_into_state(scenarios[info['key']]['data'], SLIDER_CONFIG)
            st.rerun()

st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)


# ==============================================================================
# Main Section: Interactive Telemetry & Feature Studio
# ==============================================================================
st.markdown('<div class="section-title">Interactive Telemetry &amp; Feature Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Adjust live battery telemetry inputs below. Each parameter displays its <b>physical meaning</b>, <b>normal operating limits</b>, and <b>degradation mechanics</b>.<br>'
    '<span style="font-size:0.75rem; color:#2563EB; font-weight:600;">ℹ️ Studio mode adjusts 12 primary physical drivers; 61 secondary attributes remain locked to baseline telemetry.</span></div>',
    unsafe_allow_html=True
)

col_studio1, col_studio2, col_studio3 = st.columns(3, gap="medium")

# ----- Panel 1: Electrochemical Wear & Capacity -----
with col_studio1:
    with st.container(border=True):
        st.markdown(compact("""
        <div class="studio-head">
            <div class="studio-head-icon" style="background:#ECFDF5; color:#059669;">🔋</div>
            <div class="studio-head-title">Electrochemical Wear &amp; Capacity</div>
        </div>
        """), unsafe_allow_html=True)

        # Battery Health
        c1 = SLIDER_CONFIG['battery_health_percent']
        v_health = st.slider(
            c1['name'], c1['min'], c1['max'],
            float(st.session_state.get('input_battery_health_percent', c1['default'])),
            c1['step'], key='input_battery_health_percent'
        )
        st.markdown(compact(f"""
        <div class="feature-explainer green-accent">
            <b>What it means:</b> {c1['desc']}<br>
            <b>Physics:</b> {c1['physics']}
            <span class="limits">{c1['safe_range']}</span>
        </div>
        """), unsafe_allow_html=True)

        # Capacity Loss
        c2 = SLIDER_CONFIG['capacity_loss_percent']
        v_cap_loss = st.slider(
            c2['name'], c2['min'], c2['max'],
            float(st.session_state.get('input_capacity_loss_percent', 100.0 - v_health)),
            c2['step'], key='input_capacity_loss_percent'
        )
        st.markdown(compact(f"""
        <div class="feature-explainer green-accent">
            <b>What it means:</b> {c2['desc']}<br>
            <b>Physics:</b> {c2['physics']}
            <span class="limits">{c2['safe_range']}</span>
        </div>
        """), unsafe_allow_html=True)

        # Internal Resistance
        c3 = SLIDER_CONFIG['internal_resistance']
        v_res = st.slider(
            c3['name'], c3['min'], c3['max'],
            float(st.session_state.get('input_internal_resistance', c3['default'])),
            c3['step'], key='input_internal_resistance'
        )
        st.markdown(compact(f"""
        <div class="feature-explainer green-accent">
            <b>What it means:</b> {c3['desc']}<br>
            <b>Physics:</b> {c3['physics']}
            <span class="limits">{c3['safe_range']}</span>
        </div>
        """), unsafe_allow_html=True)

        # Cycle Count
        c4 = SLIDER_CONFIG['cycle_count']
        v_cycles = st.slider(
            c4['name'], c4['min'], c4['max'],
            int(st.session_state.get('input_cycle_count', c4['default'])),
            c4['step'], key='input_cycle_count'
        )
        st.markdown(compact(f"""
        <div class="feature-explainer green-accent">
            <b>What it means:</b> {c4['desc']}<br>
            <b>Physics:</b> {c4['physics']}
            <span class="limits">{c4['safe_range']}</span>
        </div>
        """), unsafe_allow_html=True)

# ----- Panel 2: Thermal Dynamics & Heat Safety -----
with col_studio2:
    with st.container(border=True):
        st.markdown(compact("""
        <div class="studio-head">
            <div class="studio-head-icon" style="background:#EFF6FF; color:#2563EB;">🌡️</div>
            <div class="studio-head-title">Thermal Dynamics &amp; Safety</div>
        </div>
        """), unsafe_allow_html=True)

        # Max Cell Temp
        c5 = SLIDER_CONFIG['cell_temperature_max']
        v_temp_max = st.slider(
            c5['name'], c5['min'], c5['max'],
            float(st.session_state.get('input_cell_temperature_max', c5['default'])),
            c5['step'], key='input_cell_temperature_max'
        )
        st.markdown(compact(f"""
        <div class="feature-explainer">
            <b>What it means:</b> {c5['desc']}<br>
            <b>Physics:</b> {c5['physics']}
            <span class="limits">{c5['safe_range']}</span>
        </div>
        """), unsafe_allow_html=True)

        # Avg Cell Temp
        c6 = SLIDER_CONFIG['cell_temperature_avg']
        v_temp_avg = st.slider(
            c6['name'], c6['min'], c6['max'],
            float(st.session_state.get('input_cell_temperature_avg', c6['default'])),
            c6['step'], key='input_cell_temperature_avg'
        )
        st.markdown(compact(f"""
        <div class="feature-explainer">
            <b>What it means:</b> {c6['desc']}<br>
            <b>Physics:</b> {c6['physics']}
            <span class="limits">{c6['safe_range']}</span>
        </div>
        """), unsafe_allow_html=True)

        # Thermal Runaway Risk
        c7 = SLIDER_CONFIG['thermal_runaway_risk']
        v_runaway = st.slider(
            c7['name'], c7['min'], c7['max'],
            float(st.session_state.get('input_thermal_runaway_risk', c7['default'])),
            c7['step'], key='input_thermal_runaway_risk'
        )
        st.markdown(compact(f"""
        <div class="feature-explainer">
            <b>What it means:</b> {c7['desc']}<br>
            <b>Physics:</b> {c7['physics']}
            <span class="limits">{c7['safe_range']}</span>
        </div>
        """), unsafe_allow_html=True)

        # Thermal Health Score
        c8 = SLIDER_CONFIG['thermal_health_score']
        v_therm_health = st.slider(
            c8['name'], c8['min'], c8['max'],
            float(st.session_state.get('input_thermal_health_score', c8['default'])),
            c8['step'], key='input_thermal_health_score'
        )
        st.markdown(compact(f"""
        <div class="feature-explainer">
            <b>What it means:</b> {c8['desc']}<br>
            <b>Physics:</b> {c8['physics']}
            <span class="limits">{c8['safe_range']}</span>
        </div>
        """), unsafe_allow_html=True)

# ----- Panel 3: Charging Power, C-Rate & Driving Stress -----
with col_studio3:
    with st.container(border=True):
        st.markdown(compact("""
        <div class="studio-head">
            <div class="studio-head-icon" style="background:#F0FDF4; color:#047857;">⚡</div>
            <div class="studio-head-title">Charging Power &amp; Dynamic Stress</div>
        </div>
        """), unsafe_allow_html=True)

        # Charge Power
        c9 = SLIDER_CONFIG['average_charge_power_kw']
        v_charge_kw = st.slider(
            c9['name'], c9['min'], c9['max'],
            float(st.session_state.get('input_average_charge_power_kw', c9['default'])),
            c9['step'], key='input_average_charge_power_kw'
        )
        st.markdown(compact(f"""
        <div class="feature-explainer green-accent">
            <b>What it means:</b> {c9['desc']}<br>
            <b>Physics:</b> {c9['physics']}
            <span class="limits">{c9['safe_range']}</span>
        </div>
        """), unsafe_allow_html=True)

        # Pack Capacity
        c10 = SLIDER_CONFIG['battery_capacity_kwh']
        v_capacity_kwh = st.slider(
            c10['name'], c10['min'], c10['max'],
            float(st.session_state.get('input_battery_capacity_kwh', c10['default'])),
            c10['step'], key='input_battery_capacity_kwh'
        )
        st.markdown(compact(f"""
        <div class="feature-explainer green-accent">
            <b>What it means:</b> {c10['desc']}<br>
            <b>Physics:</b> {c10['physics']}
            <span class="limits">{c10['safe_range']}</span>
        </div>
        """), unsafe_allow_html=True)

        # Aggressive Acceleration
        c11 = SLIDER_CONFIG['aggressive_acceleration_score']
        v_accel = st.slider(
            c11['name'], c11['min'], c11['max'],
            float(st.session_state.get('input_aggressive_acceleration_score', c11['default'])),
            c11['step'], key='input_aggressive_acceleration_score'
        )
        st.markdown(compact(f"""
        <div class="feature-explainer green-accent">
            <b>What it means:</b> {c11['desc']}<br>
            <b>Physics:</b> {c11['physics']}
            <span class="limits">{c11['safe_range']}</span>
        </div>
        """), unsafe_allow_html=True)

        # Hard Braking
        c12 = SLIDER_CONFIG['hard_braking_score']
        v_brake = st.slider(
            c12['name'], c12['min'], c12['max'],
            float(st.session_state.get('input_hard_braking_score', c12['default'])),
            c12['step'], key='input_hard_braking_score'
        )
        st.markdown(compact(f"""
        <div class="feature-explainer green-accent">
            <b>What it means:</b> {c12['desc']}<br>
            <b>Physics:</b> {c12['physics']}
            <span class="limits">{c12['safe_range']}</span>
        </div>
        """), unsafe_allow_html=True)


# ==============================================================================
# Stage 9/10: Physical Input Validation Layer
# ==============================================================================
validation_warnings = []
validation_errors = []

# Physical consistency check: Temperature Spread
if v_temp_max < v_temp_avg:
    validation_errors.append(f"Physical Anomaly: Maximum Cell Hotspot Temperature ({v_temp_max:.1f}°C) cannot be lower than Average Pack Temperature ({v_temp_avg:.1f}°C).")

# Physical consistency check: Degradation Conservation
health_loss_sum = v_health + v_cap_loss
if abs(health_loss_sum - 100.0) > 6.0:
    validation_warnings.append(f"Collinearity Drift Notice: State of Health ({v_health:.1f}%) and Capacity Loss ({v_cap_loss:.1f}%) sum to {health_loss_sum:.1f}% (Nominal conservation sum is ~100%).")

# Outlier bounds check against training set
if v_temp_max > feature_ranges['cell_temperature_max']['iqr_upper']:
    validation_warnings.append(f"Statistical Outlier: Max Hotspot Temperature ({v_temp_max:.1f}°C) exceeds upper training IQR boundary ({feature_ranges['cell_temperature_max']['iqr_upper']:.1f}°C).")

if v_res > feature_ranges['internal_resistance']['iqr_upper']:
    validation_warnings.append(f"High Resistance: Internal Resistance ({v_res:.3f} Ω) lies in the top 1% extreme wear tail.")

if validation_errors:
    for err in validation_errors:
        st.error(f"⛔ {err}")
    st.stop()

if validation_warnings:
    with st.expander("⚠️ Telemetry Consistency & Boundary Advisories", expanded=False):
        for warn in validation_warnings:
            st.warning(warn)


# ==============================================================================
# Model Feature Engineering & Dual-Task Inference
# ==============================================================================
active_scenario_name = st.session_state['active_scenario']
current_data = scenarios[active_scenario_name]['data'].copy()

# Ensure medians/defaults are filled for non-slider features
for col in meta['features']:
    if col not in current_data or pd.isna(current_data[col]):
        if col in meta['num_defaults']:
            current_data[col] = meta['num_defaults'][col]
        elif col in meta['cat_options']:
            current_data[col] = meta['cat_options'][col][0]

# Overwrite modified slider parameters
current_data['battery_health_percent'] = v_health
current_data['capacity_loss_percent'] = v_cap_loss
current_data['internal_resistance'] = v_res
current_data['cycle_count'] = v_cycles
current_data['cell_temperature_max'] = v_temp_max
current_data['cell_temperature_avg'] = v_temp_avg
current_data['thermal_runaway_risk'] = v_runaway
current_data['thermal_health_score'] = v_therm_health
current_data['average_charge_power_kw'] = v_charge_kw
current_data['battery_capacity_kwh'] = v_capacity_kwh
current_data['aggressive_acceleration_score'] = v_accel
current_data['hard_braking_score'] = v_brake

# Build DataFrame and run featurize()
input_df = featurize(pd.DataFrame([current_data]))

# Preprocess through the 156-feature ColumnTransformer
input_transformed = ct.transform(input_df[meta['features']])

# Task 1: RUL Cycles Regression
pred_rul = float(model_rul.predict(input_transformed)[0])

# Task 2: Failure Probability Inference
if "Logistic" in engine_choice:
    prob_failure = float(model_lr.predict_proba(input_transformed)[0, 1])
    active_model_name = "Safety-Calibrated Logistic Regression"
else:
    prob_failure = float(model_xgb.predict_proba(input_transformed)[0, 1])
    active_model_name = "Tuned XGBoost (PR-AUC Champion)"

# Determine Hazard Status against active threshold tau
is_flagged = prob_failure >= tau_slider
cycle_fraction = min(max(pred_rul / 12500.0, 0.0), 1.0)
est_years = pred_rul / 350.0  # labeled as fleet assumption

if not is_flagged:
    status_key, status_label = "safe", "NOMINAL"
elif prob_failure < 0.50 and is_flagged:
    status_key, status_label = "warn", "EARLY WARNING"
else:
    status_key, status_label = "crit", "CRITICAL"

badge_html = f'<span class="status-badge sb-{status_key}"><i></i>{status_label}</span>'


# ==============================================================================
# Overview Metrics (4 Prominent KPI Cards)
# ==============================================================================
def metric_card(title, value_html, detail_html, is_green=False):
    detail_cls = "metric-detail green" if is_green else "metric-detail"
    return compact(f"""
    <div class="metric-card">
        <div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value_html}</div>
        </div>
        <div class="{detail_cls}">{detail_html}</div>
    </div>
    """)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(metric_card(
        "Predicted Remaining Life (RUL)",
        f'{pred_rul:,.0f}<span class="unit">cycles</span>',
        f"≈ {est_years:.1f} yr horizon (assumed 350 cycles/yr)",
        is_green=True
    ), unsafe_allow_html=True)

with m2:
    st.markdown(metric_card(
        "Failure Probability P(Failure)",
        f"{prob_failure:.2%}",
        _html.escape(active_model_name)
    ), unsafe_allow_html=True)

with m3:
    st.markdown(metric_card(
        "Active Safety Cutoff (τ)",
        f"{tau_slider:.3f}",
        "Default 0.500 · Safety 0.193"
    ), unsafe_allow_html=True)

with m4:
    st.markdown(metric_card(
        "Operational Safety Status",
        badge_html,
        f"P = {prob_failure:.2%} {'≥' if is_flagged else '&lt;'} τ = {tau_slider:.3f}"
    ), unsafe_allow_html=True)

# Plain Language Summary for Executives
plain_status = "FLAGGED FOR PREVENTATIVE WORKSHOP INSPECTION" if is_flagged else "CLEARED FOR ROUTINE FLEET OPERATION"
st.info(
    f"📋 **Executive Decision Summary:** This battery pack is currently **{plain_status}**. "
    f"It has an estimated **{pred_rul:,.0f} cycles** remaining (~{est_years:.1f} years at assumed 350 cycles/year). "
    f"Estimated failure risk is **{prob_failure:.2%}**, evaluated against cutoff **τ = {tau_slider:.3f}**."
)


# ==============================================================================
# Diagnostic Alert Banner
# ==============================================================================
if status_key == "safe":
    alert_title = "BMS Telemetry Status: Normal Operation"
    alert_action = "No elevated-risk flag raised. Battery pack operating within nominal electro-thermal boundaries."
    alert_note = (
        f"Estimated failure probability (<b>{prob_failure:.2%}</b>) is below the active operating cutoff "
        f"(τ = {tau_slider:.3f}). Regular fleet telemetry logging is maintained."
    )
elif status_key == "warn":
    alert_title = f"Early Warning: Intercepted by Safety Cutoff (τ = {tau_slider:.3f})"
    alert_action = "Recommended action: Schedule preventative dealer diagnostic; inspect thermal harness and cell balance."
    alert_note = (
        f"<b>Early Warning Interception:</b> Standard cutoff (0.50) would categorize this vehicle as normal (P = {prob_failure:.2%}). "
        f"The safety-calibrated threshold flags this battery early, enabling depot inspection prior to on-road failure."
    )
else:
    alert_title = f"Critical Alert: Failure Probability Exceeds Cutoff (P = {prob_failure:.2%})"
    alert_action = "Recommended action: Urgent workshop service required; prioritize full high-voltage electrical check."
    alert_note = (
        f"Failure probability (<b>{prob_failure:.2%}</b>) exceeds critical operational boundaries, "
        f"indicating severe internal impedance, rapid thermal escalation, or cell divergence."
    )

st.markdown(compact(f"""
<div class="alert alert-{status_key}">
    <div class="alert-head">{badge_html}<span class="alert-title">{alert_title}</span></div>
    <div class="alert-grid">
        <div><div class="alert-k">Failure Probability</div><div class="alert-v">{prob_failure:.2%}</div></div>
        <div><div class="alert-k">Operating Cutoff</div><div class="alert-v"><span class="sym">τ</span> = {tau_slider:.3f}</div></div>
        <div><div class="alert-k">Recommended Protocol</div><div class="alert-v act">{alert_action}</div></div>
    </div>
    <div class="alert-note">{alert_note}</div>
</div>
"""), unsafe_allow_html=True)


# ==============================================================================
# Dual-Task Deep Dive & Master Leaderboards (Tabs)
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "01 · LIVE PROGNOSTICS & ALERT-BURDEN",
    "02 · FEATURE ATTRIBUTION & AUDIT",
    "03 · GROUP NECRONS MASTER LEADERBOARD",
    "04 · DATA HEALTH & METHODOLOGY"
])

with tab1:
    col_t1, col_t2 = st.columns(2, gap="large")

    with col_t1:
        rul_bar = progress_bar(
            cycle_fraction, "Estimated Lifespan Remaining (Scale: 12,500 Max Cycles)", f"{cycle_fraction:.1%}"
        )
        st.markdown(compact(f"""
        <div class="panel">
            <div class="panel-eyebrow">Task 1 · Lifecycle Regression</div>
            <div class="panel-title">Battery Useful Life Prognostics</div>
            <div class="panel-sub"><b>Champion Model:</b> Tuned HistGradientBoosting &nbsp;·&nbsp; R² = 0.8972, RMSE = 528.64 cycles</div>
            {rul_bar}
            <div class="kv"><span>Estimated Fleet Horizon <small>(assumed 350 cycles/year)</small></span><b>~{est_years:.1f} years</b></div>
            <div class="kv"><span>State of Health (SOH)</span><b>{v_health:.1f}%</b></div>
            <div class="kv"><span>Internal Resistance (ESR)</span><b>{v_res:.3f} Ω</b></div>
            <div class="kv"><span>Resistance Aging Rate</span><b>{input_df['resistance_per_1000_cycles'].values[0]:.4f} Ω / 1,000 cycles</b></div>
            <div class="kv"><span>Estimated Total Lifetime</span><b>{(v_cycles + pred_rul):,.0f} cycles</b></div>
        </div>
        """), unsafe_allow_html=True)

        st.markdown(compact("""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:0.8rem 1rem; margin-top:0.8rem; font-size:0.78rem; color:#475569;">
            <b>Typical Model Prediction Error Band:</b> ±423 cycles (MAE) / ±529 cycles (RMSE).<br>
            <i>Methodological Notice (Report Section 9.2, Table 12):</i> In the lowest-RUL quartile (batteries near end-of-life), 
            the model exhibits an average overprediction bias of <b>-201.62 cycles</b>. 
            Field operators should apply a conservative safety buffer when predicted RUL falls below 3,000 cycles.
        </div>
        """), unsafe_allow_html=True)

    with col_t2:
        fail_bar = progress_bar(
            prob_failure, "Failure Risk Probability", f"{prob_failure:.2%}",
            marker=tau_slider, marker_label=f"Cutoff τ = {tau_slider:.3f}",
            is_risk=True
        )
        st.markdown(compact(f"""
        <div class="panel">
            <div class="panel-eyebrow">Task 2 · Cost-Sensitive Risk</div>
            <div class="panel-title">Critical Failure Risk Calibration</div>
            <div class="panel-sub"><b>Active Engine:</b> {_html.escape(active_model_name)}</div>
            {fail_bar}
            <div class="kv"><span>Decision Rule</span><b>Flag = 1 if P(Failure) ≥ {tau_slider:.3f}</b></div>
            <div class="kv"><span>Class Imbalance Ratio (Non-Failure : Failure)</span><b>13.45 : 1</b></div>
            <div class="kv"><span>Safety Recall at τ = 0.193 on Evaluation Set</span><b>90.25%</b></div>
            <div class="kv wrap"><span>Operational Advantage:</span> <b>Safety threshold (τ=0.193) intercepts 90.25% of true failures with 139 false alarms (175 fewer false alarms than the uncalibrated boosting baseline of 314 FP).</b></div>
        </div>
        """), unsafe_allow_html=True)

    # Live Alert-Burden Calculator from test_scores.csv
    st.markdown("---")
    st.markdown(f"#### Live Test-Set Performance at Operating Cutoff (τ = {tau_slider:.3f})")
    
    active_prob_col = 'p_lr' if "Logistic" in engine_choice else 'p_xgb'
    y_true_eval = test_scores_df['y_true'].values
    p_eval = test_scores_df[active_prob_col].values

    preds_at_tau = (p_eval >= tau_slider).astype(int)
    tp = int(np.sum((preds_at_tau == 1) & (y_true_eval == 1)))
    fp = int(np.sum((preds_at_tau == 1) & (y_true_eval == 0)))
    fn = int(np.sum((preds_at_tau == 0) & (y_true_eval == 1)))
    tn = int(np.sum((preds_at_tau == 0) & (y_true_eval == 0)))

    live_recall = tp / (tp + fn + 1e-10)
    live_precision = tp / (tp + fp + 1e-10)
    live_f1 = 2 * (live_precision * live_recall) / (live_precision + live_recall + 1e-10)

    ab1, ab2, ab3, ab4 = st.columns(4)
    ab1.metric("Recall (Interception Rate)", f"{live_recall:.2%}", help="Fraction of all true failures flagged")
    ab2.metric("Precision (True Alarms)", f"{live_precision:.2%}", help="Fraction of alarms that are true failures")
    ab3.metric("Missed Failures (FN)", f"{fn}", delta=f"{fn - 27} vs Safety τ=0.193", delta_color="inverse")
    ab4.metric("False Alarms (FP)", f"{fp}", delta=f"{fp - 139} vs Safety τ=0.193", delta_color="inverse")

    cm_display = pd.DataFrame([
        {"Actual State": "True Failure (277 in test set)", "Flagged for Inspection": f"✅ {tp} (True Positives)", "Cleared Normal": f"⛔ {fn} (Missed Failures)"},
        {"Actual State": "Normal Battery (3,723 in test set)", "Flagged for Inspection": f"⚠️ {fp} (False Alarms)", "Cleared Normal": f"✅ {tn} (True Negatives)"}
    ])
    st.table(cm_display)

    # Fleet Decision Simulator
    with st.expander("💼 Fleet Manager Decision & Workshop Workload Simulator", expanded=False):
        st.markdown("Simulate workshop inspection capacity and undetected battery failures across a customized fleet size based on test-set error rates:")
        sim_col1, sim_col2 = st.columns(2)
        fleet_size = sim_col1.number_input("Fleet Size (Total Electric Vehicles):", min_value=100, max_value=100000, value=5000, step=500)
        depot_capacity = sim_col2.number_input("Depot Monthly Inspection Capacity (Inspections / Month):", min_value=10, max_value=2000, value=300, step=25)

        failure_rate_expected = 277 / 4000.0  # 6.925% prevalence
        expected_failures = fleet_size * failure_rate_expected
        expected_normal = fleet_size * (1.0 - failure_rate_expected)

        est_flags = (expected_failures * live_recall) + (expected_normal * (fp / 3723.0))
        est_missed = expected_failures * (1.0 - live_recall)

        s1, s2, s3 = st.columns(3)
        s1.metric("Expected Monthly Inbound Alarms", f"{est_flags:,.0f} EVs", delta=f"{est_flags - depot_capacity:,.0f} vs Depot Capacity", delta_color="inverse")
        s2.metric("Expected Intercepted Failures", f"{expected_failures * live_recall:,.0f} EVs")
        s3.metric("Projected Undetected On-Road Failures", f"{est_missed:,.0f} EVs", delta_color="inverse")

        if est_flags > depot_capacity:
            st.warning(f"⚠️ Inspection capacity bottleneck: Current cutoff triggers {est_flags:,.0f} inspection notices, exceeding workshop monthly capacity of {depot_capacity}. Consider increasing τ slightly toward F1-optimal (0.278) to throttle false alarm burden.")
        else:
            st.success(f"✅ Workshop bandwidth adequate: {est_flags:,.0f} monthly inspections fits within depot capacity of {depot_capacity}.")

    # Accuracy Paradox Expander
    with st.expander("💡 The Accuracy Paradox in EV Safety: Why 93% Accuracy Fails", expanded=False):
        st.markdown("""
        In imbalanced failure classification (where only **6.925%** of batteries fail), raw **Accuracy** is dangerously misleading:
        * A **Dummy Majority Classifier** that unconditionally predicts every battery is healthy achieves **93.08% Accuracy**.
        * However, it misses **277 out of 277 catastrophic failures** (0% Recall, maximum false negative cost).
        """)
        p1, p2 = st.columns(2)
        p1.error("❌ **Naive Majority Baseline:** Accuracy = 93.08% | Failures Intercepted = 0 / 277 (100% Missed)")
        p2.success("✅ **Safety Model (τ = 0.193):** Accuracy = 95.85% | Failures Intercepted = 250 / 277 (90.25% Recall)")

with tab2:
    st.markdown(compact("""
    <div class="section-title">Engineered Physical Features Attribution</div>
    <div class="section-sub">Computed domain features derived directly from raw BMS sensor channels for this vehicle snapshot:</div>
    """), unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3, gap="medium")
    f1.metric("Thermal Spread (Max - Avg)", f"{input_df['temperature_spread'].values[0]:.2f} °C", help="Elevated spread indicates localized cooling obstruction.")
    f2.metric("Health-Loss Interaction", f"{input_df['health_loss_interaction'].values[0]:.2f}", help="Coupled electro-chemical degradation interaction term.")
    f3.metric("Effective C-Rate Proxy", f"{input_df['c_rate_proxy'].values[0]:.3f} C", help="Ratio of charging kW to pack kWh.")

    st.markdown('<div class="table-title">Full 7 Domain Features Breakdown</div>', unsafe_allow_html=True)
    domain_summary = pd.DataFrame([
        {'Feature': 'temperature_spread', 'Formula': 'cell_temp_max - cell_temp_avg', 'Observed Value': f"{input_df['temperature_spread'].values[0]:.3f} °C", 'Significance': 'Detects thermal gradients and localized module hot-spots'},
        {'Feature': 'health_loss_interaction', 'Formula': '(health % * capacity_loss %) / 100', 'Observed Value': f"{input_df['health_loss_interaction'].values[0]:.3f}", 'Significance': 'Captures non-linear compounding capacity fade'},
        {'Feature': 'c_rate_proxy', 'Formula': 'charge_power_kw / capacity_kwh', 'Observed Value': f"{input_df['c_rate_proxy'].values[0]:.4f} C", 'Significance': 'Normalizes charging stress across battery pack sizes'},
        {'Feature': 'resistance_per_1000_cycles', 'Formula': '(resistance / (cycles + 1)) * 1000', 'Observed Value': f"{input_df['resistance_per_1000_cycles'].values[0]:.4f} Ω", 'Significance': 'Evaluates rate of ohmic degradation per cycle'},
        {'Feature': 'cell_voltage_spread', 'Formula': 'cell_voltage_std / cell_voltage_avg', 'Observed Value': f"{input_df['cell_voltage_spread'].values[0]:.5f}", 'Significance': 'Monitors parallel cell string voltage balancing'},
        {'Feature': 'stress_index', 'Formula': 'accel_score * braking_score', 'Observed Value': f"{input_df['stress_index'].values[0]:.2f}", 'Significance': 'Measures compound mechanical dynamic driving fatigue'},
        {'Feature': 'efficiency_gap', 'Formula': '|charge_eff - discharge_eff|', 'Observed Value': f"{input_df['efficiency_gap'].values[0]:.3f} %", 'Significance': 'Quantifies parasitic thermodynamic hysteresis loss'}
    ])
    st.markdown(render_table(domain_summary, model_col="Feature", status_col=None), unsafe_allow_html=True)

    # Feature Attribution (Why this Score)
    st.markdown('<div class="table-title blue-accent">Linear Attribution Influencers for Current Battery (Task 2)</div>', unsafe_allow_html=True)
    if "Logistic" in engine_choice:
        coefs = model_lr.coef_[0]
        raw_contrib = input_transformed[0] * coefs
        top_pos_idx = np.argsort(raw_contrib)[-4:][::-1]
        top_neg_idx = np.argsort(raw_contrib)[:4]
        
        attr_data = []
        feat_names = ct.get_feature_names_out()
        for idx in top_pos_idx:
            attr_data.append({"Feature": feat_names[idx], "Impact Direction": "🔺 Elevates Failure Risk", "Contribution Weight": f"{raw_contrib[idx]:+.3f}"})
        for idx in top_neg_idx:
            attr_data.append({"Feature": feat_names[idx], "Impact Direction": "🟢 Promotes Health/Normalcy", "Contribution Weight": f"{raw_contrib[idx]:+.3f}"})
        st.markdown(render_table(pd.DataFrame(attr_data), model_col="Feature", status_col=None), unsafe_allow_html=True)
        st.caption("Linear coefficient attribution: Feature value × Logistic weight. *Note: Reflects statistical association in the model, not verified physical causality.*")
    else:
        st.info("Active model is XGBoost. Linear attribution unavailable; feature usage is dictated by gradient tree splits.")

    st.markdown('<div class="table-title">Observed Telemetry Values in Current Profile</div>', unsafe_allow_html=True)
    raw_telemetry_display = input_df[[
        'battery_health_percent', 'capacity_loss_percent', 'cell_temperature_max',
        'cell_temperature_avg', 'internal_resistance', 'cycle_count',
        'thermal_runaway_risk', 'thermal_health_score', 'charge_efficiency',
        'discharge_efficiency', 'average_charge_power_kw', 'battery_capacity_kwh'
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
    <div class="section-sub">Consolidated benchmarks across all models evaluated during the dual-task project lifecycle (Report Tables 8, 26, 28, 29).</div>
    """), unsafe_allow_html=True)

    st.markdown('<div class="table-title">Task 1: Remaining Useful Life (RUL) Cycles Regression</div>', unsafe_allow_html=True)
    t1_table = pd.DataFrame([
        {'Model': 'Dummy Regressor (Mean Floor)', 'Test R2': -0.0015, 'Test RMSE': 1649.79, 'Test MAE': 1349.30, 'Status': 'Baseline Floor'},
        {'Model': 'Ordinary Least Squares (OLS)', 'Test R2': 0.8963, 'Test RMSE': 530.95, 'Test MAE': 423.45, 'Status': 'Linear Benchmark'},
        {'Model': 'Ridge Regression (Tuned alpha=31.62)', 'Test R2': 0.8963, 'Test RMSE': 530.83, 'Test MAE': 423.56, 'Status': 'L2 Regularized'},
        {'Model': 'Lasso Regression (Tuned alpha=1.00)', 'Test R2': 0.8966, 'Test RMSE': 530.16, 'Test MAE': 423.02, 'Status': 'L1 Sparse (90 Zeroed)'},
        {'Model': 'Decision Tree (max_depth=10)', 'Test R2': 0.8444, 'Test RMSE': 650.27, 'Test MAE': 512.39, 'Status': 'Tree Baseline'},
        {'Model': 'Random Forest (100 Trees)', 'Test R2': 0.8939, 'Test RMSE': 537.00, 'Test MAE': 430.62, 'Status': 'Bagging Ensemble'},
        {'Model': 'HistGradientBoosting (Tuned CV)', 'Test R2': 0.8972, 'Test RMSE': 528.64, 'Test MAE': 423.11, 'Status': '★ Task 1 Champion'}
    ])
    st.markdown(render_table(
        t1_table,
        num_cols=('Test R2', 'Test RMSE', 'Test MAE'),
        decimals={'Test R2': 4, 'Test RMSE': 2, 'Test MAE': 2}
    ), unsafe_allow_html=True)

    st.markdown('<div class="table-title blue-accent">Task 2: Critical Battery Failure Classification (13.45:1 Imbalance)</div>', unsafe_allow_html=True)
    t2_table = pd.DataFrame([
        {'Model': 'Dummy Classifier (Majority)', 'Recall': '0.00%', 'Precision': '0.00%', 'PR-AUC': 0.0693, 'ROC-AUC': 0.5000, 'Status': 'Naive Floor'},
        {'Model': 'Unweighted Decision Tree', 'Recall': '54.15%', 'Precision': '54.35%', 'PR-AUC': 0.3261, 'ROC-AUC': 0.7538, 'Status': 'Tree Baseline'},
        {'Model': 'Balanced Random Forest', 'Recall': '51.26%', 'Precision': '62.56%', 'PR-AUC': 0.6015, 'ROC-AUC': 0.9583, 'Status': 'Cost-Sensitive Bagging'},
        {'Model': 'Cost-Sensitive XGBoost (scale_pos_weight=13.45)', 'Recall': '91.34%', 'Precision': '44.62%', 'PR-AUC': 0.7420, 'ROC-AUC': 0.9756, 'Status': 'High-Recall Boosting'},
        {'Model': 'Tuned XGBoost (RandomizedSearch CV)', 'Recall': '91.70%', 'Precision': '57.08%', 'PR-AUC': 0.7804, 'ROC-AUC': 0.9812, 'Status': '★ Champion Tree Model'},
        {'Model': 'Safety-Calibrated Logistic (tau=0.193)', 'Recall': '90.25%', 'Precision': '64.27%', 'PR-AUC': 0.7896, 'ROC-AUC': 0.9830, 'Status': '★ Champion Linear Model'}
    ])
    st.markdown(render_table(
        t2_table,
        num_cols=('Recall', 'Precision', 'PR-AUC', 'ROC-AUC'),
        decimals={'PR-AUC': 4, 'ROC-AUC': 4}
    ), unsafe_allow_html=True)
    st.caption("Footnote: All Task 2 models evaluated on the 4,000-row stratified test set (277 failures). Safety-Calibrated Logistic uses τ = 0.193; all other models use default threshold (0.50). *Methodological Note: τ=0.193 was calibrated on the evaluation partition to demonstrate safety thresholding.*")

with tab4:
    st.markdown(compact("""
    <div class="section-title">Data Quality &amp; Machine Learning Pipeline Architecture</div>
    <div class="section-sub">Verified dataset structure and preprocessing pipeline integrity from Report Sections 3, 5, and 11:</div>
    """), unsafe_allow_html=True)

    dq1, dq2, dq3, dq4 = st.columns(4)
    dq1.metric("Raw Dataset Size", "20,000 records", "70 telemetry attributes")
    dq2.metric("Incomplete Records", "18,663 records (93.3%)", "Missingness in 67/70 fields")
    dq3.metric("Exact Duplicates", "0 records (0.00%)", "Validated unique battery IDs")
    dq4.metric("Excluded Unlabelled RUL", "898 records (4.49%)", "Excluded only from Task 1")

    st.markdown(compact("""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:1.2rem; margin:1.2rem 0; font-size:0.84rem; line-height:1.6; color:#1E293B;">
        <b style="color:#2563EB;">End-to-End Pipeline Stepper:</b><br>
        1. <b>Data Ingestion:</b> 20,000 rows × 70 raw features (60 numerical, 10 categorical).<br>
        2. <b>Domain Feature Engineering:</b> 7 non-linear physical interaction features computed before splitting.<br>
        3. <b>Leakage-Safe Partitioning:</b> Task 1 uses 80/20 random split (19,102 rows). Task 2 uses 80/20 stratified split (4,000 test rows, 277 failures).<br>
        4. <b>ColumnTransformer Pipeline:</b> Numerical fields pass through SimpleImputer(median) + StandardScaler. Categoricals pass through SimpleImputer(most_frequent) + OneHotEncoder.<br>
        5. <b>Feature Dimension:</b> 73 model predictors expand to 156 transformed features.<br>
        6. <b>Inference & Decision Calibration:</b> HistGradientBoosting predicts RUL cycles. Calibrated Logistic Regression evaluates failure probability at safety threshold τ = 0.193.
    </div>
    """), unsafe_allow_html=True)

    st.markdown("#### System Boundaries & Academic Limitations (Report Section 15)")
    st.warning("""
    * **Decision-Support Scope:** This software prototype is designed for operational fleet analytics and warranty planning. It is not an automotive safety certification system.
    * **RUL Extrapolation Boundary:** Model RUL predictions exhibit an average overprediction bias (-201 cycles) on batteries nearing end-of-life (Q1 quartile). Physical load tests should supplement predictions when RUL < 3,000 cycles.
    * **Synthetic Manipulation Warning:** Manually dragging sliders to uncorrelated combinations (e.g. SOH 95% and Capacity Loss 40%) creates synthetic points outside the empirical training distribution.
    """)


# ==============================================================================
# Expandable System Self-Test Panel & Battery Passport Export
# ==============================================================================
st.markdown("---")
exp_col1, exp_col2 = st.columns([1, 1])

with exp_col1:
    with st.expander("🧪 System Self-Test Pipeline (Automated Test Suite)", expanded=False):
        st.markdown("Automated validation running all preset scenarios through the feature engineering, imputation, scaling, and dual-model inference pipeline:")
        test_results = []
        for sc_name, sc_obj in scenarios.items():
            try:
                raw_input = pd.DataFrame([sc_obj['data']])
                featurized = featurize(raw_input)
                transformed = ct.transform(featurized[meta['features']])
                r_pred = model_rul.predict(transformed)[0]
                p_pred = model_lr.predict_proba(transformed)[0, 1]
                valid_shape = (transformed.shape[1] == 156)
                valid_p = (0.0 <= p_pred <= 1.0)
                valid_r = (not np.isnan(r_pred))
                test_results.append({
                    "Scenario": sc_name.split(":")[0],
                    "Transformed Features": transformed.shape[1],
                    "RUL Prediction": f"{r_pred:,.1f} cycles",
                    "Failure Risk": f"{p_pred:.4%}",
                    "Integrity Status": "✅ PASS" if (valid_shape and valid_p and valid_r) else "❌ FAIL"
                })
            except Exception as ex:
                test_results.append({
                    "Scenario": sc_name.split(":")[0],
                    "Transformed Features": "ERR",
                    "RUL Prediction": "ERR",
                    "Failure Risk": "ERR",
                    "Integrity Status": f"❌ FAIL ({ex})"
                })
        st.dataframe(pd.DataFrame(test_results), use_container_width=True, hide_index=True)

with exp_col2:
    with st.expander("📄 Export Battery Diagnostic Passport", expanded=False):
        passport_text = f"""# EV BATTERY DIAGNOSTIC HEALTH PASSPORT
**Generated by:** Group Necrons Dual-Task BMS PHM System (SLIIT IT3051)
**Vehicle Reference:** {active_scenario_name.split(':')[0]}
**Operating Status:** {status_label} (Cutoff τ = {tau_slider:.3f})

---
### 1. Prognostic Indicators
- **Predicted Remaining Useful Life (RUL):** {pred_rul:,.0f} cycles (Typical Error: ±423 cycles)
- **Estimated Operating Horizon:** ~{est_years:.1f} years (at assumed 350 cycles/year)
- **Estimated Failure Probability:** {prob_failure:.2%}
- **Active Model Tier:** {active_model_name}

### 2. Primary Telemetry Snapshot
- **State of Health (SOH):** {v_health:.2f}%
- **Capacity Loss:** {v_cap_loss:.2f}%
- **Internal Resistance:** {v_res:.3f} Ω
- **Completed Cycles:** {v_cycles:,} cycles
- **Max Cell Hotspot Temperature:** {v_temp_max:.1f} °C
- **Average Pack Temperature:** {v_temp_avg:.1f} °C
- **Thermal Spread (Hotspot Gradient):** {(v_temp_max - v_temp_avg):.2f} °C
- **Thermal Runaway Risk Index:** {v_runaway:.1f} / 100

### 3. Recommended Operational Protocol
{alert_action}

---
*Disclaimer: Decision-support prototype developed for academic evaluation (SLIIT IT3051). Not certified as an automotive safety system.*
"""
        st.download_button(
            "📥 Download Diagnostic Passport (.md)",
            data=passport_text,
            file_name=f"Battery_Passport_{status_label}.md",
            mime="text/markdown",
            use_container_width=True
        )


# ==============================================================================
# Footer
# ==============================================================================
st.markdown(
    '<div class="app-footer">'
    '<div><b>⚡</b> SLIIT IT3051 Data Mining Project &nbsp;|&nbsp; Group: <b>Necrons</b> &nbsp;|&nbsp; Decision-support prototype, not an automotive safety certification</div>'
    '<div>Dual-Task EV Battery PHM Deployment System &nbsp;·&nbsp; Streamlit Web Dashboard</div>'
    '</div>',
    unsafe_allow_html=True,
)