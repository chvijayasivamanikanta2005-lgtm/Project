"""
app.py — Explainable AI EV Battery Management Dashboard
"""

import os
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

from utils.inference import (
    load_models,
    prepare_gru_sequence,
    predict_soh,
    calibrate_soh,
    estimate_cycle,
    estimate_current,
    construct_rl_state,
    predict_rl_action,
    get_battery_health_label,
)
from utils.xai import (
    compute_shap_values,
    generate_feature_importance_plot,
    generate_shap_distribution_plot,
    generate_shap_heatmap,
    generate_temperature_dependence,
    generate_cycle_dependence,
    generate_current_dependence,
    generate_action_influence_plot,
    generate_feature_ranking_plot,
    generate_combined_xai_plot,
    generate_reasoning_text,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable AI EV Battery Management Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS (Refined UI) ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    .dashboard-header {
        background: #ffffff;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
        margin-bottom: 30px;
        text-align: center;
        width: 100%;
    }

    .dashboard-title {
        color: #111827;
        font-weight: 700;
        margin: 0 !important;
        font-size: 1.8rem;
        line-height: 1.2;
    }
    
    .dashboard-subtitle {
        color: #4b5563;
        margin: 8px 0 0 0 !important;
        font-size: 0.95rem;
    }

    /* ── Neumorphic Number Inputs (Global) ─────────────────── */
    div[data-testid="stNumberInput"] {
        margin-bottom: 18px;
    }

    div[data-testid="stNumberInput"] label {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        letter-spacing: 0.3px !important;
        text-transform: uppercase !important;
        margin-bottom: 8px !important;
    }

    /* Outer container — the pill shape */
    div[data-testid="stNumberInput"] > div[data-testid="stNumberInput-StepperContainer"],
    div[data-testid="stNumberInput"] > div:last-child {
        background: #e0e5ec !important;
        border-radius: 50px !important;
        padding: 2px 8px !important;
        box-shadow:
            inset 6px 6px 12px #bec3c9,
            inset -6px -6px 12px #ffffff !important;
        transition: box-shadow 0.3s ease !important;
        border: none !important;
        overflow: hidden !important;
    }

    div[data-testid="stNumberInput"] > div:last-child:hover {
        box-shadow:
            inset 8px 8px 16px #b5bac0,
            inset -8px -8px 16px #ffffff !important;
    }

    /* Inner baseweb input container — fully transparent */
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-color: transparent !important;
        padding: 0 !important;
    }

    div[data-testid="stNumberInput"] div[data-baseweb="input"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* The actual input element */
    div[data-testid="stNumberInput"] input[type="number"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 10px 12px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        font-family: 'Inter', sans-serif !important;
    }

    div[data-testid="stNumberInput"] input[type="number"]:focus {
        outline: none !important;
        box-shadow: none !important;
        color: #111827 !important;
    }

    /* Style +/- buttons — blend into pill */
    div[data-testid="stNumberInput"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #94a3b8 !important;
        padding: 4px 10px !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        color: #3b82f6 !important;
        background: rgba(59, 130, 246, 0.06) !important;
    }

    /* ── Dashboard Card (for prediction panels) ────────────── */
    .dashboard-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 24px;
        width: 100%;
    }

    /* ══════════════════════════════════════════════════════════════════════════════
       RESPONSIVE MEDIA QUERIES
    ══════════════════════════════════════════════════════════════════════════════ */
    
    html, body, [data-testid="stAppViewContainer"], .stApp {
        max-width: 100vw !important;
        overflow-x: hidden !important;
    }

    @media screen and (min-width: 1920px) {
        .dashboard-header { padding: 35px; }
        .dashboard-title { font-size: 2.2rem; }
        div[data-testid="stNumberInput"] input[type="number"] { font-size: 16px !important; }
    }

    @media screen and (max-width: 1600px) {
        .dashboard-title { font-size: 1.7rem; }
    }

    @media screen and (max-width: 1440px) {
        .dashboard-header { padding: 20px; }
        .dashboard-title { font-size: 1.6rem; }
    }

    @media screen and (max-width: 1280px) {
        .dashboard-subtitle { font-size: 0.9rem; }
    }

    @media screen and (max-width: 1024px) {
        .dashboard-title { font-size: 1.5rem; }
        .dashboard-card { padding: 20px; }
        div[data-testid="stNumberInput"] { margin-bottom: 14px; }
    }

    @media screen and (max-width: 900px) {
        .dashboard-header { padding: 18px; }
        .dashboard-title { font-size: 1.4rem; }
    }

    @media screen and (max-width: 768px) {
        div[data-testid="column"] { 
            width: 100% !important; 
            flex: 1 1 100% !important; 
            min-width: 100% !important;
        }
        .dashboard-title { font-size: 1.3rem; }
        .dashboard-subtitle { font-size: 0.85rem; }
        div[data-testid="stNumberInput"] label { font-size: 0.75rem !important; }
        div[data-baseweb="tab-list"] { flex-wrap: wrap !important; gap: 8px; }
        .js-plotly-plot, .plotly { width: 100% !important; }
    }

    @media screen and (max-width: 600px) {
        .dashboard-header { padding: 15px; margin-bottom: 20px; }
        .dashboard-card { padding: 15px; margin-bottom: 15px; }
        .dashboard-title { font-size: 1.2rem; }
        div[data-testid="stButton"] { display: flex; justify-content: center; }
        div[data-testid="stButton"] button { width: 100%; padding: 12px !important; }
        div[data-baseweb="slider"] { padding: 10px 0 !important; }
    }

    @media screen and (max-width: 480px) {
        .dashboard-title { font-size: 1.1rem; }
        .dashboard-subtitle { font-size: 0.8rem; }
        div[data-testid="stNumberInput"] > div:last-child { padding: 1px 4px !important; }
        div[data-testid="stNumberInput"] input[type="number"] { font-size: 14px !important; padding: 8px 10px !important; }
    }

    @media screen and (max-width: 360px) {
        .dashboard-header { padding: 12px; }
        .dashboard-title { font-size: 1rem; }
        div[data-testid="stNumberInput"] label { font-size: 0.7rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ── Initialize Session State ─────────────────────────────────────────────────
if "led_toggle" not in st.session_state:
    st.session_state["led_toggle"] = True

defaults = {
    "ir": 0.04, "qc": 1.5, "qd": 1.5,
    "tavg": 30.0, "tmax": 35.0, "chargetime": 5000,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER — Styled Header Card
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '''
    <div class="dashboard-header">
        <h1 class="dashboard-title">⚡ Explainable AI EV Battery Management Dashboard</h1>
        <p class="dashboard-subtitle">AI-driven battery health prediction and charging optimisation | GRU + Double DQN + SHAP</p>
    </div>
    ''',
    unsafe_allow_html=True
)

# ── Custom Streamlit Components ──────────────────────────────────────────────
led_switch = components.declare_component("led_switch", path="assets/led_switch")

# ── Sensor Inputs Section ────────────────────────────────────────────────────
# Toggle switch (has built-in "🔧 Battery Sensor Inputs" label)
led_switch(
    checked=st.session_state.get("led_toggle", True),
    key="led_toggle",
    default=st.session_state.get("led_toggle", True),
)

# ── Battery Sensor Inputs panel ──────────────────────────────────────────────
if st.session_state.get("led_toggle", True):
    input_container = st.container()
    with input_container:
        i_col1, i_col2, i_col3 = st.columns(3)
        with i_col1:
            st.session_state.ir = st.number_input("Internal Resistance (Ω)", value=0.04, step=0.001, format="%.4f", key="ir_input")
            st.session_state.tavg = st.number_input("Average Temp (°C)", value=30.0, step=0.1, format="%.1f", key="tavg_input")
        with i_col2:
            st.session_state.qc = st.number_input("Charge Capacity (Ah)", value=1.5, step=0.1, format="%.2f", key="qc_input")
            st.session_state.tmax = st.number_input("Maximum Temp (°C)", value=35.0, step=0.1, format="%.1f", key="tmax_input")
        with i_col3:
            st.session_state.qd = st.number_input("Discharge Capacity (Ah)", value=1.5, step=0.1, format="%.2f", key="qd_input")
            st.session_state.chargetime = st.number_input("Charge Time (s)", value=5000, step=100, key="ct_input")

# ══════════════════════════════════════════════════════════════════════════════
#  PREPARE UI PLACEHOLDER
# ══════════════════════════════════════════════════════════════════════════════
# This empty container is placed strictly after the inputs.
# It ensures all Heavy ML charts render perfectly in sequence WITHOUT flashing intermediately.
main_dashboard_container = st.container()

# ── Load Models ──────────────────────────────────────────────────────────────
# Models are cached via @st.cache_resource in inference.py
gru_model, dqn_model, scaler_X, scaler_y = load_models()

# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
# We separate inference from UI to allow caching to work effectively.
# Caching ensures that toggling the visibility does NOT rerun expensive SHAP/Calculations.
sequence = prepare_gru_sequence(
    scaler_X,
    st.session_state.ir, st.session_state.qc, st.session_state.qd,
    st.session_state.tavg, st.session_state.tmax, st.session_state.chargetime,
)

# Only show the spinner during the actual computation, not the layout render.
with st.spinner("Processing AI Insights..."):
    raw_soh = predict_soh(_gru_model=gru_model, _scaler_y=scaler_y, sequence=sequence)
    est_cycle = estimate_cycle(st.session_state.qd, st.session_state.qc)
    est_current = estimate_current(st.session_state.qc, st.session_state.chargetime)

    cal_soh = calibrate_soh(raw_soh, est_cycle)
    health_label = get_battery_health_label(cal_soh)

    rl_state = construct_rl_state(cal_soh, st.session_state.tavg, est_cycle, est_current)
    action, q_values = predict_rl_action(_dqn_model=dqn_model, state=rl_state)

    action_text_map = {0: "Decrease Charging", 1: "Maintain Charging", 2: "Increase Charging"}
    action_text = action_text_map.get(action, "Analysis Complete")

    shap_values, chosen_action, _ = compute_shap_values(_dqn_model=dqn_model, state=rl_state)
    gru_raw_inputs = [st.session_state.ir, st.session_state.qc, st.session_state.qd, st.session_state.tavg, st.session_state.tmax, st.session_state.chargetime]

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION PANELS — Side-by-Side
# ══════════════════════════════════════════════════════════════════════════════
with main_dashboard_container:
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        st.markdown('<div style="font-size: 1rem; font-weight: 700; color: #111827; margin-bottom: 15px;">🔋 Battery Health Status</div>', unsafe_allow_html=True)
        
        cal_pct = cal_soh * 100
        st.markdown(f'<div style="font-size: 2.2rem; font-weight: 800; color: #111827; text-align: center;">{cal_pct:.1f}%</div>', unsafe_allow_html=True)
        
        gauge_color = {"Healthy": "#10b981", "Moderate": "#f59e0b", "Degrading": "#f97316", "Severely Degraded": "#ef4444"}.get(health_label, "#f59e0b")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=cal_pct,
            number={"suffix": "%", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [{"range": [0, 70], "color": "#fee2e2"}, {"range": [70, 90], "color": "#fef3c7"}, {"range": [90, 100], "color": "#d1fae5"}],
            }
        ))
        fig_g.update_layout(height=180, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, width="stretch", key="battery_health_gauge")

    with p_col2:
        st.markdown('<div style="font-size: 1rem; font-weight: 700; color: #111827; margin-bottom: 15px;">⚡ AI Charging Decision</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div style="text-align:center; font-weight:800; color: #2563eb; font-size: 1.2rem; margin-bottom: 20px;">{action_text}</div>', unsafe_allow_html=True)
        
        fig_q = go.Figure(data=[go.Bar(
            x=["Decrease", "Maintain", "Increase"], y=q_values.tolist(),
            marker_color=["#ef4444", "#f59e0b", "#10b981"]
        )])
        fig_q.update_layout(height=220, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_q, width="stretch", key="ai_decision_bar_chart")

    # ══════════════════════════════════════════════════════════════════════════════
    #  EXPLAINABILITY — 3×3 Grid
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown('<div style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-bottom: 5px;">📊 Model Explainability (SHAP)</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6b7280; font-size: 0.85rem; margin-bottom: 20px;">Understanding feature impact on AI policy</p>', unsafe_allow_html=True)

    shap_rows = [st.columns(3) for _ in range(3)]
    plots = [
        ("Global Importance", lambda: generate_feature_importance_plot(shap_values, chosen_action=chosen_action)),
        ("Distribution", lambda: generate_shap_distribution_plot(shap_values, rl_state, chosen_action=chosen_action)),
        ("Impact Map", lambda: generate_shap_heatmap(shap_values)),
        ("Temp Dependency", lambda: generate_temperature_dependence(dqn_model, rl_state)),
        ("Cycle Dependency", lambda: generate_cycle_dependence(dqn_model, rl_state)),
        ("Current Dependency", lambda: generate_current_dependence(dqn_model, rl_state)),
        ("Action Influence", lambda: generate_action_influence_plot(shap_values)),
        ("Decision Ranking", lambda: generate_feature_ranking_plot(shap_values)),
        ("Combined XAI", lambda: generate_combined_xai_plot(shap_values, rl_state, gru_raw_inputs, chosen_action))
    ]

    for i, (title, plot_func) in enumerate(plots):
        with shap_rows[i//3][i%3]:
            st.markdown(f'<p style="font-size: 0.8rem; font-weight: 600; color: #4b5563; margin-bottom: 5px;">{title}</p>', unsafe_allow_html=True)
            try:
                fig = plot_func()
                st.pyplot(fig, width="stretch")
            except: st.warning("Graph unavailable")

    # ══════════════════════════════════════════════════════════════════════════════
    #  AI REASONING
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown('<div style="font-size: 1.1rem; font-weight: 700; color: #111827; margin-bottom: 15px; margin-top: 20px;">🤖 AI Reasoning Engine</div>', unsafe_allow_html=True)
    reasoning = generate_reasoning_text(action, cal_soh, st.session_state.tavg, est_cycle, est_current)
    st.markdown(f'<div style="background: #f9fafb; padding: 20px; border-radius: 12px; border: 1px solid #e5e7eb; color: #374151; line-height: 1.6; font-size: 0.95rem;">{reasoning}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div style="text-align: center; color: #9ca3af; font-size: 0.8rem; padding: 20px;">EV Battery AI Management • Explained by SHAP • GRU + Double DQN</div>', unsafe_allow_html=True)
