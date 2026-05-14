"""
app.py — Explainable AI EV Battery Management Dashboard
Main application entrypoint.
"""

import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable AI EV Battery Management Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
with open("assets/styles.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Imports ──────────────────────────────────────────────────────────────────
from utils.inference import load_models, predict_soh, predict_rl_action
from utils.preprocessing import (
    prepare_gru_sequence,
    estimate_cycle,
    estimate_current,
    calibrate_soh,
    get_battery_health_label,
    construct_rl_state,
)
from utils.shap_utils import compute_shap_values

from components.header import render_header, render_footer
from components.inputs import render_sensor_inputs
from components.soh_gauge import render_soh_panel
from components.rl_decision import render_rl_panel
from components.shap_charts import render_shap_section
from components.reasoning import render_reasoning


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
render_header()


# ══════════════════════════════════════════════════════════════════════════════
#  BATTERY SENSOR INPUTS
# ══════════════════════════════════════════════════════════════════════════════
inputs = render_sensor_inputs()


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════
gru_model, dqn_model, scaler_X, scaler_y = load_models()


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
sequence = prepare_gru_sequence(
    scaler_X,
    inputs["ir"], inputs["qc"], inputs["qd"],
    inputs["tavg"], inputs["tmax"], inputs["chargetime"],
)

with st.spinner("Processing AI Insights..."):
    # SoH Prediction
    raw_soh = predict_soh(_gru_model=gru_model, _scaler_y=scaler_y, sequence=sequence)
    est_cycle = estimate_cycle(inputs["qd"], inputs["qc"])
    est_current = estimate_current(inputs["qc"], inputs["chargetime"])

    cal_soh = calibrate_soh(raw_soh, est_cycle)
    health_label = get_battery_health_label(cal_soh)

    # RL Decision
    rl_state = construct_rl_state(cal_soh, inputs["tavg"], est_cycle, est_current)
    action, q_values = predict_rl_action(_dqn_model=dqn_model, state=rl_state)

    # SHAP Explainability
    shap_values, chosen_action, _ = compute_shap_values(_dqn_model=dqn_model, state=rl_state)
    gru_raw_inputs = [
        inputs["ir"], inputs["qc"], inputs["qd"],
        inputs["tavg"], inputs["tmax"], inputs["chargetime"],
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION PANELS (SoH + RL Decision)
# ══════════════════════════════════════════════════════════════════════════════
p_col1, p_col2 = st.columns(2)

with p_col1:
    render_soh_panel(cal_soh, health_label)

with p_col2:
    render_rl_panel(action, q_values)


# ══════════════════════════════════════════════════════════════════════════════
#  SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
render_shap_section(shap_values, chosen_action, rl_state, dqn_model, gru_raw_inputs)


# ══════════════════════════════════════════════════════════════════════════════
#  AI REASONING ENGINE
# ══════════════════════════════════════════════════════════════════════════════
render_reasoning(action, cal_soh, inputs["tavg"], est_cycle, est_current)


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
render_footer()
