"""
reasoning.py — AI Reasoning Engine display component for the EV-BMS Dashboard.
"""

import streamlit as st
from utils.shap_utils import generate_reasoning_text


def render_reasoning(action, cal_soh, tavg, est_cycle, est_current):
    """Render the AI reasoning engine section."""
    st.markdown(
        '<div class="section-title">🤖 AI Reasoning Engine</div>',
        unsafe_allow_html=True,
    )

    reasoning = generate_reasoning_text(action, cal_soh, tavg, est_cycle, est_current)

    st.markdown(
        f'<div class="reasoning-card">{reasoning}</div>',
        unsafe_allow_html=True,
    )
