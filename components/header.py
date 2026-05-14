"""
header.py — Header and footer components for the EV-BMS Dashboard.
"""

import streamlit as st


def render_header():
    """Render the dashboard header card with title and subtitle."""
    st.markdown(
        """
        <div class="dashboard-header">
            <h1 class="dashboard-title">⚡ Explainable AI EV Battery Management Dashboard</h1>
            <p class="dashboard-subtitle">
                AI-driven battery health prediction and charging optimization | GRU + Double DQN + SHAP
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    """Render the fixed footer."""
    st.markdown(
        """
        <div class="footer">
            EV Battery AI Management – Explained by SHAP · GRU + Double DQN
        </div>
        """,
        unsafe_allow_html=True,
    )
