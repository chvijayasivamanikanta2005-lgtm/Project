"""
inputs.py — Battery sensor input components for the EV-BMS Dashboard.
"""

import streamlit as st


def render_sensor_inputs():
    """
    Render the battery sensor input panel in a 3-column responsive layout.
    Returns a dict of the current input values.
    """
    st.markdown(
        '<div class="section-title">🔌 Battery Sensor Inputs</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        ir = st.number_input(
            "Internal Resistance (Ω)",
            value=0.04,
            step=0.001,
            format="%.4f",
            key="ir_input",
        )
        tavg = st.number_input(
            "Average Temp (°C)",
            value=30.0,
            step=0.1,
            format="%.1f",
            key="tavg_input",
        )

    with col2:
        qc = st.number_input(
            "Charge Capacity (Ah)",
            value=1.5,
            step=0.1,
            format="%.2f",
            key="qc_input",
        )
        tmax = st.number_input(
            "Maximum Temp (°C)",
            value=35.0,
            step=0.1,
            format="%.1f",
            key="tmax_input",
        )

    with col3:
        qd = st.number_input(
            "Discharge Capacity (Ah)",
            value=1.5,
            step=0.1,
            format="%.2f",
            key="qd_input",
        )
        chargetime = st.number_input(
            "Charge Time (s)",
            value=5000,
            step=100,
            key="ct_input",
        )

    return {
        "ir": ir,
        "qc": qc,
        "qd": qd,
        "tavg": tavg,
        "tmax": tmax,
        "chargetime": chargetime,
    }
