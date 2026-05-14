"""
soh_gauge.py — SoH prediction display with Plotly gauge for the EV-BMS Dashboard.
"""

import streamlit as st
import plotly.graph_objects as go


def _create_gauge(soh_pct, health_label):
    """Create a Plotly gauge chart for battery health."""
    gauge_color = {
        "Healthy": "#10b981",
        "Moderate": "#f59e0b",
        "Degrading": "#f97316",
        "Severely Degraded": "#ef4444",
    }.get(health_label, "#f59e0b")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=soh_pct,
            number={"suffix": "%", "font": {"size": 28, "color": "#1e293b"}},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"size": 10, "color": "#6b7280"}},
                "bar": {"color": gauge_color, "thickness": 0.75},
                "steps": [
                    {"range": [0, 70], "color": "#fee2e2"},
                    {"range": [70, 90], "color": "#fef3c7"},
                    {"range": [90, 100], "color": "#d1fae5"},
                ],
                "threshold": {
                    "line": {"color": "#1e293b", "width": 2},
                    "thickness": 0.8,
                    "value": soh_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def render_soh_panel(cal_soh, health_label):
    """Render the SoH prediction panel with percentage and gauge."""
    soh_pct = cal_soh * 100

    st.markdown(
        '<div class="card-title">🔋 Battery Health Status</div>',
        unsafe_allow_html=True,
    )

    # Health badge color
    badge_colors = {
        "Healthy": ("#d1fae5", "#059669"),
        "Moderate": ("#fef3c7", "#d97706"),
        "Degrading": ("#ffedd5", "#ea580c"),
        "Severely Degraded": ("#fee2e2", "#dc2626"),
    }
    bg, fg = badge_colors.get(health_label, ("#f3f4f6", "#4b5563"))

    st.markdown(
        f"""
        <div style="text-align: center;">
            <div class="soh-value">{soh_pct:.1f}%</div>
            <div style="margin: 8px 0;">
                <span class="status-badge" style="background: {bg}; color: {fg};">{health_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = _create_gauge(soh_pct, health_label)
    st.plotly_chart(fig, use_container_width=True, key="battery_health_gauge")
