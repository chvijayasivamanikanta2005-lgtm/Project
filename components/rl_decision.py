"""
rl_decision.py — RL charging decision display for the EV-BMS Dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np


ACTION_MAP = {
    0: ("Decrease Charging", "#ef4444", "decrease"),
    1: ("Maintain Charging", "#f59e0b", "maintain"),
    2: ("Increase Charging", "#10b981", "increase"),
}


def _create_decision_bar(q_values, selected_action):
    """Create a Plotly bar chart showing Q-values for all 3 actions."""
    labels = ["Decrease", "Maintain", "Increase"]
    colors = ["#ef4444", "#f59e0b", "#10b981"]

    # Highlight the selected action with full opacity, dim others
    opacities = [0.4 if i != selected_action else 1.0 for i in range(3)]
    bar_colors = [
        f"rgba({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)}, {o})"
        for c, o in zip(colors, opacities)
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=q_values.tolist(),
                marker_color=bar_colors,
                marker_line=dict(
                    color=[colors[selected_action] if i == selected_action else "#d1d5db" for i in range(3)],
                    width=[2 if i == selected_action else 0.5 for i in range(3)],
                ),
                text=[f"{v:.1f}" for v in q_values],
                textposition="outside",
                textfont=dict(size=11, color="#374151"),
            )
        ]
    )
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
        yaxis=dict(
            showgrid=True,
            gridcolor="#e5e7eb",
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor="#d1d5db",
            title=dict(text="Q-Value", font=dict(size=11, color="#6b7280")),
            tickfont=dict(size=10, color="#6b7280"),
        ),
        xaxis=dict(
            tickfont=dict(size=12, color="#374151", weight=600),
        ),
        bargap=0.35,
    )
    return fig


def render_rl_panel(action, q_values):
    """Render the RL charging decision panel with Q-value bars and action label."""
    action_text, action_color, action_class = ACTION_MAP.get(action, ("Unknown", "#6b7280", "maintain"))

    st.markdown(
        '<div class="card-title">⚡ AI Charging Decision</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 12px;">
            <span class="action-badge {action_class}">{action_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = _create_decision_bar(q_values, action)
    st.plotly_chart(fig, use_container_width=True, key="ai_decision_bar_chart")
