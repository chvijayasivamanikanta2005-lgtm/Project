"""
shap_charts.py — All SHAP/XAI visualization components for the EV-BMS Dashboard.

Nine Plotly charts:
    1. Global Feature Importance
    2. SHAP Distribution Plot
    3. SHAP Heatmap
    4. Temperature Dependence
    5. Cycle Dependence
    6. Current Dependence
    7. RL Action Influence (grouped bar)
    8. Feature Ranking
    9. Combined GRU + RL XAI
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.shap_utils import (
    safe_shap_for_action,
    all_actions_shap,
    RL_FEATURES,
    GRU_FEATURES,
    ACTIONS,
)

# -----------------------------------------------------------
# Color palette
# -----------------------------------------------------------
PRIMARY = "#3b82f6"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER = "#ef4444"
MUTED = "#64748b"
ACT_COLORS = [DANGER, WARNING, SUCCESS]

_COMMON_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#fafbfc",
    font=dict(family="Inter, sans-serif", size=11, color="#374151"),
    margin=dict(l=50, r=20, t=40, b=40),
    height=300,
)


def _apply_layout(fig, title="", **overrides):
    """Apply a consistent layout to a Plotly figure."""
    layout = {**_COMMON_LAYOUT, "title": dict(text=title, font=dict(size=13, color="#1e293b"), x=0.5)}
    layout.update(overrides)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#e5e7eb", gridwidth=0.5, zeroline=True, zerolinecolor="#d1d5db")
    fig.update_yaxes(gridcolor="#e5e7eb", gridwidth=0.5, zeroline=True, zerolinecolor="#d1d5db")
    return fig


# -----------------------------------------------------------
# 1. Global Feature Importance
# -----------------------------------------------------------
def generate_feature_importance_plot(shap_values, chosen_action=0):
    vals = np.abs(safe_shap_for_action(shap_values, chosen_action))
    order = np.argsort(vals)
    colors = [PRIMARY, "#6366f1", "#8b5cf6", "#a78bfa"]

    fig = go.Figure(
        go.Bar(
            x=vals[order],
            y=np.array(RL_FEATURES)[order],
            orientation="h",
            marker_color=[colors[i % 4] for i in range(len(order))],
            text=[f"{v:.4f}" for v in vals[order]],
            textposition="outside",
            textfont=dict(size=10, color=MUTED),
        )
    )
    return _apply_layout(fig, "SoH Prediction Feature Importance",
                         xaxis=dict(title=dict(text="Mean |SHAP Value|")))


# -----------------------------------------------------------
# 2. SHAP Distribution Plot
# -----------------------------------------------------------
def generate_shap_distribution_plot(shap_values, state, chosen_action=0):
    sv = safe_shap_for_action(shap_values, chosen_action)
    order = np.argsort(np.abs(sv))[::-1]
    colors = [SUCCESS if v >= 0 else DANGER for v in sv[order]]
    labels = [f"{RL_FEATURES[i]} = {state[0][i]:.2f}" for i in order]

    fig = go.Figure(
        go.Bar(
            x=sv[order],
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in sv[order]],
            textposition="outside",
            textfont=dict(size=10),
        )
    )
    return _apply_layout(fig, "SHAP Distribution Plot",
                         xaxis=dict(title=dict(text="SHAP Value")))


# -----------------------------------------------------------
# 3. SHAP Heatmap
# -----------------------------------------------------------
def generate_shap_heatmap(shap_values):
    matrix = all_actions_shap(shap_values)
    vmax = max(abs(matrix.min()), abs(matrix.max())) or 1.0

    # Build annotation text
    annotations = []
    for i in range(3):
        for j in range(len(RL_FEATURES)):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=f"{matrix[i, j]:.3f}",
                    showarrow=False,
                    font=dict(size=11, color="#1e293b", weight="bold"),
                )
            )

    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=RL_FEATURES,
            y=ACTIONS,
            colorscale=[[0, DANGER], [0.5, "#fef3c7"], [1, SUCCESS]],
            zmin=-vmax, zmax=vmax,
            showscale=True,
            colorbar=dict(thickness=12, tickfont=dict(size=9)),
        )
    )
    fig.update_layout(annotations=annotations)
    return _apply_layout(fig, "SHAP Heatmap (All Actions)")


# -----------------------------------------------------------
# 4–6. Dependence Plots
# -----------------------------------------------------------
def _dependence(dqn_model, state, idx, name, rng, n=30):
    xs = np.linspace(rng[0], rng[1], n)
    curves = {a: [] for a in range(3)}
    for v in xs:
        s = state.copy()
        s[0, idx] = v
        q = dqn_model(s, training=False).numpy()[0]
        for a in range(3):
            curves[a].append(q[a])

    fig = go.Figure()
    for a in range(3):
        fig.add_trace(
            go.Scatter(
                x=xs, y=curves[a],
                mode="lines",
                name=ACTIONS[a],
                line=dict(color=ACT_COLORS[a], width=2.5),
                opacity=0.85,
            )
        )
    fig.add_vline(x=state[0, idx], line_dash="dash", line_color=PRIMARY, line_width=1.5,
                  annotation_text=f"Current ({state[0, idx]:.1f})", annotation_font_size=9)
    return _apply_layout(fig, f"{name} Dependence",
                         xaxis=dict(title=dict(text=name)),
                         yaxis=dict(title=dict(text="Q-Value")),
                         legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", font=dict(size=9)))


def generate_temperature_dependence(dqn_model, state):
    return _dependence(dqn_model, state, 1, "Temperature (°C)", (5, 55))


def generate_cycle_dependence(dqn_model, state):
    return _dependence(dqn_model, state, 2, "Cycle Count", (1, 2000))


def generate_current_dependence(dqn_model, state):
    return _dependence(dqn_model, state, 3, "Charging Current (A)", (0, 150))


# -----------------------------------------------------------
# 7. RL Action Influence (grouped bar)
# -----------------------------------------------------------
def generate_action_influence_plot(shap_values):
    matrix = all_actions_shap(shap_values)

    fig = go.Figure()
    for a in range(3):
        fig.add_trace(
            go.Bar(
                name=ACTIONS[a],
                x=RL_FEATURES,
                y=matrix[a],
                marker_color=ACT_COLORS[a],
                opacity=0.85,
            )
        )
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.1)
    return _apply_layout(fig, "RL Charging Decision Feature Influence",
                         yaxis=dict(title=dict(text="SHAP Value")),
                         legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", font=dict(size=9)))


# -----------------------------------------------------------
# 8. Feature Ranking
# -----------------------------------------------------------
def generate_feature_ranking_plot(shap_values):
    matrix = np.abs(all_actions_shap(shap_values))
    agg = matrix.mean(axis=0)
    order = np.argsort(agg)[::-1]
    colors = [PRIMARY, "#6366f1", "#8b5cf6", "#a78bfa"]

    fig = go.Figure(
        go.Bar(
            x=agg[order],
            y=np.array(RL_FEATURES)[order],
            orientation="h",
            marker_color=[colors[i % 4] for i in range(len(order))],
            text=[f"{v:.4f}" for v in agg[order]],
            textposition="outside",
            textfont=dict(size=10, color=MUTED),
        )
    )
    return _apply_layout(fig, "Feature Ranking",
                         xaxis=dict(title=dict(text="Aggregate |SHAP| (all actions)")))


# -----------------------------------------------------------
# 9. Combined GRU + RL XAI
# -----------------------------------------------------------
def generate_combined_xai_plot(shap_values, state, gru_inputs, chosen_action=0):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("GRU Input Features", "RL Decision SHAP"),
                        horizontal_spacing=0.15)

    # Left: GRU input magnitudes
    gru_vals = np.array(gru_inputs).flatten()
    gru_norm = gru_vals / gru_vals.max() if gru_vals.max() > 0 else gru_vals

    fig.add_trace(
        go.Bar(
            x=gru_norm,
            y=GRU_FEATURES,
            orientation="h",
            marker_color=PRIMARY,
            opacity=0.8,
            text=[f"{rv:.2f}" for rv in gru_vals],
            textposition="outside",
            textfont=dict(size=9, color=MUTED),
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Right: RL SHAP
    sv = safe_shap_for_action(shap_values, chosen_action)
    colors = [SUCCESS if v >= 0 else DANGER for v in sv]

    fig.add_trace(
        go.Bar(
            x=sv,
            y=RL_FEATURES,
            orientation="h",
            marker_color=colors,
            showlegend=False,
            text=[f"{v:.4f}" for v in sv],
            textposition="outside",
            textfont=dict(size=9),
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Normalised Magnitude", row=1, col=1, gridcolor="#e5e7eb")
    fig.update_xaxes(title_text="SHAP Value", row=1, col=2, gridcolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb")

    return _apply_layout(fig, "", height=320, margin=dict(l=60, r=20, t=50, b=40))


# -----------------------------------------------------------
# Render all SHAP charts in a 3×3 grid
# -----------------------------------------------------------
def render_shap_section(shap_values, chosen_action, rl_state, dqn_model, gru_raw_inputs):
    """Render the complete SHAP explainability section."""
    st.markdown(
        """
        <div class="section-title">📊 Model Explainability (SHAP)</div>
        <p style="color: #6b7280; font-size: 0.85rem; margin-bottom: 20px;">
            Understanding feature impact on AI policy
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Row 1
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.markdown('<p class="chart-label">Global Importance</p>', unsafe_allow_html=True)
        try:
            fig = generate_feature_importance_plot(shap_values, chosen_action=chosen_action)
            st.plotly_chart(fig, use_container_width=True, key="shap_importance")
        except Exception:
            st.warning("Chart unavailable")

    with r1c2:
        st.markdown('<p class="chart-label">Distribution</p>', unsafe_allow_html=True)
        try:
            fig = generate_shap_distribution_plot(shap_values, rl_state, chosen_action=chosen_action)
            st.plotly_chart(fig, use_container_width=True, key="shap_distribution")
        except Exception:
            st.warning("Chart unavailable")

    with r1c3:
        st.markdown('<p class="chart-label">Impact Map</p>', unsafe_allow_html=True)
        try:
            fig = generate_shap_heatmap(shap_values)
            st.plotly_chart(fig, use_container_width=True, key="shap_heatmap")
        except Exception:
            st.warning("Chart unavailable")

    # Row 2
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.markdown('<p class="chart-label">Temp Dependency</p>', unsafe_allow_html=True)
        try:
            fig = generate_temperature_dependence(dqn_model, rl_state)
            st.plotly_chart(fig, use_container_width=True, key="shap_temp")
        except Exception:
            st.warning("Chart unavailable")

    with r2c2:
        st.markdown('<p class="chart-label">Cycle Dependency</p>', unsafe_allow_html=True)
        try:
            fig = generate_cycle_dependence(dqn_model, rl_state)
            st.plotly_chart(fig, use_container_width=True, key="shap_cycle")
        except Exception:
            st.warning("Chart unavailable")

    with r2c3:
        st.markdown('<p class="chart-label">Current Dependency</p>', unsafe_allow_html=True)
        try:
            fig = generate_current_dependence(dqn_model, rl_state)
            st.plotly_chart(fig, use_container_width=True, key="shap_current")
        except Exception:
            st.warning("Chart unavailable")

    # Row 3
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        st.markdown('<p class="chart-label">Action Influence</p>', unsafe_allow_html=True)
        try:
            fig = generate_action_influence_plot(shap_values)
            st.plotly_chart(fig, use_container_width=True, key="shap_action_influence")
        except Exception:
            st.warning("Chart unavailable")

    with r3c2:
        st.markdown('<p class="chart-label">Decision Ranking</p>', unsafe_allow_html=True)
        try:
            fig = generate_feature_ranking_plot(shap_values)
            st.plotly_chart(fig, use_container_width=True, key="shap_ranking")
        except Exception:
            st.warning("Chart unavailable")

    with r3c3:
        st.markdown('<p class="chart-label">Combined XAI</p>', unsafe_allow_html=True)
        try:
            fig = generate_combined_xai_plot(shap_values, rl_state, gru_raw_inputs, chosen_action)
            st.plotly_chart(fig, use_container_width=True, key="shap_combined")
        except Exception:
            st.warning("Chart unavailable")
