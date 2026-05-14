"""
shap_utils.py — SHAP computation and explanation utilities for EV-BMS Dashboard.
"""

import numpy as np
import streamlit as st

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------
RL_FEATURES = ["SoH", "Temp", "Cycle", "Current"]
GRU_FEATURES = ["IR", "QC", "QD", "Tavg", "Tmax", "ChargeTime"]
ACTIONS = ["Decrease", "Maintain", "Increase"]


# -----------------------------------------------------------
# SHAP value helpers
# -----------------------------------------------------------
def safe_shap_for_action(shap_values, chosen_action):
    """Extract SHAP values for chosen action, handling various shapes."""
    if isinstance(shap_values, list):
        sv = np.array(shap_values[chosen_action])
        return sv[0] if sv.ndim >= 2 else sv
    sv = np.array(shap_values)
    if sv.ndim == 3:
        return sv[0, :, chosen_action]
    elif sv.ndim == 2:
        return sv[0]
    return sv


def all_actions_shap(shap_values):
    """Return (3, num_features) matrix of SHAP values across actions."""
    if isinstance(shap_values, list):
        return np.array([sv[0] for sv in shap_values])
    sv = np.array(shap_values)
    if sv.ndim == 3:
        return sv[0].T
    return np.stack([sv[0]] * 3)


# -----------------------------------------------------------
# Core SHAP Computation (cached)
# -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_shap_values(_dqn_model, state, feature_names=None):
    """Compute SHAP values for the DQN model given a state."""
    import shap

    if feature_names is None:
        feature_names = RL_FEATURES

    def predict_fn(x):
        return _dqn_model(x, training=False).numpy()

    background = np.array([[1.0, 25.0, 100.0, 50.0]])
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(state, silent=True)

    q_values = predict_fn(state)[0]
    chosen_action = int(np.argmax(q_values))

    if not isinstance(shap_values, list):
        sv = np.array(shap_values)
        if sv.ndim == 3:
            shap_values = [sv[:, :, i] for i in range(sv.shape[2])]
        else:
            shap_values = [shap_values]

    return shap_values, chosen_action, q_values


# -----------------------------------------------------------
# AI Reasoning Text
# -----------------------------------------------------------
def generate_reasoning_text(action, soh, temp, cycle, current):
    """Generate human-readable AI reasoning text for the charging decision."""
    action_map = {
        0: "Decrease Charging",
        1: "Maintain Charging",
        2: "Increase Charging",
    }
    action_str = action_map.get(action, "Unknown")
    reasons = []

    if action == 0:  # Decrease
        if temp > 35:
            reasons.append(
                f"Battery temperature is elevated ({temp:.1f}°C), requiring "
                "reduced current to prevent thermal runaway."
            )
        if soh < 0.70:
            reasons.append(
                f"Calibrated SoH is critically low ({soh:.2f}), indicating "
                "severe degradation vulnerability."
            )
        if cycle > 500:
            reasons.append(
                f"High cycle count ({cycle}) indicates significant aging and "
                "increased degradation risk."
            )
        if current > 80:
            reasons.append(
                f"Charging current ({current:.1f}A) is high; reducing it "
                "extends remaining useful life."
            )
        if not reasons:
            reasons.append(
                "The controller decreases charging current because battery "
                "cycle count is high and temperature conditions indicate "
                "potential degradation risk."
            )
    elif action == 1:  # Maintain
        if 20 <= temp <= 35:
            reasons.append(
                f"Battery temperature ({temp:.1f}°C) is within the optimal "
                "operating window."
            )
        if soh >= 0.70:
            reasons.append(
                f"Calibrated SoH ({soh:.2f}) supports current charging load."
            )
        if not reasons:
            reasons.append(
                "The current charging regime is optimal for minimising "
                "degradation while maintaining performance."
            )
    elif action == 2:  # Increase
        if temp < 30:
            reasons.append(
                f"Battery temperature ({temp:.1f}°C) is cool enough to "
                "safely accept a higher charge rate."
            )
        if soh > 0.80:
            reasons.append(
                f"Battery health is strong ({soh:.2f}), enabling faster charging."
            )
        if current < 40:
            reasons.append(
                f"Charging current ({current:.1f}A) is low and can be "
                "safely increased."
            )
        if not reasons:
            reasons.append(
                "Battery parameters are under safe thresholds, allowing "
                "accelerated charging speed."
            )

    text = f"<strong>Decision: {action_str}</strong><br><br>\n"
    text += f"The AI controller recommended to <strong>{action_str.lower()}</strong> based on:<ul>\n"
    for r in reasons:
        text += f"<li>{r}</li>\n"
    text += "</ul>"
    return text
