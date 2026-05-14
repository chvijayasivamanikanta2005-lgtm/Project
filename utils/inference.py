"""
inference.py — Model loading and prediction pipeline for EV-BMS Dashboard.
"""

import os

# -----------------------------------------------------------
# Environment setup
# -----------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import joblib
import streamlit as st
import keras

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

ACTION_LABELS = {
    0: "Decrease Charging",
    1: "Maintain Charging",
    2: "Increase Charging",
}


# -----------------------------------------------------------
# Safe model loader
# -----------------------------------------------------------
def _load_model_safe(path):
    """Load Keras models safely using standalone Keras 3."""
    try:
        model = keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"Failed to load model at {path}: {e}")
        return None
    return model


# -----------------------------------------------------------
# Cached model loading
# -----------------------------------------------------------
@st.cache_resource
def load_models():
    """Load all models and scalers (cached)."""
    if hasattr(keras, "backend") and hasattr(keras.backend, "clear_session"):
        keras.backend.clear_session()

    gru_model = _load_model_safe(os.path.join(MODEL_DIR, "gru_soh_model.keras"))
    dqn_model = _load_model_safe(os.path.join(MODEL_DIR, "double_dqn_calibrated.keras"))
    scaler_X = joblib.load(os.path.join(MODEL_DIR, "gru_scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(MODEL_DIR, "gru_scaler_y.pkl"))

    return gru_model, dqn_model, scaler_X, scaler_y


# -----------------------------------------------------------
# SoH prediction
# -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def predict_soh(_gru_model, _scaler_y, sequence):
    """Predict SoH using the GRU model."""
    pred_scaled = _gru_model.predict(sequence, verbose=0)
    pred = _scaler_y.inverse_transform(pred_scaled)
    return float(pred[0][0])


# -----------------------------------------------------------
# RL action prediction — physics-corrected
# -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def predict_rl_action(_dqn_model, state):
    """
    Predict RL charging action with physics-corrected Q-values.

    The trained DQN model's raw action indices are inverted relative to
    battery physics:
      - Raw index 2 gets highest Q for degraded/hot states (protective)
      - Raw index 0 gets highest Q for healthy/cool states (aggressive)

    Corrections applied:
      1. Swap indices 0↔2 so our output maps correctly:
         0 = Decrease, 1 = Maintain, 2 = Increase
      2. Temperature correction: hot → boost Decrease, cold → boost Increase
      3. SoH correction: low SoH → boost Decrease, high SoH → boost Increase
    """
    soh, temp, cycle, current = state[0]

    # Raw model Q-values
    q_raw = _dqn_model.predict(state, verbose=0)[0]

    # Step 1: Swap action indices 0↔2 to align with correct physics
    # Model: [Q_aggressive, Q_maintain, Q_protective]
    # Ours:  [Q_decrease,   Q_maintain, Q_increase  ]
    q_corrected = np.array([q_raw[2], q_raw[1], q_raw[0]])

    # Step 2: Temperature correction
    # Normalized: -1.0 at 0°C, 0.0 at 25°C, +1.0 at 50°C
    temp_factor = np.clip((temp - 25.0) / 25.0, -1.0, 1.0)
    temp_correction = np.array([
        temp_factor * 70.0,     # Decrease boosted when hot
        0.0,                    # Maintain stays neutral
        -temp_factor * 70.0,    # Increase boosted when cold
    ])

    # Step 3: SoH correction
    # Normalized: -1.0 at SoH=0.60, 0.0 at SoH=0.80, +1.0 at SoH=1.00
    soh_factor = np.clip((soh - 0.80) / 0.20, -1.0, 1.0)
    soh_correction = np.array([
        -soh_factor * 30.0,     # Decrease boosted for low SoH
        0.0,                    # Maintain stays neutral
        soh_factor * 30.0,      # Increase boosted for high SoH
    ])

    q_final = q_corrected + temp_correction + soh_correction
    action = int(np.argmax(q_final))

    return action, q_final.astype(float)
