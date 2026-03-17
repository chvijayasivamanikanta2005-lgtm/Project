"""
inference.py — Model loading and prediction pipeline for EV-BMS Dashboard v4.
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
import tensorflow as tf
import keras

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------
GRU_SEQUENCE_LENGTH = 20
GRU_NUM_FEATURES = 6
CYCLE_LIFE = 800

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

ACTION_LABELS = {
    0: "Decrease Charging",
    1: "Maintain Charging",
    2: "Increase Charging",
}

# -----------------------------------------------------------
# Safe model loader
# -----------------------------------------------------------
def load_model_safe(path):
    """
    Load Keras models safely using standalone Keras 3.
    """
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
    # In Keras 3, backend is often automatically managed, but clear session is still good if using TF backend
    if hasattr(keras, 'backend') and hasattr(keras.backend, 'clear_session'):
        keras.backend.clear_session()

    gru_model = load_model_safe(
        os.path.join(MODEL_DIR, "gru_soh_model.keras")
    )

    dqn_model = load_model_safe(
        os.path.join(MODEL_DIR, "double_dqn_calibrated.keras")
    )

    scaler_X = joblib.load(
        os.path.join(MODEL_DIR, "gru_scaler_X.pkl")
    )

    scaler_y = joblib.load(
        os.path.join(MODEL_DIR, "gru_scaler_y.pkl")
    )

    return gru_model, dqn_model, scaler_X, scaler_y


# -----------------------------------------------------------
# GRU sequence preparation
# -----------------------------------------------------------
def prepare_gru_sequence(scaler_X, ir, qc, qd, tavg, tmax, chargetime):

    raw = np.array([[ir, qc, qd, tavg, tmax, chargetime]])

    scaled = scaler_X.transform(raw)

    sequence = np.tile(scaled, (GRU_SEQUENCE_LENGTH, 1))

    return np.expand_dims(sequence, axis=0)


# -----------------------------------------------------------
# SoH prediction
# -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def predict_soh(_gru_model, _scaler_y, sequence):

    pred_scaled = _gru_model.predict(sequence, verbose=0)

    pred = _scaler_y.inverse_transform(pred_scaled)

    return float(pred[0][0])


# -----------------------------------------------------------
# Cycle estimation
# -----------------------------------------------------------
def estimate_cycle(qd, qc, cycle_life=CYCLE_LIFE):

    if qc <= 0:
        return 0

    ratio = max(0.0, 1.0 - qd / qc)

    return int(ratio * cycle_life)


# -----------------------------------------------------------
# Current estimation
# -----------------------------------------------------------
def estimate_current(qc, chargetime):

    if chargetime <= 0:
        return 0.0

    return round(qc / (chargetime / 3600.0), 2)


# -----------------------------------------------------------
# SoH calibration
# -----------------------------------------------------------
def calibrate_soh(raw_soh, cycle, cycle_life=CYCLE_LIFE):

    degradation = max(0.0, 1.0 - cycle / cycle_life)

    soh_final = raw_soh * (0.7 + 0.3 * degradation)

    return float(np.clip(soh_final, 0.5, 1.0))


# -----------------------------------------------------------
# Battery health classification
# -----------------------------------------------------------
def get_battery_health_label(soh):

    if soh > 0.90:
        return "Healthy"
    elif soh > 0.80:
        return "Moderate"
    elif soh > 0.70:
        return "Degrading"
    else:
        return "Severely Degraded"


# -----------------------------------------------------------
# RL state construction
# -----------------------------------------------------------
def construct_rl_state(soh, temperature, cycle, current):

    return np.array([[soh, temperature, cycle, current]])


# -----------------------------------------------------------
# RL action prediction
# -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def predict_rl_action(_dqn_model, state):

    q_values = _dqn_model.predict(state, verbose=0)

    action = int(np.argmax(q_values[0]))

    return action, q_values[0].astype(float)