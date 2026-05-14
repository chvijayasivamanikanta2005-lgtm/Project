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

import joblib
import streamlit as st
import keras

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

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