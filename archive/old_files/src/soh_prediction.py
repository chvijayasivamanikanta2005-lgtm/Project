import numpy as np
import streamlit as st

CYCLE_LIFE = 800

@st.cache_data(show_spinner=False)
def predict_soh(_gru_model, _scaler_y, sequence):
    pred_scaled = _gru_model.predict(sequence, verbose=0)
    pred = _scaler_y.inverse_transform(pred_scaled)
    return float(pred[0][0])

def calibrate_soh(raw_soh, cycle, cycle_life=CYCLE_LIFE):
    degradation = max(0.0, 1.0 - cycle / cycle_life)
    soh_final = raw_soh * (0.7 + 0.3 * degradation)
    return float(np.clip(soh_final, 0.5, 1.0))

def get_battery_health_label(soh):
    if soh > 0.90:
        return "Healthy"
    elif soh > 0.80:
        return "Moderate"
    elif soh > 0.70:
        return "Degrading"
    else:
        return "Severely Degraded"
