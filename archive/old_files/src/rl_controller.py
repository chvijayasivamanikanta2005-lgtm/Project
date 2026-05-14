import numpy as np
import streamlit as st

ACTION_LABELS = {
    0: "Decrease Charging",
    1: "Maintain Charging",
    2: "Increase Charging",
}

# Calibration bias to correct the trained model's bias against action 0.
# The model's raw Q-values for action 0 are systematically ~100-200 points
# below the others. This bias brings action 0 into a competitive range
# so that extreme conditions (high temp, low SoH, high cycle) can trigger it.
Q_BIAS = np.array([150.0, 0.0, -30.0])

def construct_rl_state(soh, temperature, cycle, current):
    return np.array([[soh, temperature, cycle, current]])

@st.cache_data(show_spinner=False)
def predict_rl_action(_dqn_model, state):
    q_values = _dqn_model.predict(state, verbose=0)
    calibrated = q_values[0] + Q_BIAS
    action = int(np.argmax(calibrated))
    return action, calibrated.astype(float)
