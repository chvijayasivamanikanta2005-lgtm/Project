"""
preprocessing.py — Data preprocessing and feature engineering for EV-BMS Dashboard.
"""

import numpy as np

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------
GRU_SEQUENCE_LENGTH = 20
CYCLE_LIFE = 800


# -----------------------------------------------------------
# GRU sequence preparation
# -----------------------------------------------------------
def prepare_gru_sequence(scaler_X, ir, qc, qd, tavg, tmax, chargetime):
    """Create a scaled sequence window of length 20 for GRU input."""
    raw = np.array([[ir, qc, qd, tavg, tmax, chargetime]])
    scaled = scaler_X.transform(raw)
    sequence = np.tile(scaled, (GRU_SEQUENCE_LENGTH, 1))
    return np.expand_dims(sequence, axis=0)


# -----------------------------------------------------------
# Derived feature estimation
# -----------------------------------------------------------
def estimate_cycle(qd, qc, cycle_life=CYCLE_LIFE):
    """Estimate battery cycle count from discharge/charge capacity."""
    if qc <= 0:
        return 0
    ratio = max(0.0, 1.0 - qd / qc)
    return int(ratio * cycle_life)


def estimate_current(qc, chargetime):
    """Estimate charging current from charge capacity and charge time."""
    if chargetime <= 0:
        return 0.0
    return round(qc / (chargetime / 3600.0), 2)


# -----------------------------------------------------------
# SoH calibration
# -----------------------------------------------------------
def calibrate_soh(raw_soh, cycle, cycle_life=CYCLE_LIFE):
    """Calibrate raw SoH using degradation factor based on cycle count."""
    degradation = max(0.0, 1.0 - cycle / cycle_life)
    soh_final = raw_soh * (0.7 + 0.3 * degradation)
    return float(np.clip(soh_final, 0.5, 1.0))


def get_battery_health_label(soh):
    """Return a human-readable health label based on SoH value."""
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
    """Construct the state vector for the RL model."""
    return np.array([[soh, temperature, cycle, current]])
