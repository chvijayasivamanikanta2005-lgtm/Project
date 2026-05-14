import numpy as np

GRU_SEQUENCE_LENGTH = 20
CYCLE_LIFE = 800

def prepare_gru_sequence(scaler_X, ir, qc, qd, tavg, tmax, chargetime):
    raw = np.array([[ir, qc, qd, tavg, tmax, chargetime]])
    scaled = scaler_X.transform(raw)
    sequence = np.tile(scaled, (GRU_SEQUENCE_LENGTH, 1))
    return np.expand_dims(sequence, axis=0)

def estimate_cycle(qd, qc, cycle_life=CYCLE_LIFE):
    if qc <= 0:
        return 0
    ratio = max(0.0, 1.0 - qd / qc)
    return int(ratio * cycle_life)

def estimate_current(qc, chargetime):
    if chargetime <= 0:
        return 0.0
    return round(qc / (chargetime / 3600.0), 2)
