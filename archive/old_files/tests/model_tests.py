import sys
import os
import numpy as np

# ensure TF doesn't use GPU and spam logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras

# Load the DQN model
model_path = "/Users/saimani/Desktop/FRONT-END/Project_Deploy/models/double_dqn_calibrated.keras"
try:
    model = keras.models.load_model(model_path, compile=False)
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

# Test cases
# Format: [SoH, Temp, Cycle, Current]
test_states = {
    "Perfect Battery, Cool, Low Current": [1.0, 20.0, 0, 5.0],
    "Degraded Battery, Hot, High Current": [0.60, 45.0, 800, 80.0],
    "Healthy Battery, Cool, High Current": [0.95, 25.0, 50, 100.0],
    "Degraded Battery, Cool, Low Current": [0.70, 20.0, 600, 10.0],
}

print("RL Action Labels to figure out what 0, 1, 2 actually are based on Q-values:")
for name, state in test_states.items():
    s = np.array([state])
    q_vals = model.predict(s, verbose=0)[0]
    best_action = int(np.argmax(q_vals))
    print(f"--- {name} ---")
    print(f"Input State: SoH={state[0]:.2f}, Temp={state[1]:.1f}, Cycle={state[2]}, Current={state[3]:.1f}")
    print(f"Q-values: {q_vals}")
    print(f"Best Action Index: {best_action}")
    print()
