import sys
import numpy as np
import keras

model_path = "/Users/saimani/Desktop/FRONT-END/Project_Deploy/models/double_dqn_calibrated.keras"
model = keras.models.load_model(model_path, compile=False)

s = np.array([[1.0, 15.0, 0, 0.1]]) # Perfect SoH, 15C, 0 Cycle, 0.1A Current
q_vals = model.predict(s, verbose=0)[0]
best_action = int(np.argmax(q_vals))
print(f"Ultra-low current Q-values: {q_vals}")
print(f"Best Action: {best_action}")
