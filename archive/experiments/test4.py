import sys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras

model_path = "/Users/saimani/Desktop/FRONT-END/Project_Deploy/models/double_dqn_calibrated.keras"
model = keras.models.load_model(model_path, compile=False)

# User's inputs:
soh = 0.932
temp = 20.0
cycle = 0   # qd=2.5, qc=2.5
current = 9.0 # 2.5/(1000/3600)

s = np.array([[soh, temp, cycle, current]])
q_vals = model.predict(s, verbose=0)[0]
print("Q-values:", q_vals)

# Action 0 = Increase
# Action 1 = Maintain
# Action 2 = Decrease
for i, name in enumerate(["Increase", "Maintain", "Decrease"]):
    print(f"Action {i} ({name}): Q = {q_vals[i]:.2f}")
