import sys
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras

model_path = "/Users/saimani/Desktop/FRONT-END/Project_Deploy/models/double_dqn_calibrated.keras"
model = keras.models.load_model(model_path, compile=False)

# Let's grid search a bit
for soh in [0.99, 0.95, 0.90]:
    for temp in [15.0, 20.0, 25.0]:
        for cycle in [0, 10, 50]:
            for current in [1.0, 5.0, 10.0]:
                s = np.array([[soh, temp, cycle, current]])
                q_vals = model.predict(s, verbose=0)[0]
                best_action = int(np.argmax(q_vals))
                if best_action == 0:
                    print(f"FOUND ACTION 0! SoH={soh}, Temp={temp}, Cycle={cycle}, Current={current}")
                    print(f"Q-values: {q_vals}")
                    sys.exit(0)

print("Action 0 was never chosen in the grid search.")
