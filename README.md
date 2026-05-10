# ⚡ Explainable AI EV Battery Management Dashboard

AI-driven battery health prediction and charging optimization using **GRU + Double DQN + SHAP**.

## 🏗 Architecture

```
ev-battery-ai/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python version for deployment
├── README.md
│
├── models/                   # Trained ML models
│   ├── gru_soh_model.keras
│   ├── double_dqn_calibrated.keras
│   ├── gru_scaler_X.pkl
│   └── gru_scaler_y.pkl
│
├── utils/                    # Backend utilities
│   ├── inference.py          # Model loading & predictions
│   ├── preprocessing.py      # Data preprocessing & feature engineering
│   └── shap_utils.py         # SHAP computation & AI reasoning
│
├── components/               # UI components
│   ├── header.py             # Header & footer
│   ├── inputs.py             # Battery sensor inputs
│   ├── soh_gauge.py          # SoH gauge display
│   ├── rl_decision.py        # RL decision display
│   ├── shap_charts.py        # SHAP visualization charts
│   └── reasoning.py          # AI reasoning engine
│
├── assets/                   # Static assets
│   ├── styles.css            # Custom CSS
│   └── images/
│
└── archive/                  # Archived files
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📋 Features

- **SoH Prediction** — GRU neural network predicts battery State of Health
- **Charging Optimization** — Double DQN recommends optimal charging action
- **SHAP Explainability** — 9 interactive charts explaining AI decisions
- **AI Reasoning** — Human-readable explanations for charging recommendations

## 🔧 Tech Stack

- Streamlit, TensorFlow/Keras, Plotly, SHAP, scikit-learn