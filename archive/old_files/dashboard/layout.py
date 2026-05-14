import streamlit as st

def apply_custom_css():
    with open("assets/css/style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
    # Any additional inline CSS can be added here
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_header():
    st.markdown(
        '''
        <div class="dashboard-header">
            <h1 class="dashboard-title">⚡ Explainable AI EV Battery Management Dashboard</h1>
            <p class="dashboard-subtitle">AI-driven battery health prediction and charging optimisation | GRU + Double DQN + SHAP</p>
        </div>
        ''',
        unsafe_allow_html=True
    )

def render_footer():
    st.markdown(
        '<div style="text-align: center; color: #9ca3af; font-size: 0.8rem; padding: 20px;">EV Battery AI Management • Explained by SHAP • GRU + Double DQN</div>', 
        unsafe_allow_html=True
    )
