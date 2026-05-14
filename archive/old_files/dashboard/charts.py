import plotly.graph_objects as go
import streamlit as st

def generate_battery_health_gauge(soh_pct, health_label):
    gauge_color = {
        "Healthy": "#10b981", 
        "Moderate": "#f59e0b", 
        "Degrading": "#f97316", 
        "Severely Degraded": "#ef4444"
    }.get(health_label, "#f59e0b")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", 
        value=soh_pct,
        number={"suffix": "%", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": gauge_color},
            "steps": [
                {"range": [0, 70], "color": "#fee2e2"}, 
                {"range": [70, 90], "color": "#fef3c7"}, 
                {"range": [90, 100], "color": "#d1fae5"}
            ],
        }
    ))
    fig.update_layout(
        height=180, 
        margin=dict(l=20, r=20, t=20, b=20), 
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def generate_ai_decision_bar(q_values):
    fig = go.Figure(data=[go.Bar(
        x=["Decrease", "Maintain", "Increase"], 
        y=q_values.tolist(),
        marker_color=["#ef4444", "#f59e0b", "#10b981"]
    )])
    fig.update_layout(
        height=220, 
        margin=dict(l=20, r=20, t=20, b=20), 
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig
