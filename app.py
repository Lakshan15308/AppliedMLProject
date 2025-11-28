import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = joblib.load("fault_model.pkl")

label_mapping = {
    0: "No Fault",
    1: "Bearing Fault",
    2: "Overheating"
}

# Colors for fault types
color_map = {
    "No Fault": "green",
    "Bearing Fault": "orange",
    "Overheating": "red"
}

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Industrial Fault Detection",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ Industrial Fault Detection System")
st.write("Monitor equipment health with real-time sensor simulation and ML prediction.")

st.markdown("---")

# --------------------------------------------------
# INPUT SECTION – User Sliders
# --------------------------------------------------

st.subheader("🔧 Input Sensor Readings")

col1, col2, col3 = st.columns(3)

with col1:
    vibration = st.slider("Vibration (mm/s)", 0.0, 1.5, 0.5, 0.01)

with col2:
    temperature = st.slider("Temperature (°C)", 50.0, 130.0, 90.0, 0.5)

with col3:
    pressure = st.slider("Pressure (bar)", 7.0, 10.0, 8.0, 0.1)

# Input array
input_data = np.array([[vibration, temperature, pressure]])

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0]

pred_label = label_mapping[prediction]
pred_color = color_map[pred_label]

st.markdown("---")

# --------------------------------------------------
# OUTPUT SECTION – Prediction
# --------------------------------------------------

st.subheader("🧠 ML Prediction Result")

# Display result in a colored box
st.markdown(
    f"""
    <div style="
    padding: 15px; 
    border-radius: 10px; 
    background-color:{pred_color}; 
    color:white; 
    font-size:22px; 
    text-align:center;">
        <b>Predicted Fault Type: {pred_label}</b>
    </div>
    """,
    unsafe_allow_html=True
)

# Gauge Indicator using Plotly
fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=proba[prediction] * 100,
        title={"text": "Prediction Confidence (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": pred_color},
            "steps": [
                {"range": [0, 40], "color": "#ffcccc"},
                {"range": [40, 70], "color": "#ffe5cc"},
                {"range": [70, 100], "color": "#d4ffd4"},
            ]
        }
    )
)

st.plotly_chart(fig_gauge)

# --------------------------------------------------
# PROBABILITY BAR CHART
# --------------------------------------------------

st.subheader("📊 Class Probability Distribution")

prob_df = pd.DataFrame(
    [proba],
    columns=[label_mapping[i] for i in range(len(proba))]
)

st.bar_chart(prob_df.T)

# --------------------------------------------------
# INPUT SUMMARY CARD
# --------------------------------------------------
st.subheader("📝 Sensor Input Summary")

st.dataframe(
    pd.DataFrame({
        "Sensor": ["Vibration", "Temperature", "Pressure"],
        "Value": [vibration, temperature, pressure]
    })
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")
st.write(
    "<p style='text-align:center;'>Created by "
    "<b>Lakshan Siriwardhana</b>, "
    "<b>Nishel Pirispulle</b>, "
    "<b>Tharanga Dissanayake</b>, "
    "<b>Lahiru Samaraweera</b></p>",
    unsafe_allow_html=True
)
