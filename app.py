import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# -----------------------------
# Load your trained model
# -----------------------------
# Assume you saved the pipeline as 'fault_model.pkl'
#joblib.dump(best_model, 'fault_model.pkl')
model = joblib.load('fault_model.pkl')

# Label mapping
label_mapping = {0: 'No Fault', 1: 'Bearing Fault', 2: 'Overheating'}

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Industrial Fault Detection System ⚙️")
st.write("Test the trained model with custom sensor inputs.")

# Input widgets
vibration = st.slider("Vibration (mm/s)", 0.0, 1.5, 0.5, 0.01)
temperature = st.slider("Temperature (°C)", 50.0, 130.0, 90.0, 0.5)
pressure = st.slider("Pressure (bar)", 7.0, 10.0, 8.0, 0.1)
rms_vibration = st.slider("RMS Vibration", 0.0, 1.5, 0.6, 0.01)
mean_temp = st.slider("Mean Temp", 50.0, 130.0, 90.0, 0.5)

# Collect inputs
input_data = np.array([[vibration, temperature, pressure]])

# Predict
prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0]

# Display results
st.subheader("Prediction Result")
st.write(f"**Predicted Fault Type:** {label_mapping[prediction]}")

st.subheader("Prediction Probabilities")
prob_df = pd.DataFrame([proba], columns=[label_mapping[i] for i in range(len(proba))])
st.bar_chart(prob_df.T)

st.write("Created by Lakshan Siriwardhana, Nishel Pirispulle, Tharanga Dissanayake & Lahiru Samaraweera")