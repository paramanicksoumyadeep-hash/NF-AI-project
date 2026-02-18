import streamlit as st
import numpy as np
import joblib

# Load models
flux_model = joblib.load("models/flux_model.pkl")
rej_model = joblib.load("models/rejection_model.pkl")
encoder = joblib.load("models/material_encoder.pkl")

st.title("AI Nanofiltration Membrane Design Predictor")

st.write("Enter membrane design parameters:")

# Inputs
material = st.selectbox("Material", ["polyamide", "PES", "PVDF"])
pressure = st.slider("Pressure (bar)", 5.0, 20.0, 10.0)
temperature = st.slider("Temperature (°C)", 20.0, 35.0, 25.0)
pH = st.slider("pH", 5.5, 8.5, 7.0)
pore_size = st.slider("Pore Size (nm)", 0.5, 1.2, 0.8)
thickness = st.slider("Thickness (nm)", 80.0, 180.0, 120.0)
surface_charge = st.slider("Surface Charge", -1.2, -0.3, -0.8)
feed_conc = st.slider("Feed Concentration", 500.0, 1500.0, 1000.0)

if st.button("Predict Performance"):

    material_encoded = encoder.transform([material])[0]

    input_data = np.array([[
        material_encoded,
        pressure,
        temperature,
        pH,
        pore_size,
        thickness,
        surface_charge,
        feed_conc
    ]])

    flux_pred = flux_model.predict(input_data)[0]
    rej_pred = rej_model.predict(input_data)[0]

    st.subheader("Predicted Results")
    st.success(f"Predicted Flux: {flux_pred:.2f} LMH")
    st.success(f"Predicted Rejection: {rej_pred:.2f} %")
