import streamlit as st 
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler

# Load the model
with open("wine_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit App
st.set_page_config(page_title="Wine Quality Predictor", layout="centered")
st.title("🍷 Wine Quality Prediction App")
st.write("Use the sliders to set chemical properties of wine and predict if it's **Good** or **Not Good**.")

# Input sliders
fixed_acidity = st.slider("Fixed Acidity", 0.0, 15.0, 7.4, step=0.1)
volatile_acidity = st.slider("Volatile Acidity", 0.0, 1.5, 0.7, step=0.01)
citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0, step=0.01)
residual_sugar = st.slider("Residual Sugar", 0.0, 15.0, 1.9, step=0.1)
chlorides = st.slider("Chlorides", 0.0000, 0.2, 0.076, step=0.0001)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 0.0, 100.0, 11.0, step=1.0)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 0.0, 300.0, 34.0, step=1.0)
density = st.slider("Density", 0.9900, 1.0050, 0.9978, step=0.0001)
pH = st.slider("pH", 2.5, 4.5, 3.51, step=0.01)
sulphates = st.slider("Sulphates", 0.0, 2.0, 0.56, step=0.01)
alcohol = st.slider("Alcohol", 8.0, 15.0, 9.4, step=0.1)

# Prediction
if st.button("Predict Wine Quality"):
    # Input array
    input_data = np.array([[
        fixed_acidity, volatile_acidity, citric_acid,
        residual_sugar, chlorides, free_sulfur_dioxide,
        total_sulfur_dioxide, density, pH, sulphates, alcohol
    ]])

    # Standardize input
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    label = "🟢 Good Quality Wine" if prediction == 1 else "🔴 Not Good Quality Wine"
    st.subheader(f"Prediction: {label}")
