import streamlit as st
import joblib
import numpy as np

# ================= LOAD MODEL =================
model = joblib.load("salary_model.pkl")

# ================= APP TITLE =================
st.title("Salary Prediction System (Linear Regression)")
st.write("Predict salary based on years of experience.")

# ================= INPUT FIELD =================
years_experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=5.0
)

# ================= PREDICTION BUTTON =================
if st.button("Predict Salary"):

    input_data = np.array([[years_experience]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Salary: {prediction[0]:,.2f}")