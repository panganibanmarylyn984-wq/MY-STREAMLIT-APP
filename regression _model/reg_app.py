import streamlit as st
import joblib
import numpy as np
import os

MODEL_PATH = "salary_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ salary_model.pkl not found.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.title("💼 Salary Prediction System (Linear Regression)")
st.write("Predict estimated salary based on years of experience.")

st.markdown("---")

# ================= INPUT =================
years_experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.1
)

# 🔥 LIMIT DECIMAL TO 2 PLACES
years_experience = round(years_experience, 2)

st.write(f"Selected Experience: **{years_experience} years**")

# ================= PREDICTION =================
if st.button("Predict Salary"):

    input_data = np.array([[years_experience]])
    prediction = model.predict(input_data)

    st.success(f"Estimated Salary: ₱{prediction[0]:,.2f}")