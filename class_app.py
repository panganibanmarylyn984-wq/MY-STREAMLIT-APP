import streamlit as st
import joblib
import numpy as np

# ================= LOAD MODEL =================
model = joblib.load("heart_disease_model.pkl")

# ================= APP TITLE =================
st.title("Heart Disease Prediction System")
st.write("Predict whether a patient has heart disease based on medical attributes.")

# ================= INPUT FIELDS =================
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = True, 0 = False)", [0, 1])
restecg = st.number_input("Resting ECG (0-2)", min_value=0, max_value=2, value=1)
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression", value=1.0)
slope = st.number_input("Slope (0-2)", min_value=0, max_value=2, value=1)
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
thal = st.number_input("Thalassemia (1-3)", min_value=1, max_value=3, value=2)

# ================= PREDICTION =================
if st.button("Predict Heart Disease"):

    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Patient is likely to have Heart Disease")
    else:
        st.success("✅ Patient is NOT likely to have Heart Disease")