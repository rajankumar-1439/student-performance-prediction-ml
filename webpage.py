import streamlit as st 
import numpy as np 
import joblib
import warnings
warnings.filterwarnings("ignore")

model = joblib.load("best_model.pkl")

st.title("Student Exam Score predictor.")

study_hours = st.slider("Study Hours par day.", 0.0, 12.0, 2.0)
attendance = st.slider("Attendance percentage.", 0.0, 100.0, 80.0)
mental_health = st.slider("Mental Health Ration (1-10).", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)

if st.button("Predict Exam Score"):

    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours]])
    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))

    st.success(f"Pedicted Exam Score : {prediction:.2f}")