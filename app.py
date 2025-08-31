import streamlit as st
import pandas as pd
import random

st.title("🧠 Depression Prediction App ")

# Collect user inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, value=20)
city = st.text_input("City")
academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5)
study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)
sleep_duration = st.slider("Sleep Duration (hours)", 0, 12, 7)
dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy", "Average"])
degree = st.text_input("Degree")
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
work_study_hours = st.slider("Work/Study Hours per day", 0, 24, 6)
financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

# Create DataFrame (optional, for display/demo purposes)
user_data = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "City": [city],
    "Academic Pressure": [academic_pressure],
    "CGPA": [cgpa],
    "Study Satisfaction": [study_satisfaction],
    "Sleep Duration": [sleep_duration],
    "Dietary Habits": [dietary_habits],
    "Degree": [degree],
    "Have you ever had suicidal thoughts ?": [suicidal_thoughts],
    "Work/Study Hours": [work_study_hours],
    "Financial Stress": [financial_stress],
    "Family History of Mental Illness": [family_history]
})

st.subheader("Your Input Data")
st.dataframe(user_data)

# Mock prediction
if st.button("Predict Depression"):
    # Simple logic or random prediction for demo
    if suicidal_thoughts == "Yes" or academic_pressure >= 4 or financial_stress >= 4:
        prediction = 1
    else:
        prediction = random.choice([0, 1])  # random for demo

    if prediction == 1:
        st.error("⚠️ High risk of Depression")
    else:
        st.success("✅ Low risk of Depression")
