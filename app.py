import streamlit as st

# Page config
st.set_page_config(page_title="Depression Check", page_icon="🧠", layout="wide")

# Background CSS
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(to right, #74ebd5, #ACB6E5);
    background-size: cover;
}
h1 {
    color: #222831;
    text-align: center;
    font-family: 'Trebuchet MS', sans-serif;
}
.stButton>button {
    background-color: #ff6b6b;
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 10px 24px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.title("🧠 Depression Self Check App")

st.markdown("### Please fill in the details below:")

# User Inputs
age = st.slider("Age", 10, 80, 20)
study_hours = st.slider("Daily Study/Work Hours", 0, 16, 4)
sleep_hours = st.slider("Sleep Duration (hours)", 0, 12, 7)
stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
academic_pressure = st.selectbox("Academic Pressure", ["Low", "Medium", "High"])
family_history = st.radio("Family history of mental illness?", ["No", "Yes"])

# Simple Scoring Logic (instead of ML model)
score = 0
if stress_level == "High":
    score += 2
elif stress_level == "Medium":
    score += 1

if academic_pressure == "High":
    score += 2
elif academic_pressure == "Medium":
    score += 1

if family_history == "Yes":
    score += 2

if sleep_hours < 5:
    score += 2
elif sleep_hours < 7:
    score += 1

if study_hours > 10:
    score += 2
elif study_hours > 7:
    score += 1

# Button to Predict
if st.button("Check Result"):
    if score >= 5:
        st.error("⚠️ Possible Depression Detected! Please seek support.")
        st.markdown(
            "<h3 style='color:red; text-align:center;'>⚠️⚠️⚠️ WARNING ⚠️⚠️⚠️</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h4 style='color:black; text-align:center;'>Take care of your mental health and consult a professional.</h4>",
            unsafe_allow_html=True,
        )
    else:
        st.success("🎉 No Depression Detected. Keep it up! 🎉")
        st.balloons()
