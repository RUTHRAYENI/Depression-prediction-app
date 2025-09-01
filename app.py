import streamlit as st
import pandas as pd
import pickle

# ------------------------------
# Load trained model (replace with your own .pkl file)
# ------------------------------
try:
    model = pickle.load(open("depression_model.pkl", "rb"))
except:
    model = None
    st.warning("⚠️ Model file not found. Please place 'depression_model.pkl' in the same folder.")

# ------------------------------
# User input function
# ------------------------------
def get_user_input():
    st.sidebar.header("Enter your details:")

    # Example numeric inputs (change according to your dataset features)
    age = st.sidebar.slider("Age", 10, 60, 25)
    cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0)
    sleep = st.sidebar.slider("Sleep Duration (hrs)", 0, 12, 6)
    study_hours = st.sidebar.slider("Work/Study Hours", 0, 15, 6)
    pressure = st.sidebar.slider("Academic Pressure (1-10)", 1, 10, 5)

    # Example categorical inputs
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    diet = st.sidebar.selectbox("Dietary Habits", ["Healthy", "Unhealthy"])
    history = st.sidebar.selectbox("Family History of Mental Illness", ["Yes", "No"])

    # Convert to dataframe
    user_data = {
        "Age": age,
        "CGPA": cgpa,
        "Sleep Duration": sleep,
        "Work/Study Hours": study_hours,
        "Academic Pressure": pressure,
        "Gender": gender,
        "Dietary Habits": diet,
        "Family History of Mental Illness": history
    }

    return pd.DataFrame([user_data])

# ------------------------------
# Main App
# ------------------------------
st.title("🧠 Depression Prediction App")

user_df = get_user_input()

st.subheader("Your Input Data")
st.write(user_df)

if model is not None:
    if st.button("🔍 Predict"):
        prediction = model.predict(user_df)[0]

        if prediction == 0:  # No depression
            st.success("✅ You are *not* likely to have depression.")
            st.balloons()  # Balloons for happy case

        else:  # Depression
            st.error("🚨 Warning: You may be at risk of depression.")
            
            # Custom flashing red warning
            st.markdown(
                """
                <style>
                @keyframes flash {
                    0% {opacity: 1;}
                    50% {opacity: 0;}
                    100% {opacity: 1;}
                }
                .warning {
                    color: red;
                    font-size: 30px;
                    font-weight: bold;
                    animation: flash 1s infinite;
                    text-align: center;
                }
                </style>
                <div class="warning">⚠️ SEEK HELP IMMEDIATELY ⚠️</div>
                """,
                unsafe_allow_html=True
            )
else:
    st.stop()
