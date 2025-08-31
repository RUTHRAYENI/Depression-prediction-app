import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load and preprocess dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaneddataset.csv")

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    return df

df = load_data()

# Features and target
X = df.drop("Depression", axis=1)
y = df["Depression"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

st.title("🧠 Depression Prediction App")

st.write("Fill in the details below to check the prediction.")

# -----------------------------
# User Input Form
# -----------------------------
with st.form("user_input_form"):
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=10, max_value=100, step=1)
    cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, step=0.1)
    sleep_duration = st.selectbox("Sleep Duration", ["<5 hours", "5-7 hours", "7-9 hours", ">9 hours"])
    study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)
    financial_stress = st.selectbox("Financial Stress", ["Yes", "No"])
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
    work_study_hours = st.number_input("Work/Study Hours per day", min_value=0, max_value=20, step=1)

    submit = st.form_submit_button("Predict")

# -----------------------------
# Process Input and Predict
# -----------------------------
if submit:
    # Map categorical inputs manually
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    sleep_map = {"<5 hours": 0, "5-7 hours": 1, "7-9 hours": 2, ">9 hours": 3}
    binary_map = {"No": 0, "Yes": 1}

    user_data = pd.DataFrame({
        "Gender": [gender_map[gender]],
        "Age": [age],
        "CGPA": [cgpa],
        "Sleep Duration": [sleep_map[sleep_duration]],
        "Study Satisfaction": [study_satisfaction],
        "Financial Stress": [binary_map[financial_stress]],
        "Family History of Mental Illness": [binary_map[family_history]],
        "Academic Pressure": [academic_pressure],
        "Work/Study Hours": [work_study_hours],
    })

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of Depression (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low risk of Depression (Probability: {probability:.2f})")
