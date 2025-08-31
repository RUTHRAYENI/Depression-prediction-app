import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Title
st.title("🧠 Depression Prediction Web App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaneddataset.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Encode categorical columns
le_dict = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # store for decoding user input later

# Features and target
X = df.drop("Depression", axis=1)
y = df["Depression"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {acc*100:.2f}%")

# --- User Input Form ---
st.subheader("Enter your details to check depression risk")

def get_user_input():
    user_data = {}
    # Gender
    if 'Gender' in X.columns:
        gender = st.selectbox("Gender", options=le_dict['Gender'].classes_)
        user_data['Gender'] = le_dict['Gender'].transform([gender])[0]
    # Age
    if 'Age' in X.columns:
        age = st.number_input("Age", min_value=10, max_value=100, value=25)
        user_data['Age'] = age
    # City
    if 'City' in X.columns:
        city = st.text_input("City", value="Your City")
        user_data['City'] = le_dict['City'].transform([city])[0] if city in le_dict['City'].classes_ else 0
    # Academic Pressure
    if 'Academic Pressure' in X.columns:
        academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
        user_data['Academic Pressure'] = academic_pressure
    # CGPA
    if 'CGPA' in X.columns:
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5)
        user_data['CGPA'] = cgpa
    # Study Satisfaction
    if 'Study Satisfaction' in X.columns:
        study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)
        user_data['Study Satisfaction'] = study_satisfaction
    # Sleep Duration
    if 'Sleep Duration' in X.columns:
        sleep_duration = st.slider("Sleep Duration (hours)", 0, 12, 7)
        user_data['Sleep Duration'] = sleep_duration
    # Dietary Habits
    if 'Dietary Habits' in X.columns:
        dietary = st.selectbox("Dietary Habits", options=le_dict['Dietary Habits'].classes_)
        user_data['Dietary Habits'] = le_dict['Dietary Habits'].transform([dietary])[0]
    # Degree
    if 'Degree' in X.columns:
        degree = st.text_input("Degree", value="BTech")
        user_data['Degree'] = le_dict['Degree'].transform([degree])[0] if degree in le_dict['Degree'].classes_ else 0
    # Suicidal thoughts
    if 'Have you ever had suicidal thoughts ?' in X.columns:
        suicidal = st.selectbox("Have you ever had suicidal thoughts?", options=le_dict['Have you ever had suicidal thoughts ?'].classes_)
        user_data['Have you ever had suicidal thoughts ?'] = le_dict['Have you ever had suicidal thoughts ?'].transform([suicidal])[0]
    # Work/Study Hours
    if 'Work/Study Hours' in X.columns:
        hours = st.number_input("Work/Study Hours per day", min_value=0, max_value=24, value=6)
        user_data['Work/Study Hours'] = hours
    # Financial Stress
    if 'Financial Stress' in X.columns:
        fin_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
        user_data['Financial Stress'] = fin_stress
    # Family History
    if 'Family History of Mental Illness' in X.columns:
        family = st.selectbox("Family History of Mental Illness", options=le_dict['Family History of Mental Illness'].classes_)
        user_data['Family History of Mental Illness'] = le_dict['Family History of Mental Illness'].transform([family])[0]

    return pd.DataFrame([user_data])

user_df = get_user_input()

# Predict
if st.button("Predict"):
    prediction = model.predict(user_df)[0]
    if prediction == 1:
        st.error("⚠ The model predicts that you may be at risk of depression.")
    else:
        st.success("✅ The model predicts that you are not at risk of depression.")
