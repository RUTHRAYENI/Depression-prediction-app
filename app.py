import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Title
st.title("Depression Prediction Web App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaneddataset.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# Encode categorical columns if any
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Features and target (assuming 'Depression' is the target column)
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

# User Input Form
st.subheader("Enter your details to check depression risk")

user_input = {}
for col in X.columns:
    val = st.number_input(f"Enter {col}", value=float(X[col].mean()))
    user_input[col] = val

# Convert to DataFrame
user_df = pd.DataFrame([user_input])

# Predict
if st.button("Predict"):
    prediction = model.predict(user_df)[0]
    if prediction == 1:
        st.error("⚠️ The model predicts that you may be at risk of depression.")
    else:
        st.success("✅ The model predicts that you are not at risk of depression.")
