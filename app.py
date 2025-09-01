import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- Page Config ---
st.set_page_config(page_title="🧠 Depression Prediction App", page_icon="💡", layout="wide")

# --- Title ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Depression Prediction Web App</h1>", unsafe_allow_html=True)
st.write("Welcome! This app predicts the risk of depression based on your details. 💡")

# --- Load Dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("cleaneddataset.csv")

df = load_data()

# --- Show Dataset ---
with st.expander("📂 See Dataset Preview"):
    st.dataframe(df.head())

# --- Encode categorical columns ---
le_dict = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# --- Features & target ---
X = df.drop("Depression", axis=1)
y = df["Depression"]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Show accuracy in progress bar ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.progress(int(acc * 100))
st.success(f"📊 Model Accuracy: **{acc*100:.2f}%**")

# --- User Input ---
st.subheader("📝 Enter Your Details")

def get_user_input():
    user_data = {}
    for col in X.columns:
        if col in le_dict:  # categorical
            options = list(le_dict[col].classes_)
            choice = st.selectbox(f"{col}", options=options)
            user_data[col] = le_dict[col].transform([choice])[0]
        else:  # numerical
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            default_val = float(X[col].mean())
            user_data[col] = st.slider(f"{col}", min_val, max_val, default_val)
    return pd.DataFrame([user_data])

user_df = get_user_input()

# --- Prediction ---
if st.button("🔍 Predict"):
    user_df = user_df[X.columns]
    prediction = model.predict(user_df)[0]

    if prediction == 1:
        st.error("⚠ The model predicts that you may be **at risk of depression**.")
        st.info("""
        💡 **Suggestions:**  
        - Maintain a regular sleep schedule 😴  
        - Eat healthy & stay hydrated 🥗💧  
        - Exercise or do yoga 🏃‍♂️🧘‍♀️  
        - Stay connected with friends/family 👨‍👩‍👧‍👦  
        - If feelings persist, talk to a mental health professional 👨‍⚕️
        """)
    else:
        st.success("✅ The model predicts that you are **not at risk of depression**.")
        st.balloons()
        st.info("""
        🎉 **Great! Keep it up:**  
        - Continue your healthy lifestyle 💪  
        - Take breaks from study/work ⏳  
        - Practice hobbies 🎶🎨  
        - Keep a positive mindset 🌸
        """)
