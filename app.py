import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- Page Config ---
st.set_page_config(page_title="🧠 Depression Prediction App", page_icon="💡", layout="wide")

# --- Custom CSS for background & styling ---
st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa, #fce4ec);
    }
    h1 {
        color: #2e7d32 !important;
        text-align: center;
    }
    h3 {
        color: #1565c0 !important;
    }
    .stAlert {
        border-radius: 12px;
        padding: 15px;
    }
    /* Flashing warning animation */
    @keyframes flash {
        0% {opacity: 1;}
        50% {opacity: 0;}
        100% {opacity: 1;}
    }
    .warning {
        color: red;
        font-size: 30px;
        font-weight: bold;
        text-align: center;
        animation: flash 1s infinite;
    }
    /* Confetti-like emoji animation */
    @keyframes fall {
        0% {transform: translateY(-100px);}
        100% {transform: translateY(100vh);}
    }
    .confetti {
        position: fixed;
        top: 0;
        left: 50%;
        font-size: 40px;
        animation: fall 3s linear forwards;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.title("🧠 Depression Prediction Web App")

# --- Load Dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("cleaneddataset.csv")

df = load_data()

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# --- Encode categorical columns ---
le_dict = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# --- Features and target ---
X = df.drop("Depression", axis=1)
y = df["Depression"]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Show accuracy ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"📊 Model Accuracy: {acc*100:.2f}%")

# --- User Input ---
st.subheader("📝 Enter your details")

def get_user_input():
    user_data = {}
    for col in X.columns:
        if col in le_dict:  # Categorical
            options = list(le_dict[col].classes_)
            choice = st.selectbox(f"{col}", options=options)
            user_data[col] = le_dict[col].transform([choice])[0]
        else:  # Numerical
            min_val = int(X[col].min()) if X[col].dtype != float else float(X[col].min())
            max_val = int(X[col].max()) if X[col].dtype != float else float(X[col].max())
            default_val = float(X[col].mean())
            if X[col].dtype == int:
                user_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=int(default_val))
            else:
                user_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=default_val)
    return pd.DataFrame([user_data])

user_df = get_user_input()

# --- Prediction ---
if st.button("🔍 Predict"):
    user_df = user_df[X.columns]
    prediction = model.predict(user_df)[0]

    if prediction == 1:
        st.error("⚠ The model predicts that you may be **at risk of depression.**")

        # Flashing warning
        st.markdown("<div class='warning'>🚨 SEEK SUPPORT 🚨</div>", unsafe_allow_html=True)

        st.info("""
        💡 **Suggestions:**  
        - Maintain a regular sleep routine 😴  
        - Stay physically active (exercise, yoga, walking) 🏃‍♂️  
        - Eat balanced meals & hydrate 🥗💧  
        - Connect with friends/family 👨‍👩‍👧‍👦  
        - Avoid excessive screen time 📵  
        - Seek professional help if symptoms persist 👩‍⚕️👨‍⚕️  
        """)

    else:
        st.success("✅ The model predicts that you are **not at risk of depression.** 🎉")
        st.balloons()

        # Confetti-like popping effect
        st.markdown("<div class='confetti'>🎊</div>", unsafe_allow_html=True)

        st.info("""
        🎯 **Tips to Stay Healthy:**  
        - Keep practicing hobbies 🎨🎶  
        - Take short breaks during study/work ⏳  
        - Practice mindfulness or meditation 🌸  
        - Surround yourself with positive people 🤝  
        - Maintain a balanced lifestyle 🌞
        """)
