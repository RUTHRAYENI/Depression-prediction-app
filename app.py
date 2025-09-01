import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- Page Config ---
st.set_page_config(page_title="🧠 Depression Prediction", page_icon="💡", layout="wide")

# --- Custom CSS for Background ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1527236438218-d82077ae1f85");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
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
    df = pd.read_csv("cleaneddataset.csv")
    return df

df = load_data()

st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# --- Encode categorical columns ---
le_dict = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # store label encoder

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
            # If unseen value, map to 0
            user_data[col] = le_dict[col].transform([choice])[0] if choice in options else 0
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

# --- Predict ---
if st.button("🔍 Predict"):
    user_df = user_df[X.columns]
    prediction = model.predict(user_df)[0]

    if prediction == 1:
        # 🚨 Depression Detected
        st.markdown(
            """
            <div style="background-color:#ff4d4d; padding:30px; border-radius:12px; text-align:center;">
                <h1 style="color:white;">⚠️ Depression Detected ⚠️</h1>
                <p style="color:white; font-size:18px;">Please seek support from friends, family, or a professional.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.error("⚠ The model predicts that you may be at risk of depression.")
        st.info("""
        💡 **Suggestions:**  
        - Maintain a regular sleep routine 😴  
        - Stay physically active (exercise, yoga, walking) 🏃‍♂️  
        - Eat balanced meals & hydrate 🥗💧  
        - Connect with friends/family 👨‍👩‍👧‍👦  
        - Seek professional help if symptoms persist 👩‍⚕️👨‍⚕️  
        """)
    else:
        # ✅ No Depression Detected
        st.success("✅ The model predicts that you are not at risk of depression. 🎉")
        st.balloons()
        st.info("""
        🎯 **Tips to Stay Healthy:**  
        - Keep practicing hobbies 🎨🎶  
        - Take short breaks during study/work ⏳  
        - Practice mindfulness or meditation 🌸  
        - Surround yourself with positive people 🤝  
        - Maintain a balanced lifestyle 🌞
        """)
