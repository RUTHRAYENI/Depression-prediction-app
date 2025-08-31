import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.title("🧠 Depression Prediction Web App")

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("cleaneddataset.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
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
st.write(f"Model Accuracy: {acc*100:.2f}%")

# --- User Input ---
st.subheader("Enter your details")

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
if st.button("Predict"):
    # Ensure columns match model
    user_df = user_df[X.columns]
    prediction = model.predict(user_df)[0]
    if prediction == 1:
        st.error("⚠ The model predicts that you may be at risk of depression.")
    else:
        st.success("✅ The model predicts that you are not at risk of depression.")
