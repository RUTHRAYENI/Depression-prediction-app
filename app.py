import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("cleaneddataset.csv")

# Encode categorical columns
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Features and target
X = df.drop("Depression", axis=1)
y = df["Depression"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
acc = accuracy_score(y_test, model.predict(X_test))

# Streamlit UI
st.set_page_config(page_title="Mental Health Predictor", layout="wide")

# Peaceful gradient background
page_bg = """
<style>
.stApp {
    background: linear-gradient(to bottom, #87CEFA, #FFDAB9); /* sky blue to peach */
    color: black;
}
h1, h2, h3, h4, h5, h6, label, .stMarkdown {
    color: black !important;
    text-shadow: 1px 1px 4px white;
}
.suggestion-box {
    background: rgba(255,255,255,0.2);
    padding: 15px;
    border-radius: 12px;
    margin-top: 20px;
    font-size: 18px;
    color: black;
}
@keyframes pop {
    0% { transform: scale(0.8); opacity: 0; }
    50% { transform: scale(1.1); opacity: 1; }
    100% { transform: scale(1); opacity: 0.9; }
}
.pop-alert {
    animation: pop 1s ease-in-out infinite alternate;
    background: red;
    color: black;
    padding: 20px;
    border-radius: 10px;
    font-size: 22px;
    text-align: center;
    font-weight: bold;
    margin-top: 20px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("Student Depression Prediction")

st.subheader("Fill in your details for a supportive prediction")

# Input form
user_input = {}
for col in X.columns:
    if col in label_encoders:
        options = list(label_encoders[col].classes_)
        user_input[col] = st.selectbox(f"{col}", options)
    else:
        user_input[col] = st.number_input(f"{col}", value=0)

# Convert user input into dataframe
user_df = pd.DataFrame([user_input])
for col in user_df.columns:
    if col in label_encoders:
        user_df[col] = label_encoders[col].transform(user_df[col])

# Prediction button
if st.button("🔮 Predict"):
    prediction = model.predict(user_df)[0]

    if prediction == 1:
        st.markdown('<div class="pop-alert">⚠️ You may be experiencing symptoms of depression</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="suggestion-box">
            💡 Here are some supportive suggestions for you:<br><br>
            ✅ Talk to a trusted friend or family member<br>
            ✅ Consider speaking with a mental health professional<br>
            ✅ Practice deep breathing or meditation<br>
            ✅ Go for a short walk and connect with nature<br>
            ✅ Remember: You are not alone 💙
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.success("🌸 You seem to be doing well. Keep taking care of your mental health!")
        st.markdown(
            """
            <div class="suggestion-box">
            🌿 Tips to stay positive:<br><br>
            🌞 Maintain a healthy sleep routine<br>
            🍎 Eat nutritious meals<br>
            🏃 Stay physically active<br>
            📖 Practice gratitude journaling<br>
            💬 Stay connected with loved ones
            </div>
            """,
            unsafe_allow_html=True,
        )
        # 🎈 Balloons popping for no depression
        st.balloons()


