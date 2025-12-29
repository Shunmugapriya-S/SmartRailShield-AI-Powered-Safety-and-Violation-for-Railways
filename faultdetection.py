import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import firebase_admin
from firebase_admin import credentials, firestore
from google import genai
FIREBASE_JSON_PATH = ""
cred = credentials.Certificate(FIREBASE_JSON_PATH)
if "app1" not in firebase_admin._apps:
    firebase_admin.initialize_app(cred, name="app1")
firebase_app = firebase_admin.get_app("app1")
db = firestore.client(app=firebase_app)
def send_alert(prediction, input_data):
    alert = {
        "vibration": float(input_data["vibration"][0]),
        "temperature": float(input_data["temperature"][0]),
        "speed": float(input_data["speed"][0]),
        "noise_level": float(input_data["noise_level"][0]),
        "pilot_drowsy": int(input_data["pilot_drowsy"][0]),
        "obstacle_detected": int(input_data["obstacle_detected"][0]),
        "fault_detected": bool(prediction),
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    db.collection("engine_alerts").add(alert)
GEMINI_API_KEY = "AIzaSyD7el6hY16qEPOV4g_KPIHatGDfkJz5gI8"  # move to env variable later
client = genai.Client(api_key=GEMINI_API_KEY)
def get_gemini_explanation(input_data, prediction):
    prompt = f"""
    Engine parameters:
    - Vibration: {input_data['vibration'][0]}
    - Temperature: {input_data['temperature'][0]}
    - Speed: {input_data['speed'][0]}
    - Noise Level: {input_data['noise_level'][0]}
    - Pilot Drowsy: {input_data['pilot_drowsy'][0]}
    - Obstacle Detected: {input_data['obstacle_detected'][0]}

    Prediction: {"Fault Detected" if prediction else "Safe to Start"}

    Explain this result in simple terms for a train operator.
    """

    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=prompt
    )
    return response.text

# =========================================================
# 3️⃣ Load Dataset & Train Models
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_railway_fault_dataset.csv")

df = load_data()

X = df[
    ["vibration", "temperature", "speed",
     "noise_level", "pilot_drowsy", "obstacle_detected"]
]
y = df["fault"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

@st.cache_resource
def train_models():
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, "rf_model.pkl")

    xgb_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, "xgb_model.pkl")

    return rf, xgb_model

rf_model, xgb_model = train_models()
st.set_page_config(page_title="Train Fault Prediction", layout="wide")
st.title("🚂 Train Engine Fault Prediction System")
st.sidebar.header("🔧 Engine Parameters")
vibration = st.sidebar.number_input("Vibration Level", 0.0, 100.0, 5.0)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 150.0, 70.0)
speed = st.sidebar.number_input("Speed (km/h)", 0.0, 200.0, 50.0)
noise_level = st.sidebar.number_input("Noise Level", 0.0, 200.0, 40.0)
pilot_drowsy = st.sidebar.selectbox("Pilot Drowsy?", [0, 1])
obstacle_detected = st.sidebar.selectbox("Obstacle Detected?", [0, 1])
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])
input_data = pd.DataFrame(
    [[vibration, temperature, speed, noise_level, pilot_drowsy, obstacle_detected]],
    columns=[
        "vibration", "temperature", "speed",
        "noise_level", "pilot_drowsy", "obstacle_detected"
    ]
)
model = rf_model if model_choice == "Random Forest" else xgb_model
if st.button("🚦 Predict Engine Condition"):
    prediction = model.predict(input_data)[0]

    send_alert(prediction, input_data)

    if prediction == 0:
        st.success("✅ Engine is SAFE to start")
    else:
        st.error("⚠️ FAULT DETECTED! Do NOT start the engine")

    explanation = get_gemini_explanation(input_data, prediction)
    st.info(f"💡 **AI Explanation:** {explanation}")
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    st.subheader(f"📊 {name} Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Confusion Matrix")
    st.dataframe(confusion_matrix(y_test, y_pred))
    st.write("Classification Report")
    st.text(classification_report(y_test, y_pred))

if st.checkbox("📈 Show Model Evaluation"):
    evaluate_model(rf_model, "Random Forest")
    evaluate_model(xgb_model, "XGBoost")
