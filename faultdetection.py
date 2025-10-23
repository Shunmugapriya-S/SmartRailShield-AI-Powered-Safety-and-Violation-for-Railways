import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st
import joblib
df = pd.read_csv("synthetic_railway_fault_dataset.csv")
X = df[["vibration", "temperature", "speed", "noise_level", "pilot_drowsy", "obstacle_detected"]]
y = df["fault"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(xgb_model, "xgb_model.pkl")
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    st.write(f"### {name} Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    st.write("Classification Report:\n", classification_report(y_test, y_pred))
st.title("Train Engine Fault Prediction 🚂")

st.sidebar.header("Enter Engine Parameters")

vibration = st.sidebar.number_input("Vibration Level", min_value=0.0, max_value=100.0, value=5.0)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=150.0, value=70.0)
speed = st.sidebar.number_input("Speed (km/h)", min_value=0.0, max_value=200.0, value=50.0)
noise_level = st.sidebar.number_input("Noise Level", min_value=0.0, max_value=200.0, value=40.0)
pilot_drowsy = st.sidebar.selectbox("Pilot Drowsy?", [0, 1])
obstacle_detected = st.sidebar.selectbox("Obstacle Detected?", [0, 1])

# Model Selection
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost"])

input_data = pd.DataFrame([[vibration, temperature, speed, noise_level, pilot_drowsy, obstacle_detected]],
                          columns=["vibration", "temperature", "speed", "noise_level", "pilot_drowsy", "obstacle_detected"])

# Load selected model
if model_choice == "Random Forest":
    model = joblib.load("rf_model.pkl")
else:
    model = joblib.load("xgb_model.pkl")

# Predict
if st.button("Predict Engine Condition"):
    prediction = model.predict(input_data)[0]
    if prediction == 0:
        st.success("✅ Engine is Safe to Start")
    else:
        st.error("⚠️ Engine may Fail. Check before starting!")

# Optional: Evaluate models on test set in the app
if st.checkbox("Show Model Evaluation"):
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")
