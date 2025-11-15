import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

MODEL_PATH = "models/stress_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

def train_initial_model():
    df = pd.DataFrame({
        "sleep_hours": [6, 8, 5, 7, 4, 9],
        "hydration": [3, 5, 2, 4, 1, 6],
        "mood": ["stressed", "great", "tired", "okay", "anxious", "great"],
        "stress_level": [1, 0, 1, 0, 1, 0]
    })
    le = LabelEncoder()
    df["mood_enc"] = le.fit_transform(df["mood"])
    X = df[["sleep_hours", "hydration", "mood_enc"]]
    y = df["stress_level"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

def load_model_and_encoder():
    try:
        model = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        return model, le
    except Exception:
        train_initial_model()
        return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)

def predict_stress(sleep_hours: float, hydration: float, mood: str):
    model, le = load_model_and_encoder()
    mood_enc = le.transform([mood]) if mood in le.classes_ else le.transform(["okay"])
    X = np.array([[sleep_hours, hydration, mood_enc[0]]])
    proba = model.predict_proba(X)[0][1]
    forecast = "Low" if proba < 0.33 else "Moderate" if proba < 0.66 else "High"
    return {"high_stress_prob": round(float(proba), 3), "forecast": forecast}
