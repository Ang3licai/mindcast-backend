# apprun.py — run the backend server safely
import os, sys
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

app = Flask(__name__)

# Allow your website(s) to access this backend
CORS(app, resources={r"/*": {"origins": [
    "https://ang3licai.github.io",   # your GitHub Pages site
    "https://mindcast.github.io"     # optional future domain
]}})

# Test route to make sure the server works
@app.route("/", methods=["GET"])
def home():
    return jsonify(ok=True, service="MindCast Stress API")

# --- make sure Python can find subfolders ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    BASE_DIR,
    os.path.join(BASE_DIR, "models"),
    os.path.join(BASE_DIR, "backend"),
])

# --- try importing your team’s trained model ---
model = None
scaler = None

def try_import_model():
    global model, scaler
    try:
        from nn_classifier import model as m, scaler as s
        model, scaler = m, s
        print("[OK] Using model from nn_classifier.py")
        return
    except Exception as e:
        print("[INFO] nn_classifier import failed:", e)

    try:
        from lstm_forecast import model as m, scaler as s
        model, scaler = m, s
        print("[OK] Using model from lstm_forecast.py")
        return
    except Exception as e:
        print("[INFO] lstm_forecast import failed:", e)

try_import_model()

# --- fallbacks so the backend still runs ---
class DummyScaler:
    def transform(self, X):
        return np.array(X, dtype=float)

class DummyModel:
    def predict(self, X):
        hr, sc = float(X[0][0]), float(X[0][1])
        if hr > 95 or sc > 0.7: return [3]    # High
        if hr > 80 or sc > 0.5: return [2]    # Moderate
        return [1]                            # Low

if scaler is None:
    scaler = DummyScaler()
    print("[WARN] Using DummyScaler")
if model is None:
    model = DummyModel()
    print("[WARN] Using DummyModel")

# --- flask app setup ---
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"ok": True, "service": "MindCast Stress API"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    try:
        hr = float(data.get("heart_rate", 80))
        sc = float(data.get("skin_conductance", 0.4))
        bt = float(data.get("body_temp", 36.6))
    except Exception:
        return jsonify({"error": "Invalid input"}), 400

    features = np.array([[hr, sc, bt]])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)
    level_num = int(pred[0])
    level = {1: "Low", 2: "Moderate", 3: "High"}.get(level_num, "Moderate")
    return jsonify({"level": level, "level_num": level_num})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"reply": "I’m here whenever you want to talk."})

    # Call OpenAI
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}]
        )
        reply = response.choices[0].message["content"]
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error talking to AI: {e}"}), 500
