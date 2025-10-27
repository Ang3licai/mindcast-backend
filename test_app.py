from flask import Flask, jsonify
import numpy as np
from stress_classifier import model, scaler  # import your trained model and scaler

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Stress Classifier API is running! Use /predict to get results."

@app.route('/predict')
def predict():
    # example data
    example = np.array([[95, 8.5, 37.2, 20]])
    example_scaled = scaler.transform(example)
    pred = model.predict(example_scaled)
    return jsonify({'predicted_stress_level': int(pred[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
