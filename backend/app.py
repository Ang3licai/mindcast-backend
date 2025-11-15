from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from ml import predict_stress, load_model_and_encoder

app = FastAPI(title="MindCast AI API")
sentiment = pipeline("sentiment-analysis")

class StressInput(BaseModel):
    sleep_hours: float
    hydration: float
    mood: str

class JournalInput(BaseModel):
    entry: str

@app.get("/health")
def health():
    load_model_and_encoder()
    return {"status": "ok"}

@app.post("/predict_stress")
def predict_stress_endpoint(payload: StressInput):
    return predict_stress(payload.sleep_hours, payload.hydration, payload.mood)

@app.post("/analyze_journal")
def analyze_journal(payload: JournalInput):
    res = sentiment(payload.entry)[0]
    return {"sentiment": res["label"], "confidence": round(float(res["score"]), 3)}
