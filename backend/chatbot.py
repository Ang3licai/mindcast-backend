# backend/chatbot.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="MindCast Chatbot")

# Load a conversational model (can swap with GPT/OpenAI API)
chat_model = pipeline("text-generation", model="gpt2")

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def chat(payload: ChatInput):
    response = chat_model(payload.message, max_length=100, num_return_sequences=1)[0]["generated_text"]
    return {"reply": response}
