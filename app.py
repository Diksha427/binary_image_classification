from fastapi import FastAPI
import torch
from src.model import SimpleCNN
import numpy as np

app = FastAPI()

model = SimpleCNN()
model.load_state_dict(torch.load("models/model.pt"))
model.eval()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: list):
    x = torch.tensor(data).float().view(1,1,28,28)
    output = model(x)
    probs = torch.softmax(output, dim=1)
    return {"prediction": torch.argmax(probs).item()}
