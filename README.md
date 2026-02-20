# Cats vs Dogs Image Classification — MLOps End-to-End 
Production-ready deep learning pipeline for binary image classification (Cats vs Dogs) using PyTorch.

#### Includes:
* CNN-based image classifier (224x224 RGB)
* MLflow experiment tracking
* FastAPI inference service (/predict, /health, /metrics)
* Docker containerization
* Kubernetes deployment manifests (Minikube-friendly)
* CI/CD with GitHub Actions (lint, tests, build, push)
* Basic monitoring via Prometheus metrics

> Author: Diksha Gupta  
> Course: MLOps (S1-25_AIMLCZG523) – Assignment 2
---
## 1. Dataset

* **Source**: Kaggle Cats vs Dogs dataset
* Binary classification:
 0 → Cat
 1 → Dog
* Images resized to 224x224
* Train / Validation / Test split applied
* Normalization using ImageNet mean & std

## 2. Quick Start (Local)
```
# Create & activate virtual env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train Model
python src/train.py

# Model saved to:
models/best_model.pt

# Run API
uvicorn src.inference:app --host 0.0.0.0 --port 8000

# Test Prediction
curl -X POST http://localhost:8000/predict \ -F "file=@cat.jpg"
```

## 3. Docker
```
# Build Image
docker build -t cats-dogs-classifier:latest .

# Run Container
docker run --rm -p 8000:8000 cats-dogs-classifier:latest

# Docker image is automatically pushed via CI to:
ghcr.io/diksha427/binary_image_classification:latest
```

## 4. Kubernetes (Minikube-Friendly)
```
Uses standard Deployment + Service

Exposes container port 8000

# Start Minikube
minikube start

# Deploy Application
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Access Service
If using NodePort:
minikube service cats-dogs-classifier --url

# If using LoadBalancer:
minikube tunnel
kubectl get svc

Then access:http://<external-ip>:8000/docs
```

## 5. CI/CD (GitHub Actions)
```
# Workflow located in:
.github/workflows/ci.yml

# Runs on push to main:
flake8 lint check
pytest unit tests
Docker image build
Push to GitHub Container Registry (GHCR)

# Tests include:
Model forward pass
Preprocessing validation
Inference API logic
```

## 6. Repository Layout
```
binary_image_classification/
├── .github/
│   └── workflows/
│       └── ci.yml
├── kubernetes/
│   ├── deployment.yaml
│   └── service.yaml
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_inference.py
│   └── test_preprocessing.py
├── models/
│   └── best_model.pt
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── classification_report.json
├── data/
│   └── processed.dvc
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```
## 7. API Contract

* POST /predict

#### Form-data request:
file = image file

Example Response:
```
{
  "filename": "cat.jpg",
  "prediction": "cat",
  "confidence": 0.9234,
  "probabilities": {
    "cat": 0.9234,
    "dog": 0.0766
  }
}
```
* GET /health
```  
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```
* GET /metrics

Prometheus metrics:
```
http_requests_total
prediction_duration_seconds
```

## 8. Experiment Tracking (MLflow)
```
# MLflow is used during training to log:
* Hyperparameters
* Training & validation metrics
* Confusion matrix
* Model artifacts

# To run MLflow UI locally:
mlflow ui
Open: http://127.0.0.1:5000

MLflow tracking is optional for deployment and used only during development.
```

## 9. Notes
```
The trained model (models/best_model.pt) is version-controlled for reproducible deployment.
Docker image is built automatically via GitHub Actions.
Kubernetes deployment uses 2 replicas with readiness & liveness probes.
Public GHCR registry eliminates need for imagePullSecrets.
```

## 10. License
Educational use for assignment submission.

