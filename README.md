# Cats vs Dogs Image Classification — MLOps End-to-End 
Production-ready deep learning pipeline for binary image classification (Cats vs Dogs) using PyTorch.

#### Includes:
* CNN-based image classifier (224x224 RGB)
* MLflow experiment tracking
* FastAPI inference service (/predict, /health, /metrics)
* Docker containerization
* Docker Compose deployment
* CI/CD with GitHub Actions (lint, tests, build, push)
* Automated CD with smoke testing
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

## 2. Quick Start
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

## 4. Deployment (Docker Compose)
```
This project uses Docker Compose as the deployment target.

The CI/CD pipeline automatically deploys the service inside the GitHub Actions runner.

Local Deployment (Optional)

If Docker is available:
docker pull ghcr.io/diksha427/binary_image_classification:latest
docker compose up

Access:
http://localhost:8000/docs
```

## 5. CI/CD (GitHub Actions)

This project uses GitHub Actions for automated CI/CD.

### Workflow located in:
.github/workflows/ci.yml

### On every push to main branch:

* Code checkout
* Dependency installation
* Linting (flake8)
* Unit tests (pytest)
* Docker image build
* Image push to GitHub Container Registry
* Deployment via Docker Compose
* Automated smoke test (/health endpoint)
  
```
Smoke Test
curl http://localhost:8000/health

Pipeline fails automatically if deployment fails.
```

```
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
├── docker-compose.yml
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
Docker image is automatically built and pushed to GitHub Container Registry.
Deployment is handled via Docker Compose.
Smoke tests validate successful deployment.
MLflow tracking is used during development only.
```

## 10. License
Educational use for assignment submission.



