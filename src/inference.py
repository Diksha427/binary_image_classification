from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import SimpleCNN

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cats vs Dogs Classifier", 
              description="API for classifying cats and dogs images",
              version="1.0.0")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.pt"  # or final_model.pt

try:
    # Initialize model
    model = SimpleCNN(num_classes=2)
    
    # Load weights
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        logger.info(f" Model loaded successfully from {MODEL_PATH}")
    else:
        logger.warning(f" Model not found at {MODEL_PATH}")
        model = None
        
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Image preprocessing
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms and add batch dimension
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Cats vs Dogs Classifier API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make prediction (POST image file)",
            "/docs": "Swagger documentation"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    if model is None:
        return {"status": "degraded", "model_loaded": False}
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an image contains a cat or a dog
    
    - Upload an image file (jpg, jpeg, png)
    - Returns class prediction and confidence
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        logger.info(f"Processing image: {file.filename}, size: {image.size}, format: {image.format}")
        
        # Preprocess
        input_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Get class and confidence
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
        
        # Class names
        classes = ['cat', 'dog']
        
        # Create response
        response = {
            "filename": file.filename,
            "prediction": classes[predicted_class],
            "confidence": round(confidence, 4),
            "probabilities": {
                "cat": round(probabilities[0].item(), 4),
                "dog": round(probabilities[1].item(), 4)
            }
        }
        
        logger.info(f"Prediction: {response['prediction']} with confidence {confidence:.4f}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_base64")
async def predict_base64(image_data: dict):
    """
    Predict from base64 encoded image
    """
    import base64
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data['image'])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess
        input_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        predicted_class = torch.argmax(probabilities).item()
        classes = ['cat', 'dog']
        
        return {
            "prediction": classes[predicted_class],
            "confidence": round(probabilities[predicted_class].item(), 4),
            "probabilities": {
                "cat": round(probabilities[0].item(), 4),
                "dog": round(probabilities[1].item(), 4)
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing base64 image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def model_info():
    """Get model information"""
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "model_type": "SimpleCNN",
        "input_size": "224x224",
        "num_classes": 2,
        "classes": ["cat", "dog"],
        "device": str(device)
    }

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)