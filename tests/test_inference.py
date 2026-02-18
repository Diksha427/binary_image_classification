import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import io

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Now import should work
from src.inference import app

class TestAPI:
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_predict_endpoint_no_file(self, client):
        """Test predict endpoint without file"""
        response = client.post("/predict")
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_predict_endpoint_with_dummy_image(self, client):
        """Test predict endpoint with dummy image"""
        # Create dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        dummy_image.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        # Send request
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        response = client.post("/predict", files=files)
        
        # If model not loaded, should return 503
        if response.status_code == 503:
            assert response.json()["detail"] == "Model not loaded"
        else:
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert data["prediction"] in ["cat", "dog"]