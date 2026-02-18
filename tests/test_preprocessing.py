import pytest
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the function
from src.inference import preprocess_image

class TestPreprocessing:
    
    def test_preprocess_image_shape(self):
        """Test that preprocessing returns correct tensor shape"""
        # Create a dummy RGB image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        
        # Preprocess
        tensor = preprocess_image(dummy_image)
        
        # Should be [1, 3, 224, 224] (batch, channels, height, width)
        assert tensor.shape == (1, 3, 224, 224)
        assert isinstance(tensor, torch.Tensor)
    
    def test_preprocess_image_normalization(self):
        """Test that image is properly normalized"""
        # Create a white image (all 255)
        dummy_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
        
        # Preprocess
        tensor = preprocess_image(dummy_image)
        
        # Check if normalized (values should be around 2-3 after normalization)
        assert tensor.max() < 3.0
        assert tensor.min() > -2.0
    
    def test_preprocess_rgb_conversion(self):
        """Test that grayscale images are converted to RGB"""
        # Create grayscale image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
        
        # Preprocess - should convert to RGB
        tensor = preprocess_image(dummy_image)
        
        # Should have 3 channels
        assert tensor.shape[1] == 3