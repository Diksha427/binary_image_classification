import pytest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.model import SimpleCNN

class TestModel:
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        model = SimpleCNN(num_classes=2)
        assert model is not None
        
        # Check number of classes
        assert model.fc3.out_features == 2
    
    def test_model_forward_shape(self):
        """Test model forward pass output shape"""
        model = SimpleCNN(num_classes=2)
        model.eval()
        
        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Output should be [1, 2] for 2 classes
        assert output.shape == (1, 2)