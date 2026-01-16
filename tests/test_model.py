"""
Unit tests for the main ExchangeNet model.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ExchangeNet, build_model


def test_exchangenet_forward():
    """Test ExchangeNet forward pass."""
    batch_size = 2
    height, width = 256, 256
    
    # Create model
    model = ExchangeNet(backbone='resnet18', pretrained=False, num_classes=2)
    model.eval()
    
    # Create dummy input
    img_A = torch.randn(batch_size, 3, height, width)
    img_B = torch.randn(batch_size, 3, height, width)
    
    # Forward pass
    with torch.no_grad():
        output = model(img_A, img_B)
    
    # Check output shape
    assert output.shape == (batch_size, 2, height, width)
    
    # Check predictions
    predictions = torch.argmax(output, dim=1)
    assert predictions.shape == (batch_size, height, width)
    assert set(predictions.unique().tolist()).issubset({0, 1})
    
    print("✓ ExchangeNet forward pass test passed")


def test_build_model():
    """Test build_model factory function."""
    config = {
        'backbone': 'resnet18',
        'pretrained': False,
        'num_classes': 2
    }
    
    model = build_model(config)
    
    # Check model type
    assert isinstance(model, ExchangeNet)
    
    # Test forward pass
    batch_size = 1
    height, width = 128, 128
    
    img_A = torch.randn(batch_size, 3, height, width)
    img_B = torch.randn(batch_size, 3, height, width)
    
    model.eval()
    with torch.no_grad():
        output = model(img_A, img_B)
    
    assert output.shape == (batch_size, 2, height, width)
    
    print("✓ build_model test passed")


def test_different_backbones():
    """Test model with different backbone options."""
    backbones = ['resnet18', 'resnet34']
    
    for backbone in backbones:
        model = ExchangeNet(backbone=backbone, pretrained=False, num_classes=2)
        model.eval()
        
        # Test forward pass
        img_A = torch.randn(1, 3, 128, 128)
        img_B = torch.randn(1, 3, 128, 128)
        
        with torch.no_grad():
            output = model(img_A, img_B)
        
        assert output.shape == (1, 2, 128, 128)
        print(f"✓ {backbone} test passed")


if __name__ == '__main__':
    test_exchangenet_forward()
    test_build_model()
    test_different_backbones()
    print("\n✓ All model tests passed!")
