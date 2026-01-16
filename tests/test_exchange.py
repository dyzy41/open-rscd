"""
Unit tests for Exchange modules.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.exchange import ExchangeBlock, SpatialExchange, ExchangeModule


def test_exchange_block():
    """Test ExchangeBlock forward pass."""
    batch_size = 2
    channels = 64
    height, width = 32, 32
    
    # Create module
    exchange = ExchangeBlock(channels, ratio=4)
    
    # Create dummy input
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    out1, out2 = exchange(x1, x2)
    
    # Check output shapes
    assert out1.shape == (batch_size, channels, height, width)
    assert out2.shape == (batch_size, channels, height, width)
    
    print("✓ ExchangeBlock test passed")


def test_spatial_exchange():
    """Test SpatialExchange forward pass."""
    batch_size = 2
    channels = 64
    height, width = 32, 32
    
    # Create module
    spatial_ex = SpatialExchange(channels)
    
    # Create dummy input
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    out1, out2 = spatial_ex(x1, x2)
    
    # Check output shapes
    assert out1.shape == (batch_size, channels, height, width)
    assert out2.shape == (batch_size, channels, height, width)
    
    print("✓ SpatialExchange test passed")


def test_exchange_module():
    """Test complete ExchangeModule forward pass."""
    batch_size = 2
    channels = 64
    height, width = 32, 32
    
    # Create module
    exchange_module = ExchangeModule(channels, ratio=4)
    
    # Create dummy input
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    out1, out2 = exchange_module(x1, x2)
    
    # Check output shapes
    assert out1.shape == (batch_size, channels, height, width)
    assert out2.shape == (batch_size, channels, height, width)
    
    print("✓ ExchangeModule test passed")


if __name__ == '__main__':
    test_exchange_block()
    test_spatial_exchange()
    test_exchange_module()
    print("\n✓ All Exchange module tests passed!")
