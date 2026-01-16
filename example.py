"""
Example script demonstrating basic usage of the ExchangeNet model.
"""

import torch
import numpy as np
from models import ExchangeNet

def main():
    """Demonstrate basic model usage."""
    
    # Create model
    print("Creating ExchangeNet model...")
    model = ExchangeNet(backbone='resnet18', pretrained=False, num_classes=2)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dummy input
    batch_size = 2
    height, width = 256, 256
    
    print(f"\nCreating dummy input: [{batch_size}, 3, {height}, {width}]")
    img_A = torch.randn(batch_size, 3, height, width)
    img_B = torch.randn(batch_size, 3, height, width)
    
    # Forward pass
    print("Running forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(img_A, img_B)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, 2, {height}, {width}]")
    
    # Get predictions
    predictions = torch.argmax(output, dim=1)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique values in predictions: {torch.unique(predictions).tolist()}")
    
    # Calculate change ratio
    change_ratio = (predictions == 1).float().mean().item()
    print(f"Change ratio: {change_ratio:.2%}")
    
    print("\n✓ Model test completed successfully!")

if __name__ == '__main__':
    main()
