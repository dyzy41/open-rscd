"""
Unit tests for utility functions (losses, metrics).
"""

import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import DiceLoss, FocalLoss, CombinedLoss, Metrics, calculate_metrics


def test_dice_loss():
    """Test Dice loss calculation."""
    batch_size = 2
    height, width = 64, 64
    num_classes = 2
    
    # Create loss function
    dice_loss = DiceLoss()
    
    # Create dummy predictions and targets
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Calculate loss
    loss = dice_loss(predictions, targets)
    
    # Check loss properties
    assert loss.item() >= 0
    assert loss.item() <= 1
    
    print("✓ DiceLoss test passed")


def test_focal_loss():
    """Test Focal loss calculation."""
    batch_size = 2
    height, width = 64, 64
    num_classes = 2
    
    # Create loss function
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Create dummy predictions and targets
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Calculate loss
    loss = focal_loss(predictions, targets)
    
    # Check loss properties
    assert loss.item() >= 0
    
    print("✓ FocalLoss test passed")


def test_combined_loss():
    """Test Combined loss calculation."""
    batch_size = 2
    height, width = 64, 64
    num_classes = 2
    
    # Create loss function
    combined_loss = CombinedLoss(ce_weight=1.0, dice_weight=1.0)
    
    # Create dummy predictions and targets
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Calculate loss
    loss = combined_loss(predictions, targets)
    
    # Check loss properties
    assert loss.item() >= 0
    
    print("✓ CombinedLoss test passed")


def test_metrics():
    """Test Metrics class."""
    # Create metrics object
    metrics = Metrics(num_classes=2)
    
    # Create perfect predictions
    predictions = torch.tensor([
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ])
    targets = torch.tensor([
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ])
    
    # Update metrics
    metrics.update(predictions, targets)
    results = metrics.get_metrics()
    
    # Check perfect scores (with small tolerance for floating point precision)
    assert abs(results['precision'] - 1.0) < 1e-6
    assert abs(results['recall'] - 1.0) < 1e-6
    assert abs(results['f1_score'] - 1.0) < 1e-6
    assert abs(results['iou'] - 1.0) < 1e-6
    assert abs(results['overall_accuracy'] - 1.0) < 1e-6
    
    print("✓ Metrics test passed (perfect predictions)")
    
    # Test with imperfect predictions
    metrics.reset()
    predictions = torch.tensor([
        [0, 0, 1, 1],
        [0, 1, 1, 1]  # One false positive
    ])
    targets = torch.tensor([
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ])
    
    metrics.update(predictions, targets)
    results = metrics.get_metrics()
    
    # Check that metrics are between 0 and 1
    assert 0 <= results['precision'] <= 1
    assert 0 <= results['recall'] <= 1
    assert 0 <= results['f1_score'] <= 1
    assert 0 <= results['iou'] <= 1
    
    print("✓ Metrics test passed (imperfect predictions)")


def test_calculate_metrics():
    """Test calculate_metrics function."""
    batch_size = 2
    height, width = 16, 16
    num_classes = 2
    
    # Create dummy predictions and targets
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, targets, num_classes=2)
    
    # Check that all expected metrics are present
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'iou' in metrics
    assert 'overall_accuracy' in metrics
    
    # Check that metrics are in valid range
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1
    assert 0 <= metrics['iou'] <= 1
    assert 0 <= metrics['overall_accuracy'] <= 1
    
    print("✓ calculate_metrics test passed")


if __name__ == '__main__':
    test_dice_loss()
    test_focal_loss()
    test_combined_loss()
    test_metrics()
    test_calculate_metrics()
    print("\n✓ All utility tests passed!")
