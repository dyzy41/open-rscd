"""
Evaluation metrics for change detection.
"""

import numpy as np
import torch


class Metrics:
    """
    Calculate evaluation metrics for change detection.
    """
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, predictions, targets):
        """
        Update confusion matrix with new predictions.
        
        Args:
            predictions (torch.Tensor or np.ndarray): Predicted labels [B, H, W]
            targets (torch.Tensor or np.ndarray): Ground truth labels [B, H, W]
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Update confusion matrix
        for pred, target in zip(predictions, targets):
            if target < self.num_classes and pred < self.num_classes:
                self.confusion_matrix[int(target), int(pred)] += 1
    
    def get_metrics(self):
        """
        Calculate all metrics from confusion matrix.
        
        Returns:
            dict: Dictionary of metric values
        """
        # For binary change detection (class 0: no change, class 1: change)
        TN = self.confusion_matrix[0, 0]
        FP = self.confusion_matrix[0, 1]
        FN = self.confusion_matrix[1, 0]
        TP = self.confusion_matrix[1, 1]
        
        # Calculate metrics
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (TP + FN + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        iou = TP / (TP + FP + FN + 1e-10)
        
        overall_accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'iou': iou,
            'overall_accuracy': overall_accuracy,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        }
        
        return metrics
    
    def get_confusion_matrix(self):
        """Get the confusion matrix."""
        return self.confusion_matrix


def calculate_metrics(predictions, targets, num_classes=2):
    """
    Calculate metrics for a batch of predictions.
    
    Args:
        predictions (torch.Tensor): Predicted logits [B, C, H, W]
        targets (torch.Tensor): Ground truth labels [B, H, W]
        num_classes (int): Number of classes
        
    Returns:
        dict: Dictionary of metric values
    """
    # Get predicted class
    if predictions.dim() == 4:
        predictions = torch.argmax(predictions, dim=1)
    
    metrics = Metrics(num_classes=num_classes)
    metrics.update(predictions, targets)
    
    return metrics.get_metrics()
