"""
Loss functions for change detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Calculate Dice loss.
        
        Args:
            predictions (torch.Tensor): Predicted logits [B, C, H, W]
            targets (torch.Tensor): Ground truth labels [B, H, W]
            
        Returns:
            torch.Tensor: Dice loss value
        """
        # Apply softmax to get probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Get the change class (class 1)
        predictions = predictions[:, 1, :, :]
        
        # Flatten tensors
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1).float()
        
        # Calculate Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        Calculate Focal loss.
        
        Args:
            predictions (torch.Tensor): Predicted logits [B, C, H, W]
            targets (torch.Tensor): Ground truth labels [B, H, W]
            
        Returns:
            torch.Tensor: Focal loss value
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: Cross Entropy + Dice Loss.
    """
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, predictions, targets):
        """
        Calculate combined loss.
        
        Args:
            predictions (torch.Tensor): Predicted logits [B, C, H, W]
            targets (torch.Tensor): Ground truth labels [B, H, W]
            
        Returns:
            torch.Tensor: Combined loss value
        """
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        
        return self.ce_weight * ce + self.dice_weight * dice


def get_loss_function(loss_name='combined', **kwargs):
    """
    Factory function to get loss by name.
    
    Args:
        loss_name (str): Name of loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        nn.Module: Loss function
    """
    losses = {
        'ce': nn.CrossEntropyLoss,
        'dice': DiceLoss,
        'focal': FocalLoss,
        'combined': CombinedLoss
    }
    
    if loss_name not in losses:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return losses[loss_name](**kwargs)
