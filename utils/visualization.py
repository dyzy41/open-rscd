"""
Visualization utilities for change detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch


def visualize_prediction(image_A, image_B, prediction, target=None, save_path=None):
    """
    Visualize change detection results.
    
    Args:
        image_A (np.ndarray or torch.Tensor): Time1 image [H, W, 3] or [3, H, W]
        image_B (np.ndarray or torch.Tensor): Time2 image [H, W, 3] or [3, H, W]
        prediction (np.ndarray or torch.Tensor): Predicted change map [H, W]
        target (np.ndarray or torch.Tensor): Ground truth change map [H, W]
        save_path (str): Path to save visualization
    """
    # Convert tensors to numpy
    if isinstance(image_A, torch.Tensor):
        image_A = image_A.cpu().numpy()
        if image_A.shape[0] == 3:
            image_A = image_A.transpose(1, 2, 0)
    
    if isinstance(image_B, torch.Tensor):
        image_B = image_B.cpu().numpy()
        if image_B.shape[0] == 3:
            image_B = image_B.transpose(1, 2, 0)
    
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    if target is not None and isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Normalize images to [0, 1]
    image_A = (image_A - image_A.min()) / (image_A.max() - image_A.min() + 1e-8)
    image_B = (image_B - image_B.min()) / (image_B.max() - image_B.min() + 1e-8)
    
    # Create figure
    if target is not None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[3].imshow(target, cmap='gray')
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(image_A)
    axes[0].set_title('Time 1')
    axes[0].axis('off')
    
    axes[1].imshow(image_B)
    axes[1].set_title('Time 2')
    axes[1].axis('off')
    
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def overlay_change_map(image, change_map, alpha=0.5):
    """
    Overlay change map on image.
    
    Args:
        image (np.ndarray): Input image [H, W, 3]
        change_map (np.ndarray): Binary change map [H, W]
        alpha (float): Transparency of overlay
        
    Returns:
        np.ndarray: Image with overlaid change map
    """
    # Normalize image
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = (image * 255).astype(np.uint8)
    
    # Create colored overlay (red for changes)
    overlay = np.zeros_like(image)
    overlay[:, :, 0] = change_map * 255  # Red channel
    
    # Blend
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    return result


def save_change_map(change_map, save_path):
    """
    Save change map as image.
    
    Args:
        change_map (np.ndarray or torch.Tensor): Change map [H, W]
        save_path (str): Path to save image
    """
    if isinstance(change_map, torch.Tensor):
        change_map = change_map.cpu().numpy()
    
    # Convert to 0-255 range
    change_map = (change_map * 255).astype(np.uint8)
    
    cv2.imwrite(save_path, change_map)


def create_comparison_grid(results, save_path=None):
    """
    Create a grid comparing multiple results.
    
    Args:
        results (list): List of dictionaries with keys 'image_A', 'image_B', 'prediction', 'target'
        save_path (str): Path to save grid
    """
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Time 1 image
        axes[i, 0].imshow(result['image_A'])
        if i == 0:
            axes[i, 0].set_title('Time 1')
        axes[i, 0].axis('off')
        
        # Time 2 image
        axes[i, 1].imshow(result['image_B'])
        if i == 0:
            axes[i, 1].set_title('Time 2')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(result['prediction'], cmap='gray')
        if i == 0:
            axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Ground truth
        axes[i, 3].imshow(result['target'], cmap='gray')
        if i == 0:
            axes[i, 3].set_title('Ground Truth')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
