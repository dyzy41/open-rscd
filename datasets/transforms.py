"""
Data augmentation and transformation utilities.
"""

import torch
import numpy as np
import cv2
from torchvision import transforms as T


class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class ToTensor:
    """Convert images and labels to PyTorch tensors."""
    def __call__(self, sample):
        img_A, img_B, label = sample['image_A'], sample['image_B'], sample['label']
        
        # Convert images from HWC to CHW format
        img_A = torch.from_numpy(img_A.transpose(2, 0, 1)).float() / 255.0
        img_B = torch.from_numpy(img_B.transpose(2, 0, 1)).float() / 255.0
        label = torch.from_numpy(label).long()
        
        sample['image_A'] = img_A
        sample['image_B'] = img_B
        sample['label'] = label
        
        return sample


class Normalize:
    """Normalize images with mean and std."""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean).reshape(3, 1, 1)
        self.std = np.array(std).reshape(3, 1, 1)
    
    def __call__(self, sample):
        img_A, img_B = sample['image_A'], sample['image_B']
        
        # Normalize
        img_A = (img_A - self.mean) / self.std
        img_B = (img_B - self.mean) / self.std
        
        sample['image_A'] = img_A
        sample['image_B'] = img_B
        
        return sample


class RandomHorizontalFlip:
    """Randomly flip images and labels horizontally."""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        if np.random.random() < self.p:
            img_A = np.flip(sample['image_A'], axis=1).copy()
            img_B = np.flip(sample['image_B'], axis=1).copy()
            label = np.flip(sample['label'], axis=1).copy()
            
            sample['image_A'] = img_A
            sample['image_B'] = img_B
            sample['label'] = label
        
        return sample


class RandomVerticalFlip:
    """Randomly flip images and labels vertically."""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        if np.random.random() < self.p:
            img_A = np.flip(sample['image_A'], axis=0).copy()
            img_B = np.flip(sample['image_B'], axis=0).copy()
            label = np.flip(sample['label'], axis=0).copy()
            
            sample['image_A'] = img_A
            sample['image_B'] = img_B
            sample['label'] = label
        
        return sample


class RandomRotation:
    """Randomly rotate images and labels by 90, 180, or 270 degrees."""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        if np.random.random() < self.p:
            k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
            
            img_A = np.rot90(sample['image_A'], k, axes=(0, 1)).copy()
            img_B = np.rot90(sample['image_B'], k, axes=(0, 1)).copy()
            label = np.rot90(sample['label'], k, axes=(0, 1)).copy()
            
            sample['image_A'] = img_A
            sample['image_B'] = img_B
            sample['label'] = label
        
        return sample


class RandomCrop:
    """Randomly crop images and labels."""
    def __init__(self, size):
        self.size = size
    
    def __call__(self, sample):
        img_A, img_B, label = sample['image_A'], sample['image_B'], sample['label']
        h, w = img_A.shape[:2]
        
        if h > self.size and w > self.size:
            top = np.random.randint(0, h - self.size)
            left = np.random.randint(0, w - self.size)
            
            img_A = img_A[top:top+self.size, left:left+self.size]
            img_B = img_B[top:top+self.size, left:left+self.size]
            label = label[top:top+self.size, left:left+self.size]
            
            sample['image_A'] = img_A
            sample['image_B'] = img_B
            sample['label'] = label
        
        return sample


def get_train_transforms(crop_size=256):
    """Get training data transforms."""
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(p=0.5),
        RandomCrop(crop_size),
        ToTensor(),
        Normalize()
    ])


def get_val_transforms():
    """Get validation data transforms."""
    return Compose([
        ToTensor(),
        Normalize()
    ])
