"""
Dataset loaders for Remote Sensing Change Detection.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class RSCDDataset(Dataset):
    """
    Remote Sensing Change Detection Dataset.
    
    Expected directory structure:
    root/
        A/  (time1 images)
        B/  (time2 images)
        label/  (change masks)
        
    Args:
        root (str): Root directory of dataset
        split (str): Dataset split ('train', 'val', 'test')
        transform (callable): Transforms to apply to images
    """
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        
        self.img_dir_A = os.path.join(root, 'A')
        self.img_dir_B = os.path.join(root, 'B')
        self.label_dir = os.path.join(root, 'label')
        
        # Get list of image files
        self.image_files = sorted([
            f for f in os.listdir(self.img_dir_A) 
            if f.endswith(('.png', '.jpg', '.tif'))
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            dict: Dictionary containing 'image_A', 'image_B', 'label', and 'filename'
        """
        filename = self.image_files[idx]
        
        # Load images
        img_A = Image.open(os.path.join(self.img_dir_A, filename)).convert('RGB')
        img_B = Image.open(os.path.join(self.img_dir_B, filename)).convert('RGB')
        label = Image.open(os.path.join(self.label_dir, filename)).convert('L')
        
        # Convert to numpy arrays
        img_A = np.array(img_A)
        img_B = np.array(img_B)
        label = np.array(label)
        
        # Binarize label (0: no change, 1: change)
        label = (label > 0).astype(np.uint8)
        
        sample = {
            'image_A': img_A,
            'image_B': img_B,
            'label': label,
            'filename': filename
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class LEVIRCDDataset(RSCDDataset):
    """
    LEVIR-CD Dataset loader.
    
    LEVIR-CD is a widely used building change detection dataset.
    """
    def __init__(self, root, split='train', transform=None):
        super().__init__(root, split, transform)


class WHUCDDataset(RSCDDataset):
    """
    WHU-CD Dataset loader.
    
    WHU-CD contains aerial images for building change detection.
    """
    def __init__(self, root, split='train', transform=None):
        super().__init__(root, split, transform)


def get_dataset(dataset_name, root, split='train', transform=None):
    """
    Factory function to get dataset by name.
    
    Args:
        dataset_name (str): Name of dataset ('LEVIR-CD', 'WHU-CD', 'RSCD')
        root (str): Root directory of dataset
        split (str): Dataset split
        transform (callable): Transforms to apply
        
    Returns:
        Dataset: Instantiated dataset
    """
    datasets = {
        'LEVIR-CD': LEVIRCDDataset,
        'WHU-CD': WHUCDDataset,
        'RSCD': RSCDDataset
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return datasets[dataset_name](root, split, transform)
