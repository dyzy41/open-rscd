"""
Training script for Remote Sensing Change Detection.
"""

import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models import build_model
from datasets import get_dataset, get_train_transforms, get_val_transforms
from utils import get_loss_function, Metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        dict: Training metrics
    """
    model.train()
    metrics = Metrics(num_classes=2)
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, sample in enumerate(pbar):
        img_A = sample['image_A'].to(device)
        img_B = sample['image_B'].to(device)
        labels = sample['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(img_A, img_B)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        metrics.update(predictions, labels)
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = metrics.get_metrics()
    epoch_metrics['loss'] = epoch_loss
    
    return epoch_metrics


def validate(model, dataloader, criterion, device, epoch):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        dict: Validation metrics
    """
    model.eval()
    metrics = Metrics(num_classes=2)
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    with torch.no_grad():
        for sample in pbar:
            img_A = sample['image_A'].to(device)
            img_B = sample['image_B'].to(device)
            labels = sample['label'].to(device)
            
            # Forward pass
            outputs = model(img_A, img_B)
            loss = criterion(outputs, labels)
            
            # Update metrics
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            metrics.update(predictions, labels)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_metrics = metrics.get_metrics()
    epoch_metrics['loss'] = epoch_loss
    
    return epoch_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Change Detection Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = get_dataset(
        config['dataset']['name'],
        config['dataset']['root'],
        split='train',
        transform=get_train_transforms(config['dataset']['crop_size'])
    )
    
    val_dataset = get_dataset(
        config['dataset']['name'],
        config['dataset']['root'],
        split='val',
        transform=get_val_transforms()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Build model
    model = build_model(config['model'])
    model = model.to(device)
    
    # Loss function
    criterion = get_loss_function(config['training']['loss'])
    
    # Optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['lr'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
    
    # Learning rate scheduler
    if config['training']['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['lr_decay_step'],
            gamma=config['training']['lr_decay_gamma']
        )
    elif config['training']['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    
    # Resume from checkpoint
    start_epoch = 0
    best_f1 = 0.0
    if config['training']['resume']:
        checkpoint = torch.load(config['training']['resume'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Create checkpoint directory
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    # Training loop
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"F1: {train_metrics['f1_score']:.4f}, "
              f"IoU: {train_metrics['iou']:.4f}")
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch + 1
        )
        print(f"Val - Loss: {val_metrics['loss']:.4f}, "
              f"F1: {val_metrics['f1_score']:.4f}, "
              f"IoU: {val_metrics['iou']:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_freq'] == 0:
            checkpoint_path = os.path.join(
                config['training']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch + 1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_metrics['f1_score'] > best_f1:
            best_f1 = val_metrics['f1_score']
            best_model_path = os.path.join(
                config['training']['checkpoint_dir'],
                'best_model.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config
            }, best_model_path)
            print(f"Saved best model with F1: {best_f1:.4f}")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
