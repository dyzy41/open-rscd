"""
Evaluation and inference script for Remote Sensing Change Detection.
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models import build_model
from datasets import get_dataset, get_val_transforms
from utils import Metrics, visualize_prediction, save_change_map


def evaluate(model, dataloader, device, save_dir=None, save_predictions=False):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader
        device: Device to evaluate on
        save_dir: Directory to save predictions
        save_predictions: Whether to save prediction visualizations
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    metrics = Metrics(num_classes=2)
    
    if save_predictions and save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    pbar = tqdm(dataloader, desc='Evaluating')
    with torch.no_grad():
        for idx, sample in enumerate(pbar):
            img_A = sample['image_A'].to(device)
            img_B = sample['image_B'].to(device)
            labels = sample['label'].to(device)
            filename = sample['filename'][0] if isinstance(sample['filename'], list) else sample['filename']
            
            # Forward pass
            outputs = model(img_A, img_B)
            predictions = torch.argmax(outputs, dim=1)
            
            # Update metrics
            metrics.update(predictions, labels)
            
            # Save predictions
            if save_predictions and save_dir:
                # Get data for visualization
                img_A_np = img_A[0].cpu().numpy().transpose(1, 2, 0)
                img_B_np = img_B[0].cpu().numpy().transpose(1, 2, 0)
                pred_np = predictions[0].cpu().numpy()
                label_np = labels[0].cpu().numpy()
                
                # Save visualization
                vis_path = os.path.join(save_dir, f'{os.path.splitext(filename)[0]}_vis.png')
                visualize_prediction(img_A_np, img_B_np, pred_np, label_np, vis_path)
                
                # Save prediction map
                pred_path = os.path.join(save_dir, f'{os.path.splitext(filename)[0]}_pred.png')
                save_change_map(pred_np, pred_path)
    
    # Calculate final metrics
    final_metrics = metrics.get_metrics()
    
    return final_metrics


def inference(model, image_A_path, image_B_path, device, output_path=None):
    """
    Run inference on a pair of images.
    
    Args:
        model: The model to use
        image_A_path: Path to time 1 image
        image_B_path: Path to time 2 image
        device: Device to run inference on
        output_path: Path to save output
        
    Returns:
        np.ndarray: Predicted change map
    """
    from PIL import Image
    from datasets.transforms import ToTensor, Normalize, Compose
    
    # Load images
    img_A = Image.open(image_A_path).convert('RGB')
    img_B = Image.open(image_B_path).convert('RGB')
    
    img_A = np.array(img_A)
    img_B = np.array(img_B)
    
    # Prepare transforms
    transform = Compose([ToTensor(), Normalize()])
    
    # Create dummy sample
    sample = {
        'image_A': img_A,
        'image_B': img_B,
        'label': np.zeros(img_A.shape[:2], dtype=np.uint8),
        'filename': 'inference'
    }
    
    sample = transform(sample)
    
    # Add batch dimension
    img_A_tensor = sample['image_A'].unsqueeze(0).to(device)
    img_B_tensor = sample['image_B'].unsqueeze(0).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(img_A_tensor, img_B_tensor)
        predictions = torch.argmax(outputs, dim=1)
    
    # Convert to numpy
    pred_np = predictions[0].cpu().numpy()
    
    # Save output
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save visualization
        img_A_vis = sample['image_A'].numpy().transpose(1, 2, 0)
        img_B_vis = sample['image_B'].numpy().transpose(1, 2, 0)
        visualize_prediction(img_A_vis, img_B_vis, pred_np, save_path=output_path)
        
        # Save prediction map
        pred_map_path = output_path.replace('.png', '_pred.png')
        save_change_map(pred_np, pred_map_path)
        
        print(f"Saved prediction to: {output_path}")
        print(f"Saved prediction map to: {pred_map_path}")
    
    return pred_np


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Change Detection Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='evaluate',
                       choices=['evaluate', 'inference'],
                       help='Mode: evaluate on dataset or inference on image pair')
    parser.add_argument('--image_A', type=str, default=None,
                       help='Path to time 1 image (for inference mode)')
    parser.add_argument('--image_B', type=str, default=None,
                       help='Path to time 2 image (for inference mode)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model
    model = build_model(config['model'])
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from: {args.checkpoint}")
    
    if args.mode == 'evaluate':
        # Create dataset
        test_dataset = get_dataset(
            config['dataset']['name'],
            config['dataset']['root'],
            split='test',
            transform=get_val_transforms()
        )
        
        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=True
        )
        
        print(f"Test samples: {len(test_dataset)}")
        
        # Evaluate
        output_dir = args.output if args.output else config['evaluation']['output_dir']
        metrics = evaluate(
            model, test_loader, device,
            save_dir=output_dir,
            save_predictions=config['evaluation']['save_predictions']
        )
        
        # Print metrics
        print("\nEvaluation Results:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"IoU: {metrics['iou']:.4f}")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        
    elif args.mode == 'inference':
        if not args.image_A or not args.image_B:
            raise ValueError("Please provide --image_A and --image_B for inference mode")
        
        output_path = args.output if args.output else 'results/inference.png'
        
        # Run inference
        pred = inference(model, args.image_A, args.image_B, device, output_path)
        print(f"Inference completed. Change pixels: {pred.sum()}")


if __name__ == '__main__':
    main()
