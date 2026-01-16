# Exchange Is All You Need for Remote Sensing Change Detection

Official implementation of "Exchange Is All You Need for Remote Sensing Change Detection" ([arXiv:2601.07805](https://arxiv.org/abs/2601.07805)).

## Overview

This repository provides a PyTorch implementation of ExchangeNet, a novel approach for remote sensing change detection that leverages exchange mechanisms for effective bi-temporal feature interaction.

### Key Features

- **Exchange Module**: Novel channel and spatial exchange mechanisms for bi-temporal feature interaction
- **Multi-scale Architecture**: Hierarchical feature extraction with ResNet backbones
- **Flexible Framework**: Support for multiple datasets and easy configuration
- **Comprehensive Utilities**: Built-in data augmentation, visualization, and evaluation tools

## Architecture

The ExchangeNet architecture consists of:
- **Siamese Encoder**: Shared ResNet backbone for feature extraction
- **Exchange Modules**: Multi-scale channel and spatial exchange at different feature levels
- **Decoder**: Progressive upsampling with skip connections
- **Segmentation Head**: Final change map generation

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (optional, for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/dyzy41/open-rscd.git
cd open-rscd

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

The framework supports standard remote sensing change detection datasets like LEVIR-CD and WHU-CD.

### Expected Directory Structure

```
data/
├── LEVIR-CD/
│   ├── A/          # Time 1 images
│   ├── B/          # Time 2 images
│   └── label/      # Change masks
├── WHU-CD/
│   ├── A/
│   ├── B/
│   └── label/
```

### Supported Datasets

- **LEVIR-CD**: Building change detection dataset
- **WHU-CD**: Aerial image change detection dataset
- **Custom datasets**: Follow the same directory structure

## Usage

### Training

```bash
# Train with default configuration
python train.py --config configs/config.yaml

# Train with custom settings
python train.py --config your_config.yaml
```

### Evaluation

```bash
# Evaluate on test set
python evaluate.py --checkpoint checkpoints/best_model.pth --mode evaluate

# Inference on image pair
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --mode inference \
    --image_A path/to/time1.png \
    --image_B path/to/time2.png \
    --output results/prediction.png
```

## Configuration

Edit `configs/config.yaml` to customize:
- Model architecture (backbone, pretrained weights)
- Training hyperparameters (batch size, learning rate, epochs)
- Dataset settings (name, root directory, crop size)
- Loss functions and optimizers

Example configuration:

```yaml
model:
  backbone: 'resnet18'
  pretrained: true
  num_classes: 2

training:
  batch_size: 8
  lr: 0.001
  epochs: 100
  loss: 'combined'
```

## Model Zoo

Coming soon - Pre-trained models on popular datasets.

## Results

Example results on LEVIR-CD dataset:

| Model | Precision | Recall | F1 | IoU |
|-------|-----------|--------|-----|-----|
| ExchangeNet (ResNet18) | TBD | TBD | TBD | TBD |
| ExchangeNet (ResNet50) | TBD | TBD | TBD | TBD |

## Project Structure

```
open-rscd/
├── models/              # Model architectures
│   ├── exchange.py      # Exchange modules
│   ├── backbone.py      # Encoder/decoder
│   └── model.py         # Main model
├── datasets/            # Data loading
│   ├── dataset.py       # Dataset classes
│   └── transforms.py    # Data augmentation
├── utils/               # Utilities
│   ├── losses.py        # Loss functions
│   ├── metrics.py       # Evaluation metrics
│   └── visualization.py # Visualization tools
├── configs/             # Configuration files
├── train.py             # Training script
└── evaluate.py          # Evaluation script
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{exchange2026,
  title={Exchange Is All You Need for Remote Sensing Change Detection},
  author={},
  journal={arXiv preprint arXiv:2601.07805},
  year={2026}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- PyTorch team for the deep learning framework
- Authors of LEVIR-CD and WHU-CD datasets
- Open source community for various tools and utilities

## Contact

For questions and issues, please open an issue on GitHub.
