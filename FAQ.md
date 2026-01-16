# Frequently Asked Questions (FAQ)

## General Questions

### Q: What is ExchangeNet?
A: ExchangeNet is a deep learning model for remote sensing change detection that uses novel exchange mechanisms to facilitate effective interaction between bi-temporal features from satellite or aerial imagery.

### Q: What types of changes can this model detect?
A: The model is primarily designed for building change detection but can be adapted for other types of change detection tasks such as deforestation, urban expansion, disaster assessment, etc.

## Installation

### Q: What are the system requirements?
A: 
- Python 3.8 or higher
- PyTorch 2.0 or higher
- 8GB+ RAM (16GB+ recommended)
- GPU with CUDA support (optional but recommended for training)

### Q: How do I install the dependencies?
A: Run `pip install -r requirements.txt` in the project directory.

## Usage

### Q: How do I prepare my own dataset?
A: Organize your data in the following structure:
```
your_dataset/
├── A/          # Time 1 images
├── B/          # Time 2 images
└── label/      # Change masks (binary: 0=no change, 1=change)
```

### Q: Can I use different backbone networks?
A: Yes, the configuration file supports ResNet18, ResNet34, and ResNet50. You can modify `configs/config.yaml` to select your preferred backbone.

### Q: How long does training take?
A: Training time depends on:
- Dataset size
- Hardware (GPU vs CPU)
- Batch size and epochs
Typically, training for 100 epochs on LEVIR-CD with a single GPU takes 2-4 hours.

### Q: What if I get out-of-memory errors?
A: Try reducing the batch size in the configuration file or using smaller input image sizes.

## Training

### Q: How do I resume training from a checkpoint?
A: Set the `resume` field in the config file to point to your checkpoint:
```yaml
training:
  resume: './checkpoints/checkpoint_epoch_50.pth'
```

### Q: How often are checkpoints saved?
A: Checkpoints are saved every `save_freq` epochs (default: 10) and the best model is automatically saved based on validation F1 score.

### Q: Can I use custom loss functions?
A: Yes, you can implement custom loss functions in `utils/losses.py` and add them to the loss factory function.

## Evaluation

### Q: How do I evaluate my trained model?
A: Use the evaluation script:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --mode evaluate
```

### Q: What metrics are reported?
A: The model reports:
- Precision
- Recall
- F1 Score
- IoU (Intersection over Union)
- Overall Accuracy

### Q: How do I run inference on my own images?
A: Use inference mode:
```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --mode inference \
    --image_A time1.png \
    --image_B time2.png \
    --output prediction.png
```

## Troubleshooting

### Q: Model training is very slow
A: 
- Ensure you're using a GPU (check with `torch.cuda.is_available()`)
- Increase batch size if memory allows
- Reduce number of workers if I/O is the bottleneck

### Q: Validation metrics are poor
A: 
- Check if your dataset is correctly labeled
- Try different hyperparameters (learning rate, batch size)
- Consider using a larger backbone (ResNet50 instead of ResNet18)
- Ensure sufficient training epochs

### Q: How do I cite this work?
A: Please cite the original paper:
```bibtex
@article{exchange2026,
  title={Exchange Is All You Need for Remote Sensing Change Detection},
  author={},
  journal={arXiv preprint arXiv:2601.07805},
  year={2026}
}
```

## Contributing

### Q: How can I contribute to this project?
A: Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Q: I found a bug. What should I do?
A: Please open an issue on GitHub with:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, PyTorch version)
