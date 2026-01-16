"""
Main change detection model integrating Exchange modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNetEncoder, Decoder
from .exchange import ExchangeModule


class ExchangeNet(nn.Module):
    """
    Exchange-based Change Detection Network.
    
    This is the main model that implements "Exchange Is All You Need" 
    for remote sensing change detection.
    
    Args:
        backbone (str): Backbone network type
        pretrained (bool): Whether to use pretrained weights
        num_classes (int): Number of output classes (typically 2 for binary change detection)
    """
    def __init__(self, backbone='resnet18', pretrained=True, num_classes=2):
        super(ExchangeNet, self).__init__()
        
        # Shared encoder for both temporal images
        self.encoder = ResNetEncoder(backbone=backbone, pretrained=pretrained)
        
        # Exchange modules at multiple scales
        self.exchange1 = ExchangeModule(self.encoder.channels[1])
        self.exchange2 = ExchangeModule(self.encoder.channels[2])
        self.exchange3 = ExchangeModule(self.encoder.channels[3])
        self.exchange4 = ExchangeModule(self.encoder.channels[4])
        
        # Difference modules
        self.diff_conv1 = nn.Sequential(
            nn.Conv2d(self.encoder.channels[1], self.encoder.channels[1], 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(self.encoder.channels[1]),
            nn.ReLU(inplace=True)
        )
        self.diff_conv2 = nn.Sequential(
            nn.Conv2d(self.encoder.channels[2], self.encoder.channels[2], 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(self.encoder.channels[2]),
            nn.ReLU(inplace=True)
        )
        self.diff_conv3 = nn.Sequential(
            nn.Conv2d(self.encoder.channels[3], self.encoder.channels[3], 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(self.encoder.channels[3]),
            nn.ReLU(inplace=True)
        )
        self.diff_conv4 = nn.Sequential(
            nn.Conv2d(self.encoder.channels[4], self.encoder.channels[4], 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(self.encoder.channels[4]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = Decoder(
            encoder_channels=self.encoder.channels,
            decoder_channels=[256, 128, 64, 32]
        )
        
        # Final segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )
        
    def forward(self, x1, x2):
        """
        Forward pass for change detection.
        
        Args:
            x1 (torch.Tensor): Time1 image [B, 3, H, W]
            x2 (torch.Tensor): Time2 image [B, 3, H, W]
            
        Returns:
            torch.Tensor: Change map logits [B, num_classes, H, W]
        """
        # Extract features from both temporal images
        feats1 = self.encoder(x1)
        feats2 = self.encoder(x2)
        
        # Apply exchange at multiple scales and compute differences
        # Exchange at level 1
        feats1[1], feats2[1] = self.exchange1(feats1[1], feats2[1])
        diff1 = torch.abs(feats1[1] - feats2[1])
        diff1 = self.diff_conv1(diff1)
        
        # Exchange at level 2
        feats1[2], feats2[2] = self.exchange2(feats1[2], feats2[2])
        diff2 = torch.abs(feats1[2] - feats2[2])
        diff2 = self.diff_conv2(diff2)
        
        # Exchange at level 3
        feats1[3], feats2[3] = self.exchange3(feats1[3], feats2[3])
        diff3 = torch.abs(feats1[3] - feats2[3])
        diff3 = self.diff_conv3(diff3)
        
        # Exchange at level 4
        feats1[4], feats2[4] = self.exchange4(feats1[4], feats2[4])
        diff4 = torch.abs(feats1[4] - feats2[4])
        diff4 = self.diff_conv4(diff4)
        
        # Prepare difference features for decoder
        diff_features = [feats1[0], diff1, diff2, diff3, diff4]
        
        # Decode to get change map
        x = self.decoder(diff_features)
        
        # Upsample to original resolution
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        
        # Generate final change map
        change_map = self.segmentation_head(x)
        
        return change_map


def build_model(config):
    """
    Build change detection model from configuration.
    
    Args:
        config (dict): Model configuration
        
    Returns:
        ExchangeNet: Instantiated model
    """
    model = ExchangeNet(
        backbone=config.get('backbone', 'resnet18'),
        pretrained=config.get('pretrained', True),
        num_classes=config.get('num_classes', 2)
    )
    return model
