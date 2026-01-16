"""
Backbone networks for feature extraction in change detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder for feature extraction.
    
    Args:
        backbone (str): ResNet variant ('resnet18', 'resnet34', 'resnet50')
        pretrained (bool): Whether to use pretrained weights
    """
    def __init__(self, backbone='resnet18', pretrained=True):
        super(ResNetEncoder, self).__init__()
        
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
    def forward(self, x):
        """
        Forward pass to extract multi-scale features.
        
        Args:
            x (torch.Tensor): Input image [B, 3, H, W]
            
        Returns:
            list: Multi-scale features [feat1, feat2, feat3, feat4, feat5]
        """
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # feat1
        
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)  # feat2
        
        x = self.layer2(x)
        features.append(x)  # feat3
        
        x = self.layer3(x)
        features.append(x)  # feat4
        
        x = self.layer4(x)
        features.append(x)  # feat5
        
        return features


class DecoderBlock(nn.Module):
    """
    Decoder block with upsampling and skip connections.
    
    Args:
        in_channels (int): Number of input channels
        skip_channels (int): Number of skip connection channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip=None):
        """
        Forward pass with upsampling and skip connection.
        
        Args:
            x (torch.Tensor): Input features [B, C_in, H, W]
            skip (torch.Tensor): Skip connection features [B, C_skip, H*2, W*2]
            
        Returns:
            torch.Tensor: Decoded features [B, C_out, H*2, W*2]
        """
        x = self.upsample(x)
        
        if skip is not None:
            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    """
    Decoder network for change detection.
    
    Args:
        encoder_channels (list): List of encoder output channels
        decoder_channels (list): List of decoder output channels
    """
    def __init__(self, encoder_channels, decoder_channels=[256, 128, 64, 32]):
        super(Decoder, self).__init__()
        
        # Reverse encoder channels for bottom-up decoding
        encoder_channels = encoder_channels[::-1]
        
        self.blocks = nn.ModuleList()
        
        # First decoder block (no skip connection)
        self.blocks.append(
            DecoderBlock(encoder_channels[0], 0, decoder_channels[0])
        )
        
        # Remaining decoder blocks with skip connections
        for i in range(1, len(decoder_channels)):
            skip_ch = encoder_channels[i] if i < len(encoder_channels) else 0
            self.blocks.append(
                DecoderBlock(decoder_channels[i-1], skip_ch, decoder_channels[i])
            )
        
    def forward(self, features):
        """
        Forward pass through decoder.
        
        Args:
            features (list): Multi-scale features from encoder (low to high res)
            
        Returns:
            torch.Tensor: Decoded features
        """
        # Reverse features for decoding (high to low level)
        features = features[::-1]
        
        x = features[0]
        x = self.blocks[0](x)
        
        for i, block in enumerate(self.blocks[1:], 1):
            skip = features[i] if i < len(features) else None
            x = block(x, skip)
        
        return x
