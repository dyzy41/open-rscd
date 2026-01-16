"""
Exchange module for Remote Sensing Change Detection.
This module implements the core exchange mechanism for feature interaction
between bi-temporal images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExchangeBlock(nn.Module):
    """
    Exchange Block for feature interaction between bi-temporal features.
    
    Args:
        in_channels (int): Number of input channels
        ratio (int): Channel reduction ratio for efficient exchange
    """
    def __init__(self, in_channels, ratio=4):
        super(ExchangeBlock, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        
        # Channel attention for exchange
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2):
        """
        Forward pass for exchange block.
        
        Args:
            x1 (torch.Tensor): Features from time1 image [B, C, H, W]
            x2 (torch.Tensor): Features from time2 image [B, C, H, W]
            
        Returns:
            tuple: Exchanged features (x1_out, x2_out)
        """
        batch_size, channels, _, _ = x1.size()
        
        # Global context pooling
        x1_avg = self.avg_pool(x1).view(batch_size, channels)
        x1_max = self.max_pool(x1).view(batch_size, channels)
        x2_avg = self.avg_pool(x2).view(batch_size, channels)
        x2_max = self.max_pool(x2).view(batch_size, channels)
        
        # Concatenate features from both time steps
        x1_global = torch.cat([x1_avg, x2_avg], dim=1)
        x2_global = torch.cat([x2_avg, x1_avg], dim=1)
        
        # Generate exchange weights
        w1 = self.sigmoid(self.fc(x1_global)).view(batch_size, channels, 1, 1)
        w2 = self.sigmoid(self.fc(x2_global)).view(batch_size, channels, 1, 1)
        
        # Exchange features
        x1_out = x1 * w1 + x2 * (1 - w1)
        x2_out = x2 * w2 + x1 * (1 - w2)
        
        return x1_out, x2_out


class SpatialExchange(nn.Module):
    """
    Spatial Exchange module for spatial feature interaction.
    
    Args:
        in_channels (int): Number of input channels
    """
    def __init__(self, in_channels):
        super(SpatialExchange, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        """
        Forward pass for spatial exchange.
        
        Args:
            x1 (torch.Tensor): Features from time1 image [B, C, H, W]
            x2 (torch.Tensor): Features from time2 image [B, C, H, W]
            
        Returns:
            tuple: Spatially exchanged features (x1_out, x2_out)
        """
        # Concatenate and process
        x_cat = torch.cat([x1, x2], dim=1)
        x_fused = self.conv(x_cat)
        
        # Generate spatial attention maps
        x1_avg = torch.mean(x1, dim=1, keepdim=True)
        x1_max, _ = torch.max(x1, dim=1, keepdim=True)
        x1_spatial = torch.cat([x1_avg, x1_max], dim=1)
        spatial_w1 = self.spatial_gate(x1_spatial)
        
        x2_avg = torch.mean(x2, dim=1, keepdim=True)
        x2_max, _ = torch.max(x2, dim=1, keepdim=True)
        x2_spatial = torch.cat([x2_avg, x2_max], dim=1)
        spatial_w2 = self.spatial_gate(x2_spatial)
        
        # Apply spatial exchange
        x1_out = x1 + x_fused * spatial_w1
        x2_out = x2 + x_fused * spatial_w2
        
        return x1_out, x2_out


class ExchangeModule(nn.Module):
    """
    Complete Exchange Module combining channel and spatial exchange.
    
    Args:
        in_channels (int): Number of input channels
        ratio (int): Channel reduction ratio
    """
    def __init__(self, in_channels, ratio=4):
        super(ExchangeModule, self).__init__()
        self.channel_exchange = ExchangeBlock(in_channels, ratio)
        self.spatial_exchange = SpatialExchange(in_channels)
        
    def forward(self, x1, x2):
        """
        Forward pass for complete exchange module.
        
        Args:
            x1 (torch.Tensor): Features from time1 image [B, C, H, W]
            x2 (torch.Tensor): Features from time2 image [B, C, H, W]
            
        Returns:
            tuple: Exchanged features (x1_out, x2_out)
        """
        # Channel exchange
        x1_ch, x2_ch = self.channel_exchange(x1, x2)
        
        # Spatial exchange
        x1_out, x2_out = self.spatial_exchange(x1_ch, x2_ch)
        
        return x1_out, x2_out
