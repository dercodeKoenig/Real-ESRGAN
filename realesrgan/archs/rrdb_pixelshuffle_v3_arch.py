import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights, make_layer, pixel_unshuffle


class ChannelAttention(nn.Module):
    def __init__(self, num_features, hidden_features):
        super(ChannelAttention, self).__init__()
            
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, hidden_features, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(hidden_features, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class ResidualDenseBlock(nn.Module):

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class HighwayRRDB(nn.Module):
    """Highway RRDB Block with feature compression/expansion."""

    def __init__(self, highway_channels, processing_channels=64, num_grow_ch=32, use_attention=True):
        super(HighwayRRDB, self).__init__()
        self.highway_channels = highway_channels
        self.processing_channels = processing_channels
        self.use_attention = use_attention
        
        # Compression: Highway → Processing
        self.compress = nn.Conv2d(highway_channels, processing_channels, kernel_size=1)
        
        # Dense processing blocks
        self.rdb1 = ResidualDenseBlock(processing_channels, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(processing_channels, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(processing_channels, num_grow_ch)
        
        # Expansion: Processing → Highway
        # Using 3x3 for spatial-aware mixing before expansion
        self.pre_expand = nn.Conv2d(processing_channels, processing_channels, kernel_size=3, padding=1)
        self.expand = nn.Conv2d(processing_channels, highway_channels, kernel_size=1)
        
        # Channel attention on highway features
        if self.use_attention:
            self.channel_attention = ChannelAttention(highway_channels, highway_channels // 2)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, highway_features):
        # Compress highway features for processing
        compressed = self.lrelu(self.compress(highway_features))
        
        # Dense processing
        processed = self.rdb1(compressed)
        processed = self.rdb2(processed)
        processed = self.rdb3(processed)
        
        # Spatial-aware pre-expansion
        processed = self.lrelu(self.pre_expand(processed))
        
        # Expand back to highway size
        expanded = self.expand(processed)
        
        # Apply channel attention
        if self.use_attention:
            expanded = self.channel_attention(expanded)
        
        return highway_features + expanded * 0.2


@ARCH_REGISTRY.register()
class RRDBNet_v3(nn.Module):
    """Highway RRDB Network with feature highway architecture.
    
    Main innovation: Each block processes features in a compact space (64 channels)
    while maintaining a rich feature highway (256-512 channels) throughout the network.
    
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        scale (int): Upsampling scale factor.
        highway_channels (int): Channel number for feature highway.
        processing_channels (int): Channel number for dense block processing.
        num_block (int): Number of RRDB blocks.
        num_grow_ch (int): Growth channels for dense blocks.
        use_attention (bool): Whether to use channel attention.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, highway_channels=256, 
                 processing_channels=64, num_block=23, num_grow_ch=32, num_pre_upscale_ch=128,
                 use_attention=True):
        super(RRDBNet_v3, self).__init__()
        self.scale = scale
        self.highway_channels = highway_channels
        
        # Handle different scales with pixel unshuffle
        upscale_factor = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
            upscale_factor = 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
            upscale_factor = 4

        assert num_pre_upscale_ch % (upscale_factor**2) == 0, "num_pre_upscale_ch must be divisible by (upscale_factor**2)"
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, highway_channels, 3, 1, 1)
   
        # Highway RRDB blocks
        self.highway_blocks = nn.ModuleList([
            HighwayRRDB(
                highway_channels=highway_channels,
                processing_channels=processing_channels,
                num_grow_ch=num_grow_ch,
                use_attention=use_attention
            ) for _ in range(num_block)
        ])

        # Final processing and upsampling
        self.conv_last_0 = nn.Conv2d(highway_channels, highway_channels, 3, 1, 1)
        self.conv_last_1 = nn.Conv2d(highway_channels, num_pre_upscale_ch, 3, 1, 1)
        self.conv_last_2 = nn.Conv2d(int(num_pre_upscale_ch / upscale_factor / upscale_factor), num_out_ch, 3, 1, 1)
        
        self.upsampler = nn.PixelShuffle(upscale_factor)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # Handle different scales
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
            
        # Initial feature extraction
        highway_feat = self.conv_first(feat)
        
        # Process through highway blocks
        for highway_block in self.highway_blocks:
            highway_feat = highway_block(highway_feat)
        
        
        highway_feat = self.conv_last_0(highway_feat)
        highway_feat = self.lrelu(highway_feat)
        
        feat = self.conv_last_1(highway_feat)
        feat = self.lrelu(feat)
        feat = self.upsampler(feat)
        feat = self.conv_last_2(feat)

        return feat
