from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


# --- Helper Residual Block ---
class ResidualBlock(nn.Module):
    """A standard two-convolutional-layer residual block."""

    def __init__(self, channels, norm_layer=spectral_norm):
        super().__init__()
        self.conv1 = norm_layer(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.conv2 = norm_layer(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return x + out  # Residual connection


# --- Advanced Discriminator ---
@ARCH_REGISTRY.register()
class AdvancedUNetDiscriminator(nn.Module):
    """
    Advanced U-Net Discriminator with Spectral Normalization, Dynamic Depth,
    and Residual Blocks.

    Arguments:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        depth (int): Number of downsampling/upsampling stages (U-Net depth). Default: 4.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, depth=4, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        self.depth = depth
        norm = spectral_norm

        # Initial Convolution (Feature Extraction)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(num_feat, norm_layer=norm),
            ResidualBlock(num_feat, norm_layer=norm)
        )

        # --- Downsampling Path ---
        self.down_blocks = nn.ModuleList()

        for i in range(depth):
            in_ch = num_feat * (2 ** i)
            out_ch = num_feat * (2 ** (i + 1))
            block = nn.Sequential(
                ResidualBlock(in_ch, norm_layer=norm),
                norm(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)),
                ResidualBlock(out_ch, norm_layer=norm)
            )
            self.down_blocks.append(block)

        # --- Bottleneck (Deepest Layer) ---
        bottleneck_ch = num_feat * (2 ** depth)
        self.bottleneck = nn.Sequential(
            ResidualBlock(bottleneck_ch, norm_layer=norm),
            ResidualBlock(bottleneck_ch, norm_layer=norm)
        )

        # --- Upsampling Path ---
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch = num_feat * (2 ** (depth - i))
            out_ch = num_feat * (2 ** (depth - 1 - i))

            block = nn.Sequential(
                ResidualBlock(in_ch, norm_layer=norm),
                ResidualBlock(in_ch, norm_layer=norm),
                norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)),
            )
            self.up_blocks.append(block)

        # --- Output Layers (Extra Convolutions) ---
        self.extra_convs = nn.Sequential(
            ResidualBlock(num_feat, norm_layer=norm),
            ResidualBlock(num_feat, norm_layer=norm),
            nn.Conv2d(num_feat, 1, 3, 1, 1)  # Final classification layer
        )

    def forward(self, x):

        x_stages = []  # Store outputs for skip connections

        # Initial feature extraction
        x = self.initial_conv(x)

        # Downsampling
        for i, down_block in enumerate(self.down_blocks):
            x_stages.append(x)
            x = down_block(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling
        for i, up_block in enumerate(self.up_blocks):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x =up_block(x)

            if self.skip_connection:
                x = x + x_stages[-1]
                del x_stages[-1]


        # Extra Convolutions / Output
        out = self.extra_convs(x)
        return out