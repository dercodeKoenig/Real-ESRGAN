from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

# Define residual block
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=False))
    
    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2, inplace=True)
        return x + residual
        
@ARCH_REGISTRY.register()
class UNetDiscriminatorSN2(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN2, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        
        # Encoder
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv3_1 = norm(nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False))
        self.conv3_2 = norm(nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False))

        self.res_block_1 = ResBlock(num_feat)
        self.res_block_2 = ResBlock(num_feat * 2)
        self.res_block_3 = ResBlock(num_feat * 4)
        
        # Decoder
        self.conv3_up2 = norm(nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False))
        self.conv3_up1 = norm(nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        
        # Output layers
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)
    
    def forward(self, x):
        # Encoder
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        x3_1 = F.leaky_relu(self.conv3_1(x3), negative_slope=0.2, inplace=True)
        x3_2 = F.leaky_relu(self.conv3_2(x3_1), negative_slope=0.2, inplace=True)
        
        # Decoder
        # First upsample
        up = F.interpolate(x3_2, scale_factor=2, mode='bilinear', align_corners=False)
        up = F.leaky_relu(self.conv3_up2(up), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            up = up + x3_1
        
        # Second upsample  
        up = F.interpolate(up, scale_factor=2, mode='bilinear', align_corners=False)
        up = F.leaky_relu(self.conv3_up1(up), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            up = up + x3
        
        # Continue upsampling
        up = F.interpolate(up, scale_factor=2, mode='bilinear', align_corners=False)
        up = F.leaky_relu(self.conv4(up), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            up = up + x2
        up = self.res_block_3(up)
            
        up = F.interpolate(up, scale_factor=2, mode='bilinear', align_corners=False)
        up = F.leaky_relu(self.conv5(up), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            up = up + x1
        up = self.res_block_2(up)
            
        up = F.interpolate(up, scale_factor=2, mode='bilinear', align_corners=False)
        up = F.leaky_relu(self.conv6(up), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            up = up + x0
        up = self.res_block_1(up)
        
        # Output
        out = F.leaky_relu(self.conv7(up), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)
        return out
