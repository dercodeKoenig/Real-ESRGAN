import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights, make_layer, pixel_unshuffle


import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchAttentionModule(nn.Module):
    def __init__(self, in_channels, patch_size=32, embed_dim=128, heads = 4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Project patch tokens to embedding space
        self.token_embed = nn.Linear(in_channels, embed_dim)

        # Simple self-attention block
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=heads, batch_first=True)

        # Project back to original channels
        self.token_unembed = nn.Linear(embed_dim, in_channels)

    def forward(self, x):
        B, C, H_0, W_0 = x.shape
        x_orig = x  # Keep reference to original input

        #assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image must be divisible by patch size"

        # Pad to make divisible by patch_size
        pad_h = (self.patch_size - H_0 % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W_0 % self.patch_size) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            padded = True
        else:
            padded = False

        B, C, H, W = x.shape

        # Step 1: Split into patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # shape: (B, C, H//P, W//P, P, P)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # (B, N, C, P, P)

        # Step 2: Average each patch spatially → (B, N, C)
        tokens = patches.mean(dim=(-1, -2))  # global average over (P, P)

        # Step 3: Project tokens to embedding dim
        tokens = self.token_embed(tokens)  # (B, N, D)

        # Step 4: Self-attention over patches
        tokens, _ = self.attn(tokens, tokens, tokens)  # (B, N, D)

        # Step 5: Project back and reshape
        tokens = self.token_unembed(tokens)  # (B, N, C)
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # Step 6: Reshape back to 2D grid
        out = tokens.view(B, h_patches, w_patches, C).permute(0, 3, 1, 2)  # (B, C, H//P, W//P)

        # Step 7: Upsample to original size
        out = F.interpolate(out, (H,W), mode='bilinear', align_corners=False)

        # Crop back to original size if padded
        if padded:
            out = out[:, :, :H_0, :W_0]

        # Optional: Residual add
        return out + x_orig  # same shape as input



class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

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
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32, heads = 4, patch_size = 32, embed_dim = 128):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.heads = heads
        if heads > 0:
          self.attn = PatchAttentionModule(in_channels=num_feat, patch_size=patch_size, heads = heads, embed_dim = embed_dim)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)

        if self.heads > 0:
          out_attn = self.attn(out)
          out = out + out_attn

        return out * 0.2 + x


@ARCH_REGISTRY.register()
class RRDBNet_pxlshuffle_attn(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32, heads=4, patch_size=32, embed_dim=128, clear_cache=False):
        super(RRDBNet_pxlshuffle_attn, self).__init__()
        self.scale = scale
        self.clear_cache = clear_cache
        if scale == 2:
            num_in_ch = num_in_ch * 4
            self.scale *= 2
        elif scale == 1:
            num_in_ch = num_in_ch * 16
            self.scale *= 4

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # Create RRDB blocks as ModuleList for sequential processing
        self.rrdb_blocks = nn.ModuleList()
        for i in range(num_block):
            self.rrdb_blocks.append(
                RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch, 
                     heads=heads, patch_size=patch_size, embed_dim=embed_dim)
            )
        
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # upsample
        self.conv_last = nn.Conv2d(num_feat, num_out_ch * scale * scale, 3, 1, 1)
        self.upsampler = nn.PixelShuffle(scale)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
            
        feat = self.conv_first(feat)
        body_feat = feat
        
        # Process RRDB blocks sequentially with optional memory cleanup
        for i, rrdb_block in enumerate(self.rrdb_blocks):
            body_feat = rrdb_block(body_feat)
            
            # Optional: Force garbage collection after every block if requested
            if self.clear_cache:
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Wait for all operations to complete
        
        # Apply residual connection from input to body
        body_feat = body_feat + feat
        feat = self.lrelu(self.conv_body(body_feat))
        
        # upsample
        feat = self.conv_last(feat)
        out = self.upsampler(feat)
        return out
