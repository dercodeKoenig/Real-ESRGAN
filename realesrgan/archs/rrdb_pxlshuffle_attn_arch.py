import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights, make_layer, pixel_unshuffle


import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import gc


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
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32, 
                 heads=4, patch_size=32, embed_dim=128, cpu_offload=False):
        super(RRDBNet_pxlshuffle_attn, self).__init__()
        self.scale = scale
        self.cpu_offload = cpu_offload
        
        if scale == 2:
            num_in_ch = num_in_ch * 4
            self.scale *= 2
        elif scale == 1:
            num_in_ch = num_in_ch * 16
            self.scale *= 4

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # Create RRDB blocks
        self.rrdb_blocks = nn.ModuleList()
        for i in range(num_block):
            block = RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch, 
                        heads=heads, patch_size=patch_size, embed_dim=embed_dim)
            self.rrdb_blocks.append(block)
        
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch * scale * scale, 3, 1, 1)
        self.upsampler = nn.PixelShuffle(scale)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def to(self, device):
        """Override to handle CPU offloading"""
        if self.cpu_offload:
            print("CPU offload is enabled!")
            self.target_device = device
            # Move everything to CPU initially
            self.conv_first.to('cpu')
            self.conv_body.to('cpu') 
            self.conv_last.to('cpu')
            self.upsampler.to('cpu')
            for block in self.rrdb_blocks:
                block.to('cpu')
        else:
            super().to(device)
        return self

    def _clear_memory(self):
        """Aggressive memory clearing for inference"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _move_to_cpu_safe(self, tensor):
        """Safely move tensor to CPU and clear GPU memory"""
        if tensor.device.type == 'cuda':
            cpu_tensor = tensor.cpu()
            del tensor
            self._clear_memory()
            return cpu_tensor
        return tensor

    @torch.no_grad()  # Ensure no gradients for inference
    def forward(self, x):
        # Handle input scaling
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        
        # Process first convolution
        if self.cpu_offload:
            self.conv_first.to(self.target_device)
            feat = self.conv_first(feat)
            self.conv_first.to('cpu')
            self._clear_memory()
        else:
            feat = self.conv_first(feat)

        # Store residual connection - move to CPU to free GPU memory
        if self.cpu_offload:
            residual = self._move_to_cpu_safe(feat.clone())
        else:
            residual = feat.clone()  # Clone to avoid in-place operations

        body_feat = feat
        
        # Process RRDB blocks with CPU offloading and intermediate CPU storage
        if self.cpu_offload:
            for i, rrdb_block in enumerate(tqdm(self.rrdb_blocks, desc="Processing RRDB blocks")):
                # Move block to GPU
                rrdb_block.to(self.target_device)
                
                # Process block
                new_body_feat = rrdb_block(body_feat)
                
                # Immediately move block back to CPU
                rrdb_block.to('cpu')
                
                # For intermediate blocks, store result on CPU to free GPU memory
                if i < len(self.rrdb_blocks) - 1:  # Not the last block
                    # Delete old body_feat first
                    del body_feat
                    # Move result to CPU, then back to GPU for next iteration
                    body_feat_cpu = self._move_to_cpu_safe(new_body_feat)
                    body_feat = body_feat_cpu.to(self.target_device)
                    del body_feat_cpu
                else:
                    # Last block - keep on GPU
                    del body_feat
                    body_feat = new_body_feat
                
                # Clear memory every block
                self._clear_memory()
        else:
            for rrdb_block in self.rrdb_blocks:
                body_feat = rrdb_block(body_feat)
        
        # Apply residual connection
        if self.cpu_offload:
            # Move residual back to GPU for addition
            residual = residual.to(self.target_device)
            body_feat = body_feat + residual
            # Explicitly delete residual to free memory
            del residual
            self._clear_memory()
        else:
            body_feat = body_feat + residual
            del residual  # Free residual even in non-CPU offload mode

        # Process conv_body
        if self.cpu_offload:
            self.conv_body.to(self.target_device)
            feat = self.conv_body(body_feat)
            del body_feat  # Free body_feat
            self.conv_body.to('cpu')
            self._clear_memory()
        else:
            feat = self.conv_body(body_feat)
            del body_feat  # Free body_feat

        # Apply LeakyReLU in-place to save memory
        feat = self.lrelu(feat)

        # Process final convolution
        if self.cpu_offload:
            self.conv_last.to(self.target_device)
            final_feat = self.conv_last(feat)
            del feat  # Free feat
            self.conv_last.to('cpu')
            self._clear_memory()
        else:
            final_feat = self.conv_last(feat)
            del feat  # Free feat

        # Upsampling
        if self.cpu_offload:
            self.upsampler.to(self.target_device)
            out = self.upsampler(final_feat)
            del final_feat  # Free final_feat
            self.upsampler.to('cpu')
            self._clear_memory()
        else:
            out = self.upsampler(final_feat)
            del final_feat  # Free final_feat
        
        return out
