import torch
from torch import nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights

import math

from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb


class TimestepEmbedding(nn.Module):
    def __init__(self, t_emb_dim = 128, hidden_dim = 256):
        super().__init__()
        self.embedding_dim = t_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(t_emb_dim, hidden_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        # t: (B) scalar between 0 and 1. We scale to 1000 for standard embedding logic
        t = t * 1000
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # Shape: (B, embedding_dim)

        return self.mlp(emb)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, rope_dim=16):
        """
        dim: Input channel dimension (e.g., 768)
        heads: Number of attention heads
        dim_head: Dimension per head (e.g., 64)
        rope_dim: Dimension to apply rotary embedding to (usually dim_head // 4, should make 1/4 x and 1/4 y)
        """
        super().__init__()

        inner_dim = heads * dim_head

        self.heads = heads

        # 1. Project Input to Query, Key, Value
        # bias=False is standard for Q, K in modern Transformers (e.g., Llama, DiT)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 2. Output Projection
        self.to_out = nn.Linear(inner_dim, dim)

        # 3. Rotary Embedding
        # 'pixel' mode creates axial freqs for 2D grids automatically
        self.rope = RotaryEmbedding(dim=rope_dim, freqs_for='pixel')

    def forward(self, x):
        """
        x shape: (Batch, Height, Width, Channels)  <- Channel last required
        """
        b, h, w, c = x.shape # input is already channel last

        # --- PREPARE INPUT ---
        # 1. Project q, k, v
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # 2. Split heads and move to front
        # Shape: (B, Heads, H, W, Dim_Head)
        # We keep H, W separate for now to apply Axial RoPE easily
        q, k, v = map(lambda t: rearrange(t, 'b h w (n d) -> b n h w d', n=self.heads), qkv)

        # --- ROTARY EMBEDDING ---
        # Get axial frequencies for the current image size
        # freqs shape: (H, W, rope_dim)
        freqs = self.rope.get_axial_freqs(h, w)

        # Apply RoPE.
        # Broadcasting aligns (H, W) in freqs with (H, W) in q/k automatically.
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)

        # --- FLASH ATTENTION ---
        # 1. Flatten H and W into a single sequence for attention
        # Shape: (B, Heads, Seq_Len, Dim_Head) where Seq_Len = H * W
        q = rearrange(q, 'b n h w d -> b n (h w) d')
        k = rearrange(k, 'b n h w d -> b n (h w) d')
        v = rearrange(v, 'b n h w d -> b n (h w) d')

        # 2. Use the optimized PyTorch function
        out = F.scaled_dot_product_attention(q, k, v)

        # --- RECONSTRUCT ---
        # 1. Reshape back to 2D grid: (B, Heads, H*W, D) -> (B, Heads, H, W, D)
        out = rearrange(out, 'b n (h w) d -> b n h w d', h=h, w=w)

        # 2. Merge heads back into channels: (B, Heads, H, W, D) -> (B, H, W, C)
        out = rearrange(out, 'b n h w d -> b h w (n d)')

        # 3. Output projection
        out = self.to_out(out)

        return out




class DiTBlock(nn.Module):
    def __init__(self, dim, heads=12, dim_head=64, time_dim=1024):
        super().__init__()
        # Normalization
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

        # Time embedding projection (Scale and Shift for each Norm)
        # We need 6 values: scale/shift for Norm1, scale/shift for Norm2, and 2 for gate
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 6)
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

        self.attn = SelfAttention(dim, heads=heads, dim_head=dim_head, rope_dim=dim_head//4)

        hidden_dim = dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x, t_emb):
        """
        x: (B, C, H, W)
        t_emb: (B, time_dim) - already processed via a TimeStep embedder
        """
        # 1. Get Time-based scale/shift parameters
        # Chunk into 6 pieces for adaptive scaling
        shift1, scale1, gate1, shift2, scale2, gate2 = self.time_mlp(t_emb).chunk(6, dim=-1)

        # reshape to channel last
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)

        # 2. Attention Path
        res = x

        # Adaptive Layer Norm 1
        x = self.norm1(x) * (1 + scale1.view(-1, 1, 1, x.shape[-1])) + shift1.view(-1, 1, 1, x.shape[-1])

        # self attention
        x = self.attn(x)

        # add to res
        x = res + gate1.view(-1, 1, 1, x.shape[-1]) * x

        # 3. FFN Path
        res = x

        # Adaptive Layer Norm 2
        x = self.norm2(x) * (1 + scale2.view(-1, 1, 1, x.shape[-1])) + shift2.view(-1, 1, 1, x.shape[-1])

        # linear network
        x = self.ffn(x)

        # add to res
        x = res + gate2.view(-1, 1, 1, x.shape[-1]) * x

        # reshape back to original format
        x = x.permute(0, 3, 1, 2) # (B, C, H, W)

        return x

class ResidualDenseBlock(nn.Module):

    def __init__(self, num_feat=64, num_grow_ch=32, inference=False):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1, padding_mode='zeros')
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1, padding_mode='zeros')
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1, padding_mode='zeros')
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1, padding_mode='zeros')
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1, padding_mode='zeros')

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=inference)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.lrelu(self.conv5(torch.cat((x, x1, x2, x3, x4), 1)))
        return x5 * 0.2 + x

class HighwayRRDB(nn.Module):
    """Highway RRDB Block with feature compression/expansion."""

    def __init__(self, highway_channels, processing_channels=64, num_grow_ch=32, t_hidden_dim = 256, inference = False, memory_efficient_inference_device = None):
        super(HighwayRRDB, self).__init__()

        self.inference = inference
        self.memory_efficient_inference_device = memory_efficient_inference_device

        self.highway_channels = highway_channels
        self.processing_channels = processing_channels

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=inference)
        
        # to modify features based on t conditioning
        self.t_mlp = nn.Sequential(
            nn.Linear(t_hidden_dim, t_hidden_dim),
            self.lrelu,
            nn.Linear(t_hidden_dim, processing_channels * 2 + 1) # scale, shift, global scale
        )
        
        # Initialize the final layer to zero
        # This ensures that at the start of training, the block acts like a normal SR block
        nn.init.zeros_(self.t_mlp[-1].weight)
        nn.init.zeros_(self.t_mlp[-1].bias)

        # Apply parallel dilated convolutions on the compressed feature map
        self.dc1 = nn.Conv2d(processing_channels, num_grow_ch, 3, 1, 1, dilation=1, padding_mode='zeros')
        self.dc2 = nn.Conv2d(processing_channels, num_grow_ch, 3, 1, 2, dilation=2, padding_mode='zeros')
        self.dc3 = nn.Conv2d(processing_channels, num_grow_ch, 3, 1, 3, dilation=3, padding_mode='zeros')
        self.dc4 = nn.Conv2d(processing_channels, num_grow_ch, 3, 1, 4, dilation=4, padding_mode='zeros')
        # Fusion layer to combine the outputs of the dilated convolutions
        self.context_fusion = nn.Conv2d(num_grow_ch * 4, processing_channels, 3, 1, 1, padding_mode='zeros')

        # Compression: Highway → Processing
        self.compress = nn.Conv2d(highway_channels, processing_channels, kernel_size=1)

        # Dense processing blocks
        self.rdb1 = ResidualDenseBlock(processing_channels, num_grow_ch, inference)
        self.rdb2 = ResidualDenseBlock(processing_channels, num_grow_ch, inference)
        self.rdb3 = ResidualDenseBlock(processing_channels, num_grow_ch, inference)

        # Expansion: Processing → Highway
        # Using 3x3 for spatial-aware mixing before expansion
        self.pre_expand = nn.Conv2d(processing_channels, processing_channels, kernel_size=3, padding=1, padding_mode='zeros')
        self.expand = nn.Conv2d(processing_channels, highway_channels, kernel_size=1)

    def forward(self, highway_features_container, t_emb):
        # highway_features_container is a list of 1 highway_features to allow for offloading to cpu and avoid holding a reference in the calling method that prevents cpu offload

        # Compress highway features for processing
        processed = self.lrelu(self.compress(highway_features_container[0]))

        if self.memory_efficient_inference_device is not None:
          highway_features_container[0] = highway_features_container[0].to("cpu")
          torch.cuda.empty_cache()

        combined = self.lrelu(self.dc1(processed))
        combined = torch.cat([combined, self.lrelu(self.dc2(processed))], dim=1)
        combined = torch.cat([combined, self.lrelu(self.dc3(processed))], dim=1)
        combined = torch.cat([combined, self.lrelu(self.dc4(processed))], dim=1)
        combined = self.lrelu(self.context_fusion(combined))
        
        processed = processed + combined
        del combined

        if self.inference:
          torch.cuda.empty_cache()

        
        # t_emb conditioning
        t_params = self.t_mlp(t_emb)
        res_scale = t_params[:,-1:].view(-1,1,1,1) # global scale multiplier based on t

        # 1. Generate local Scale (gamma) and Shift (beta)
        # t_emb is (B, emb_dim) -> t_params is (B, C*2, 1, 1)
        gamma_beta = t_params[:, :-1].view(t_emb.size(0), -1, 1, 1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        # 2. Apply modulation
        # We use (1 + gamma) so that a zero-initialized gamma means "multiply by 1"
        processed = processed * (1 + gamma) + beta

        # Dense processing
        processed = self.rdb1(processed)
        processed = self.rdb2(processed)
        processed = self.rdb3(processed)

        # Spatial-aware pre-expansion
        processed = self.lrelu(self.pre_expand(processed))

        # Expand back to highway size
        expanded = self.expand(processed)

        if self.memory_efficient_inference_device is not None:
            highway_features_container[0] = highway_features_container[0].to(self.memory_efficient_inference_device)

        if not self.inference:
            # Training: keep safe for autograd
            return highway_features_container[0] + expanded * res_scale
        else:
            # Inference: do it in-place to save VRAM, very important for processing large images!
            expanded.mul_(res_scale)
            expanded.add_(highway_features_container[0]) # dont modify highway_features, it might be used elsewhere. but expanded is no longer used so it can modify inplace
            return expanded

@ARCH_REGISTRY.register()
class RRDB_UNet_v8_t(nn.Module):

    def __init__(self, num_in_ch, num_out_ch, highway_channels_base=32, processing_channels_base=16, num_grow_ch_base=8,
                 encoder_blocks=4, decoder_blocks=6, ae_channel_multipliers = [1,2,4,8,16], body_rrdb_blocks=12, body_dit_blocks=4, t_hidden_dim = 1024, memory_efficient_inference_device = None, inference = False, checkpoint = False):

        super(RRDB_UNet_v8_t, self).__init__()

        self.memory_efficient_inference_device = memory_efficient_inference_device
        self.inference = inference or self.memory_efficient_inference_device is not None

        self.checkpoint = checkpoint

        print("highway_channels_base", highway_channels_base)
        print("processing_channels_base", processing_channels_base)
        print("num_grow_ch_base", num_grow_ch_base)
        print("encoder_blocks", encoder_blocks)
        print("decoder_blocks", decoder_blocks)
        print("ae_channel_multipliers", ae_channel_multipliers)
        print("body_rrdb_blocks", body_rrdb_blocks)
        print("body_dit_blocks", body_dit_blocks)

        self.timestep_embed = TimestepEmbedding(hidden_dim = t_hidden_dim)

        lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=inference)

        self.w_h_multiple = 2 ** (len(ae_channel_multipliers)-1)

        self.prep = nn.ModuleList()
        self.prep.append(nn.Conv2d(num_in_ch, highway_channels_base * ae_channel_multipliers[0], 3, 1, 1, padding_mode='reflect'))  # get the image to initial channel num

        # create encoder
        self.encoder = nn.ModuleList()

        for i in range(len(ae_channel_multipliers)-1):
            encoder_block = nn.ModuleList()

            current_multiplier = ae_channel_multipliers[i]
            next_multiplier = ae_channel_multipliers[i+1]
            ## rrdbs
            for _ in range(encoder_blocks):
                encoder_block.append(
                    HighwayRRDB(
                        highway_channels=highway_channels_base * current_multiplier,
                        processing_channels=processing_channels_base * current_multiplier,
                        num_grow_ch=num_grow_ch_base * current_multiplier,
                        t_hidden_dim = t_hidden_dim,
                        inference = self.inference,
                        memory_efficient_inference_device = self.memory_efficient_inference_device
                    )
                )
            ## pixelUnShuffle & channel match for next block
            encoder_block.append(nn.PixelUnshuffle(2))
            encoder_block.append(nn.Conv2d(highway_channels_base * current_multiplier * 4, highway_channels_base * next_multiplier, 3, 1,1, padding_mode='zeros'))  # from 4x pixelUnShuffle channel growth to target channels for the next encoder block
            encoder_block.append(lrelu)

            self.encoder.append(encoder_block)

        ## between encoder / decoder perform some crazy number crunching
        body_channel_multiplier = ae_channel_multipliers[-1]



        body1_blocks = body_rrdb_blocks // 2
        self.body1 = nn.ModuleList()
        for _ in range(body1_blocks):
            self.body1.append(
                HighwayRRDB(
                    highway_channels=highway_channels_base * body_channel_multiplier,
                    processing_channels=processing_channels_base * body_channel_multiplier,
                    num_grow_ch=num_grow_ch_base * body_channel_multiplier,
                    t_hidden_dim = t_hidden_dim,
                    inference = self.inference,
                    memory_efficient_inference_device = self.memory_efficient_inference_device
                )
            )

        self.dit = nn.ModuleList()
        for _ in range(body_dit_blocks):
            self.dit.append(
                DiTBlock(
                    dim = highway_channels_base * body_channel_multiplier, 
                    heads = 12,
                    dim_head = highway_channels_base * body_channel_multiplier // 8,
                    time_dim = t_hidden_dim
                )
            )

        self.body2 = nn.ModuleList()
        for _ in range(body_rrdb_blocks - body1_blocks):
            self.body2.append(
                HighwayRRDB(
                    highway_channels=highway_channels_base * body_channel_multiplier,
                    processing_channels=processing_channels_base * body_channel_multiplier,
                    num_grow_ch=num_grow_ch_base * body_channel_multiplier,
                    t_hidden_dim = t_hidden_dim,
                    inference = self.inference,
                    memory_efficient_inference_device = self.memory_efficient_inference_device
                )
            )

            # create decoder
        self.decoder = nn.ModuleList()
        for i in range(len(ae_channel_multipliers)-1):
            current_multiplier = ae_channel_multipliers[len(ae_channel_multipliers)-2-i]
            last_multiplier = ae_channel_multipliers[len(ae_channel_multipliers)-1-i]

            decoder_block = nn.ModuleList()

            # the encoder residual is is concat to the features and then channels are reduced again
            decoder_block.append(nn.Conv2d(highway_channels_base * last_multiplier * 2, highway_channels_base * last_multiplier, 3,1,1, padding_mode='zeros'))
            decoder_block.append(lrelu)


            ## pixelShuffle input up
            decoder_block.append(nn.PixelShuffle(2))
            decoder_block.append(nn.Conv2d(highway_channels_base * last_multiplier // 4, highway_channels_base * current_multiplier, 3, 1, 1, padding_mode='zeros'))  # from 1/4x pixelShuffle channel growth to target channels for the current decoder block
            decoder_block.append(lrelu)

            ## rrdbs
            for _ in range(decoder_blocks):
                decoder_block.append(
                    HighwayRRDB(
                        highway_channels=highway_channels_base * current_multiplier,
                        processing_channels=processing_channels_base * current_multiplier,
                        num_grow_ch=num_grow_ch_base * current_multiplier,
                        t_hidden_dim = t_hidden_dim,
                        inference = self.inference,
                        memory_efficient_inference_device = self.memory_efficient_inference_device
                    )
                )
            self.decoder.append(decoder_block)



        # --- THE TAIL (Refinement) ---
        base_multiplier = ae_channel_multipliers[0]

        # We need to calculate input channels for the tail.
        # It equals: Feature Channels + Original Image Channels (due to concatenation)
        feat_ch = highway_channels_base * base_multiplier
        cat_ch = feat_ch + num_in_ch

        final_inner_ch = (highway_channels_base) * base_multiplier

        self.tail = nn.Sequential(
            # Note the input channels: cat_ch
            nn.Conv2d(cat_ch, final_inner_ch, 3, 1, 1, padding_mode='zeros'), # reflect causes oom on large images, zeros has to do
            lrelu,
            nn.Conv2d(final_inner_ch, final_inner_ch, 3, 1, 1, padding_mode='zeros'),
            lrelu,
            nn.Conv2d(final_inner_ch, num_out_ch, 1, 1, 0)
        )


    def forward(self, x, t):
        B, C, H, W = x.shape

        # compute target size (next multiple of w_h_multiple)
        target_H = ((H + self.w_h_multiple - 1) // self.w_h_multiple) * self.w_h_multiple
        target_W = ((W + self.w_h_multiple - 1) // self.w_h_multiple) * self.w_h_multiple

        # compute padding on all sides (symmetric)
        pad_top = (target_H - H) // 2
        pad_bottom = target_H - H - pad_top
        pad_left = (target_W - W) // 2
        pad_right = target_W - W - pad_left

        # pad input (reflect padding is usually safest for images)
        feat = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

        res1 = feat
        

        if self.memory_efficient_inference_device != None:
            self.timestep_embed.to(self.memory_efficient_inference_device)
        t_emb = self.timestep_embed(t)
        if self.memory_efficient_inference_device != None:
            self.timestep_embed.to("cpu")
        
        if self.memory_efficient_inference_device != None:
            self.prep.to(self.memory_efficient_inference_device)
        for element in self.prep:
            feat = element(feat)
        if self.memory_efficient_inference_device != None:
            self.prep.to("cpu")
            torch.cuda.empty_cache()

        residuals = []
        # run through encoder and capture residuals
        for encoder_block in self.encoder:
            for element in encoder_block:
                if self.memory_efficient_inference_device != None:
                    element.to(self.memory_efficient_inference_device)
                
                if isinstance(element, HighwayRRDB):
                    feat = [feat] # to allow cpu offload in memory efficient inference, the features are wrapped in a list
                    if self.checkpoint:
                        feat = checkpoint(element, feat, t_emb, use_reentrant=False)
                    else:
                        feat = element(feat, t_emb)
                else:
                    feat = element(feat) # Standard Conv/Shuffle
                
                if self.memory_efficient_inference_device != None:
                    element.to("cpu")
                    torch.cuda.empty_cache()
            
            if self.memory_efficient_inference_device != None:
              residuals.append(feat.to("cpu"))
              torch.cuda.empty_cache()
            else:
              residuals.append(feat)


        for element in self.body1:
            if self.memory_efficient_inference_device != None:
                element.to(self.memory_efficient_inference_device)
                
            feat = [feat]
            if self.checkpoint:
                feat = checkpoint(element, feat, t_emb, use_reentrant=False)
            else:
                feat = element(feat, t_emb)
                    
            if self.memory_efficient_inference_device != None:
                element.to("cpu")
                torch.cuda.empty_cache()

        for element in self.dit:
            if self.memory_efficient_inference_device != None:
                element.to(self.memory_efficient_inference_device)
                
            if self.checkpoint:
                feat = checkpoint(element, feat, t_emb, use_reentrant=False)
            else:
                feat = element(feat, t_emb)
                    
            if self.memory_efficient_inference_device != None:
                element.to("cpu")
                torch.cuda.empty_cache()

        
        for element in self.body2:
            if self.memory_efficient_inference_device != None:
                element.to(self.memory_efficient_inference_device)
                
            feat = [feat]
            if self.checkpoint:
                feat = checkpoint(element, feat, t_emb, use_reentrant=False)
            else:
                feat = element(feat, t_emb)
                    
            if self.memory_efficient_inference_device != None:
                element.to("cpu")
                torch.cuda.empty_cache()


        # run through decoder and insert residuals
        for decoder_block in self.decoder:
            residual = residuals.pop()
            if self.memory_efficient_inference_device != None:
              residual = residual.to(self.memory_efficient_inference_device)
            feat = torch.cat([feat, residual], dim=1)
            
            del residual # this is no longer required, delete to free memory
            if self.inference:
                torch.cuda.empty_cache() # because the residual was deleted, free up the gpu memory

            for element in decoder_block:
                if self.memory_efficient_inference_device != None:
                    element.to(self.memory_efficient_inference_device)
                
                if isinstance(element, HighwayRRDB):
                    feat = [feat]
                    if self.checkpoint:
                        feat = checkpoint(element, feat, t_emb, use_reentrant=False)
                    else:
                        feat = element(feat, t_emb)
                else:
                    feat = element(feat) # Standard Conv/Shuffle
                    
                if self.memory_efficient_inference_device != None:
                    element.to("cpu")
                    torch.cuda.empty_cache()
            

        # Instead of just adding the original image at the very end, we concatenate features + img here so that it can do some final refinement with consideration of the original image
        feat = torch.cat([feat, res1], dim=1)

        # Run the refinement tail
        if self.memory_efficient_inference_device != None:
            self.tail.to(self.memory_efficient_inference_device)
        feat = self.tail(feat)
        if self.memory_efficient_inference_device != None:
            self.tail.to("cpu")
            torch.cuda.empty_cache()

        # crop back to original size
        feat = feat[:, :, pad_top:pad_top + H, pad_left:pad_left + W]

        return feat


    # torchao can convert linear layers into float8, this method allows torchao to know what to convert if we decide to convert
    def fp8_filter_fn(self, module, fqn):
        # convert the linear layers in the transformer to fp8
        # "dit" is the transformer module list
        # this includes to_qkv, to_out, the 2 ffn
        if fqn.startswith("dit") and not "time_mlp" in fqn:
            print("fp8 allowed for", fqn)
            return True
        return False