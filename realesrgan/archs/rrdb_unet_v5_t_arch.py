import torch
from torch import nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights

import math

class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        # t: (B) scalar between 0 and 1. We scale to 1000 for standard embedding logic
        t = t * 1000
        half_dim = 128 // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # Shape: (B, 128)
        
        # If your base embedding_dim is not 128, you can adjust above or 
        # just use a Linear layer to project to the desired dim
        return self.mlp(emb)
        
class ChannelAttention(nn.Module):
    def __init__(self, num_features, hidden_features, t_emb_dim=256):
        super(ChannelAttention, self).__init__()
        
        # We pool features to (B, C, 1, 1), then concat t_emb (B, t_emb_dim)
        # So the input to the first conv will be C + t_emb_dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(num_features + t_emb_dim, hidden_features, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(hidden_features, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, t_emb):
        b, c, h, w = x.shape
        
        # 1. Squeeze: Global Average Pooling
        pooled = self.avg_pool(x) # Shape: (B, C, 1, 1)
        
        # 2. Concat Time: Reshape t_emb to (B, t_emb_dim, 1, 1) and concat
        t_reshaped = t_emb.view(b, -1, 1, 1)
        combined = torch.cat([pooled, t_reshaped], dim=1) # Shape: (B, C + t_emb_dim, 1, 1)
        
        # 3. Excitation: Run through MLP to get channel weights
        attention_weights = self.mlp(combined) # Shape: (B, C, 1, 1)
        
        # 4. Scale original features
        return x * attention_weights


class ResidualDenseBlock(nn.Module):

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1, padding_mode='zeros')
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1, padding_mode='zeros')
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1, padding_mode='zeros')
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1, padding_mode='zeros')
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1, padding_mode='zeros')

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

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

    def __init__(self, highway_channels, processing_channels=64, num_grow_ch=32, use_attention=True, inference = False):
        super(HighwayRRDB, self).__init__()

        self.inference = inference

        self.highway_channels = highway_channels
        self.processing_channels = processing_channels
        self.use_attention = use_attention

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # to modify features based on t conditioning
        self.t_mlp = nn.Sequential(
            nn.Linear(256, processing_channels),
            self.lrelu,
            nn.Linear(processing_channels, processing_channels * 2) 
        )
        
        # Initialize the final layer to zero
        # This ensures that at the start of training, the block acts like a normal SR block
        nn.init.zeros_(self.t_mlp[-1].weight)
        nn.init.zeros_(self.t_mlp[-1].bias)

        # Apply parallel dilated convolutions on the compressed feature map
        self.dc1 = nn.Conv2d(processing_channels, num_grow_ch, 3, 1, 1, dilation=1, padding_mode='reflect')
        self.dc2 = nn.Conv2d(processing_channels, num_grow_ch, 3, 1, 2, dilation=2, padding_mode='reflect')
        self.dc3 = nn.Conv2d(processing_channels, num_grow_ch, 3, 1, 3, dilation=3, padding_mode='reflect')
        self.dc4 = nn.Conv2d(processing_channels, num_grow_ch, 3, 1, 4, dilation=4, padding_mode='reflect')
        # Fusion layer to combine the outputs of the dilated convolutions
        self.context_fusion = nn.Conv2d(num_grow_ch * 4, processing_channels, 3, 1, 1, padding_mode='zeros')

        # Compression: Highway → Processing
        self.compress = nn.Conv2d(highway_channels, processing_channels, kernel_size=1)

        # Dense processing blocks
        self.rdb1 = ResidualDenseBlock(processing_channels, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(processing_channels, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(processing_channels, num_grow_ch)

        # Expansion: Processing → Highway
        # Using 3x3 for spatial-aware mixing before expansion
        self.pre_expand = nn.Conv2d(processing_channels, processing_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.expand = nn.Conv2d(processing_channels, highway_channels, kernel_size=1)

        # Channel attention on highway features
        if self.use_attention:
            self.channel_attention = ChannelAttention(highway_channels, highway_channels // 2)

    def forward(self, highway_features, t_emb):
        # Compress highway features for processing
        processed = self.lrelu(self.compress(highway_features))

        processed = processed + self.lrelu(self.context_fusion(torch.cat(
            [
                self.lrelu(self.dc1(processed)),
                self.lrelu(self.dc2(processed)),
                self.lrelu(self.dc3(processed)),
                self.lrelu(self.dc4(processed))
            ],
            dim=1))
        )
        
        # t_emb conditioning
        # 1. Generate local Scale (gamma) and Shift (beta)
        # t_emb is (B, 256) -> t_params is (B, C*2, 1, 1)
        t_params = self.t_mlp(t_emb).view(t_emb.size(0), -1, 1, 1)
        gamma, beta = torch.chunk(t_params, 2, dim=1)

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

        # Apply channel attention
        if self.use_attention:
            expanded = self.channel_attention(expanded, t_emb)

        if not self.inference:
            # Training: keep safe for autograd
            return highway_features + expanded * 0.2
        else:
            # Inference: do it in-place to save VRAM, very important for processing large images!
            expanded.mul_(0.2)
            expanded.add_(highway_features) # dont modify highway_features, it might be used elsewhere. but expanded is no longer used so it can modify inplace
            return expanded


@ARCH_REGISTRY.register()
class RRDB_UNet_v5_t(nn.Module):

    def __init__(self, num_in_ch, num_out_ch, highway_channels_base=32, processing_channels_base=16, num_grow_ch_base=8,
                 ae_rrdb_blocks=4, ae_channel_multipliers = [1,2,4,8,16],use_attention=True, body_rrdb_blocks=12, res1_add = True, memory_efficient_inference_device = None, inference = False):

        super(RRDB_UNet_v5_t, self).__init__()

        self.memory_efficient_inference_device = memory_efficient_inference_device
        self.inference = inference or self.memory_efficient_inference_device is not None

        self.res1_add = res1_add

        self.timestep_embed = TimestepEmbedding()

        print("highway_channels_base", highway_channels_base)
        print("processing_channels_base", processing_channels_base)
        print("num_grow_ch_base", num_grow_ch_base)
        print("ae_rrdb_blocks", ae_rrdb_blocks)
        print("ae_channel_multipliers", ae_channel_multipliers)
        print("use_attention", use_attention)
        print("body_rrdb_blocks", body_rrdb_blocks)
        print("using res1 addition", self.res1_add)

        self.w_h_multiple = 2 ** (len(ae_channel_multipliers)-1)

        self.prep = nn.ModuleList()
        self.prep.append(nn.Conv2d(num_in_ch, highway_channels_base * ae_channel_multipliers[0], 3, 1, 1, padding_mode='reflect'))  # get the image to initial channel num
        self.prep.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

        # create encoder
        self.encoder = nn.ModuleList()

        for i in range(len(ae_channel_multipliers)-1):
            encoder_block = nn.ModuleList()

            current_multiplier = ae_channel_multipliers[i]
            next_multiplier = ae_channel_multipliers[i+1]
            ## rrdbs
            for _ in range(ae_rrdb_blocks):
                encoder_block.append(
                    HighwayRRDB(
                        highway_channels=highway_channels_base * current_multiplier,
                        processing_channels=processing_channels_base * current_multiplier,
                        num_grow_ch=num_grow_ch_base * current_multiplier,
                        use_attention=use_attention,
                        inference = self.inference
                    )
                )
            ## pixelUnShuffle & channel match for next block
            encoder_block.append(nn.PixelUnshuffle(2))
            encoder_block.append(nn.Conv2d(highway_channels_base * current_multiplier * 4, highway_channels_base * next_multiplier, 3, 1,1, padding_mode='zeros'))  # from 4x pixelUnShuffle channel growth to target channels for the next encoder block
            encoder_block.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

            self.encoder.append(encoder_block)

        ## between encoder / decoder perform some crazy number crunching
        body_channel_multiplier = ae_channel_multipliers[-1]
        self.body = nn.ModuleList()
        for _ in range(body_rrdb_blocks):
            self.body.append(
                HighwayRRDB(
                    highway_channels=highway_channels_base * body_channel_multiplier,
                    processing_channels=processing_channels_base * body_channel_multiplier,
                    num_grow_ch=num_grow_ch_base * body_channel_multiplier,
                    use_attention=use_attention,
                    inference = self.inference
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
            decoder_block.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))


            ## pixelShuffle input up
            decoder_block.append(nn.PixelShuffle(2))
            decoder_block.append(nn.Conv2d(highway_channels_base * last_multiplier // 4, highway_channels_base * current_multiplier, 3, 1, 1, padding_mode='zeros'))  # from 1/4x pixelShuffle channel growth to target channels for the current decoder block
            decoder_block.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

            ## rrdbs
            for _ in range(ae_rrdb_blocks):
                decoder_block.append(
                    HighwayRRDB(
                        highway_channels=highway_channels_base * current_multiplier,
                        processing_channels=processing_channels_base * current_multiplier,
                        num_grow_ch=num_grow_ch_base * current_multiplier,
                        use_attention=use_attention,
                        inference = self.inference
                    )
                )
            self.decoder.append(decoder_block)



        # --- THE TAIL (Refinement) ---
        base_multiplier = ae_channel_multipliers[0]

        # We need to calculate input channels for the tail.
        # It equals: Feature Channels + Original Image Channels (due to concatenation)
        feat_ch = highway_channels_base * base_multiplier
        cat_ch = feat_ch + num_in_ch

        final_inner_ch = (processing_channels_base + highway_channels_base) * base_multiplier

        self.tail = nn.Sequential(
            # Note the input channels: cat_ch
            nn.Conv2d(cat_ch, final_inner_ch, 3, 1, 1, padding_mode='zeros'), # reflect causes oom on large images, zeros has to do
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(final_inner_ch, final_inner_ch, 3, 1, 1, padding_mode='zeros'),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
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
                    feat = element(feat, t_emb) # Pass t_emb
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


        for element in self.body:
            if self.memory_efficient_inference_device != None:
                element.to(self.memory_efficient_inference_device)
                
            if isinstance(element, HighwayRRDB):
                feat = element(feat, t_emb) # Pass t_emb
            else:
                feat = element(feat) # Standard Conv/Shuffle
                    
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
                    feat = element(feat, t_emb) # Pass t_emb
                else:
                    feat = element(feat) # Standard Conv/Shuffle
                    
                if self.memory_efficient_inference_device != None:
                    element.to("cpu")
                    torch.cuda.empty_cache()
            

        # Instead of just adding the original image at the very end, we concatenate features + img here so that it can do some final refinement with consideration of the original image
        feat = torch.cat([feat, res1], dim=1)

        if self.memory_efficient_inference_device != None:
            self.tail.to(self.memory_efficient_inference_device)
        # Run the refinement tail
        feat = self.tail(feat)
        if self.memory_efficient_inference_device != None:
            self.tail.to("cpu")
            torch.cuda.empty_cache()

        if self.res1_add:
            feat = feat + res1 # and also add the original image at the end back so it only needs to learn the difference

        # crop back to original size
        feat = feat[:, :, pad_top:pad_top + H, pad_left:pad_left + W]

        return feat
