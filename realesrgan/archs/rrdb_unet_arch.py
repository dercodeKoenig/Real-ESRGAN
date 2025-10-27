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

        if torch.is_grad_enabled():
            # Training: keep safe for autograd
            return highway_features + expanded * 0.2
        else:
            # Inference: do it in-place to save VRAM, very important for processing large images!
            expanded.mul_(0.2)
            highway_features.add_(expanded)
            return highway_features


@ARCH_REGISTRY.register()
class RRDB_UNet(nn.Module):

    def __init__(self, num_in_ch, num_out_ch, highway_channels=32, processing_channels=16, num_blocks=4,
                 num_downscales=4, num_grow_ch=8, use_attention=True, downscale_channel_growth=2, body_rrdb_blocks=12):
        super(RRDB_UNet, self).__init__()

        self.w_h_multiple = 2 ** num_downscales

        self.highway_channels = highway_channels

        # create encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Conv2d(num_in_ch, highway_channels, 3, 1, 1))  # get the rdb to initial channel num
        self.encoder.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        for d in range(num_downscales):
            new_c_multiplier = (downscale_channel_growth ** d)  # increase channels while downscale

            ## rrdbs
            for _ in range(num_blocks):
                self.encoder.append(
                    HighwayRRDB(
                        highway_channels=highway_channels * new_c_multiplier,
                        processing_channels=processing_channels * new_c_multiplier,
                        num_grow_ch=num_grow_ch * new_c_multiplier,
                        use_attention=use_attention
                    )
                )
                ## pixelUnShuffle & channel match for next block
            self.encoder.append(nn.PixelUnshuffle(2))
            self.encoder.append(nn.Conv2d(highway_channels * new_c_multiplier * 4,
                                          highway_channels * new_c_multiplier * downscale_channel_growth, 3, 1,
                                          1))  # from 4x pixelUnShuffle channel growth to downscale_channel_growth for the next encoder block
            self.encoder.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

        ## between encoder / decoder perform some crazy number crunching
        body_channel_multiplier = (downscale_channel_growth ** num_downscales)
        self.body = nn.ModuleList()
        for _ in range(body_rrdb_blocks):
            self.body.append(
                HighwayRRDB(
                    highway_channels=highway_channels * body_channel_multiplier,
                    processing_channels=processing_channels * body_channel_multiplier,
                    num_grow_ch=num_grow_ch * body_channel_multiplier,
                    use_attention=use_attention
                )
            )

            # create decoder
        self.decoder = nn.ModuleList()
        for d in range(num_downscales):
            d = num_downscales - d - 1  # inverse now

            new_c_multiplier = (downscale_channel_growth ** d)  # decrease channels while upscale

            ## pixelShuffle input up
            input_channels_multiplier = (downscale_channel_growth ** (d + 1))
            self.decoder.append(nn.PixelShuffle(2))
            self.decoder.append(
                nn.Conv2d(highway_channels * input_channels_multiplier // 4, highway_channels * new_c_multiplier, 3, 1,
                          1))  # from 1/4x pixelShuffle channel growth to 1/downscale_channel_growth for the next decoder block
            self.decoder.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

            ## rrdbs
            for _ in range(num_blocks):
                self.decoder.append(
                    HighwayRRDB(
                        highway_channels=highway_channels * new_c_multiplier,
                        processing_channels=processing_channels * new_c_multiplier,
                        num_grow_ch=num_grow_ch * new_c_multiplier,
                        use_attention=use_attention
                    )
                )

                # final cnns
        self.decoder.append(nn.Conv2d(highway_channels, highway_channels, 3, 1, 1))
        self.decoder.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

        self.decoder.append(nn.Conv2d(highway_channels, highway_channels, 3, 1, 1))
        self.decoder.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

        self.decoder.append(nn.Conv2d(highway_channels, num_out_ch, 1, 1, 0))

    def forward(self, x):
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

        # run through encoder, body, decoder
        for element in self.encoder:
            feat = element(feat)

        for element in self.body:
            feat = element(feat)

        for element in self.decoder:
            feat = element(feat)

        # crop back to original size
        feat = feat[:, :, pad_top:pad_top + H, pad_left:pad_left + W]

        return feat
