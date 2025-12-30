import cv2
import math
import numpy as np
import os
import queue
import threading
import torch
from basicsr.utils.download_util import load_file_from_url
from torch.nn import functional as F


class RealESRGANer1x():

    def __init__(self,
                 model_path,
                 model,
                 pre_pad=8,
                 device="cuda"
                 ):
        self.pre_pad = pre_pad
        self.device = device

        if model_path is not None:
            loadnet = torch.load(model_path)
            # prefer to use params_ema
            if 'params_ema' in loadnet:
                keyname = 'params_ema'
            else:
                keyname = 'params'
            model.load_state_dict(loadnet[keyname], strict=True)
        else:
            print("test mode - no model weights loaded!")
        model.eval()
        #model.to(self.device) # todo: MANUAL to cuda in forward to support large images layer by layer
        self.model = model

    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (self.pre_pad, self.pre_pad, self.pre_pad, self.pre_pad), 'reflect').float()

    
    def process(self):
            with torch.no_grad():
                with torch.amp.autocast(self.device):
                    self.output = self.model(self.img)

    def post_process(self):
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, self.pre_pad:h - self.pre_pad, self.pre_pad:w - self.pre_pad]


        output_img = self.output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img, (1, 2, 0))  # (C,H,W) → (H,W,C)
        
        return output_img

    @torch.no_grad()
    def enhance(self, imgBGR_BGRA, alpha_upsampler='realesrgan', outscale=None):
        if outscale is not None:
            print("outscale is ignored for this program")
        img = imgBGR_BGRA
        h_input, w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range
        
        if len(img.shape) == 2 or img.shape[2] == 1:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ------------------- process image (without the alpha channel) ------------------- #
        self.pre_process(img)
        self.process()
        output_img = self.post_process()
        
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR) # swap channels

        
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                self.process()
                output_alpha = self.post_process()
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_RGB2GRAY)
            
            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #
        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        return output, img_mode
