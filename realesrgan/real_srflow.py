import cv2
import math
import numpy as np
import os
import queue
import threading
import torch
from tqdm import tqdm
from basicsr.utils.download_util import load_file_from_url
from torch.nn import functional as F


class RealSRFLOW():
    def __init__(self, model_path, model, pre_pad=8, device="cuda", num_steps=20):
        self.pre_pad = pre_pad
        self.device = device
        self.num_steps = num_steps # Number of refinement steps

        if model_path is not None:
            loadnet = torch.load(model_path, map_location='cpu')
            keyname = 'params_ema' if 'params_ema' in loadnet else 'params'
            model.load_state_dict(loadnet[keyname], strict=True)
        
        model.eval()
        self.model = model

    def pre_process(self, img):
        # Convert to tensor and move to device
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)

        if self.pre_pad != 0:
            img = F.pad(img, (self.pre_pad, self.pre_pad, self.pre_pad, self.pre_pad), 'reflect')
        
        return img

    def process(self, lq):
        """
        The Sampling Loop (Euler Method)
        Matches training: t=0 (Noise) -> t=1 (Clean)
        """
        batch_size, _, h, w = lq.shape

        lq.to(self.device)
        
        # 1. Start with pure Gaussian noise (xt at t=0)
        xt = torch.randn_like(lq)
        
        # 2. Define timesteps (from 0.0 up to 1.0)
        steps = torch.linspace(0.0, 1.0, self.num_steps + 1).to(self.device)
        dt = 1.0 / self.num_steps 
    
        outputs = []
        
        with torch.no_grad():
            for i in range(self.num_steps):
                t = steps[i].expand(batch_size)
                
                # 3. Prepare 6-channel input
                model_input = torch.cat([xt, lq], dim=1)
                
                # 4. Predict velocity (v)
                with torch.amp.autocast(self.device):
                    v_pred = self.model(model_input, t)
                
                # 5. Euler Step: move xt forward toward t=1
                # x_{t+dt} = x_t + (dt * v_pred)
                xt = xt + dt * v_pred
                
                outputs.append(xt.detach().cpu())
    
        return outputs

        
    def post_process(self, img):
        # Remove padding
        if self.pre_pad != 0:
            _, _, h, w = img.size()
            output_img = img[:, :, self.pre_pad:h - self.pre_pad, self.pre_pad:w - self.pre_pad]

        # Convert back to numpy (H, W, C)
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        return np.transpose(output_img, (1, 2, 0))

    @torch.no_grad()
    def enhance(self, imgBGR):
        # 1. Standardize input (0-1 range)
        img = imgBGR.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Run the pipeline
        lq_tensor = self.pre_process(img)
        processed = self.process(lq_tensor) # This now runs the multi-step sampler
        output_imgs = []
        for i in processed:
            output = self.post_process(i)
            # 3. Convert back to BGR 8-bit
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            output = (output * 255.0).round().astype(np.uint8)
            
            output_imgs.append(output)
        
        return output_imgs