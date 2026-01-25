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
    def __init__(self, model_path, model, pre_pad=8, device="cuda"):
        self.pre_pad = pre_pad
        self.device = device

        if model_path is not None:
            loadnet = torch.load(model_path, map_location='cpu')
            keyname = 'params_ema' if 'params_ema' in loadnet else 'params'
            model.load_state_dict(loadnet[keyname], strict=True)
            print("loaded", keyname)
        
        model.eval()
        self.model = model

    def pre_process(self, img):
        # Convert to tensor and move to device
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)

        if self.pre_pad != 0:
            img = F.pad(img, (self.pre_pad, self.pre_pad, self.pre_pad, self.pre_pad), 'reflect')
        
        return img

    def process(self, lq, num_steps):
        """
        The Sampling Loop (Euler Method)
        Matches training: t=0 (Noise) -> t=1 (Clean)
        """
        batch_size, _, h, w = lq.shape
        
        # 1. Start with pure Gaussian noise (xt at t=0)
        xt = torch.randn_like(lq)
        
        # 2. Define timesteps (from 0.0 up to 1.0)
        steps = torch.linspace(0.0, 1.0, num_steps + 1).to(self.device)
        dt = 1.0 / num_steps

        eta = 0.2

        guidance_scale = 1
    
        outputs = []
        
        with torch.no_grad():
            for i in tqdm(range(num_steps)):
                t = steps[i].expand(batch_size)
                
                model_input_cond = torch.cat([xt, (lq*2)-1], dim=1)
                model_input_uncond = torch.cat([xt, torch.zeros_like(lq)], dim=1)
                if guidance_scale == 1:
                    p_uncond = torch.zeros_like(lq)

                with torch.amp.autocast(self.device):
                    p_cond = self.model(model_input_cond, t)
                    if guidance_scale != 1:
                        p_uncond = self.model(model_input_uncond, t)
                
                pred = p_uncond + guidance_scale * (p_cond - p_uncond)

                xt = xt + dt * pred

                # 3. Add Stochastic "Shake" (The Noise Back)
                
                # We scale by (1-t) because the model is trained to handle 
                # more drift/noise at the beginning than at the end.
                noise = torch.randn_like(xt)
                xt = xt + noise * eta * math.sqrt(dt) * (1.0 - steps[i+1])
                
                outputs.append((xt.detach().cpu()+1)/2)
    
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
    def enhance(self, imgBGR, steps):
        # 1. Standardize input (0-1 range)
        img = imgBGR.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Run the pipeline
        lq_tensor = self.pre_process(img)
        processed = self.process(lq_tensor, steps) # This now runs the multi-step sampler
        output_imgs = []
        for i in processed:
            output = self.post_process(i)
            # 3. Convert back to BGR 8-bit
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            output = (output * 255.0).round().astype(np.uint8)
            
            output_imgs.append(output)
        
        return output_imgs