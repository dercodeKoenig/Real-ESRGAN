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


class RealSRFIELD():
    def __init__(self, model_path, model, pre_pad=8, device="cuda"):
        self.pre_pad = pre_pad
        self.device = device

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

    def process(self, lq, num_steps, step_size):
        
        batch_size, _, h, w = lq.shape
        
        xt = torch.randn_like(lq)
        
        eta = 0.2

        guidance_scale = 2
    
        outputs = []
        
        with torch.no_grad():
            for i in tqdm(range(num_steps)):

                model_input_cond = torch.cat([xt, (lq*2)-1], dim=1)
                model_input_uncond = torch.cat([xt, torch.zeros_like(lq)], dim=1)
                if guidance_scale == 1:
                    p_uncond = torch.zeros_like(lq)

                with torch.amp.autocast(self.device):
                    p_cond = self.model(model_input_cond)
                    if guidance_scale != 1:
                        p_uncond = self.model(model_input_uncond)
                
                # CFG: Push the prediction AWAY from the "blurry/unconditioned" version
                # guidance_scale > 1.0 (e.g., 1.5 or 3.0)
                pred = p_uncond + guidance_scale * (p_cond - p_uncond)
                
                xt = xt + pred * step_size

                noise = torch.randn_like(xt)
                noise_scale = eta * step_size * (1.0 - ((i+1) / num_steps))
                xt = xt * (1-noise_scale) + noise * noise_scale
                
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
    def enhance(self, imgBGR, steps, step_size):
        # 1. Standardize input (0-1 range)
        img = imgBGR.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Run the pipeline
        lq_tensor = self.pre_process(img)
        processed = self.process(lq_tensor, steps, step_size) # This now runs the multi-step sampler
        output_imgs = []
        for i in processed:
            output = self.post_process(i)
            # 3. Convert back to BGR 8-bit
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            output = (output * 255.0).round().astype(np.uint8)
            
            output_imgs.append(output)
        
        return output_imgs