import numpy as np
import random
import torch
import torchvision
from collections import OrderedDict
from torch.nn import functional as F
from torch.amp import autocast, GradScaler

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.losses.loss_util import get_refined_artifact_map
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY

import torch.distributed as dist

# idk claude says it is faster
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


@MODEL_REGISTRY.register()
class SRFLOW(SRModel):

    def __init__(self, opt):
        super(SRFLOW, self).__init__(opt)

        if(opt['train'].get("use_compile", True)):
            print("using model compile")
            self.net_g.compile(dynamic=False) 

        self.jpeger = DiffJPEG(differentiable=False).to(self.device)  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().to(self.device)  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

        self.min_scale = opt['train'].get('min_scale', 1.0)
        self.max_scale = opt['train'].get('max_scale', 4.0)
        print("scales:", self.min_scale, self.max_scale)

        self.gradient_accumulation_steps = opt['train'].get('gradient_accumulation_steps', 1)
        self._accum_steps = 0  # internal counter for gradient accumulation
        print("gradient_accumulation_steps:",  self.gradient_accumulation_steps)

        self.check_ddp_consistency()

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader and add degradations with dynamic scaling."""
        if self.is_train and self.opt.get('high_order_degradation', True):
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # Random scale factor between min_scale and max_scale
            scale_factor = random.uniform(self.min_scale, self.max_scale)
            target_h = int(ori_h / scale_factor)
            target_w = int(ori_w / scale_factor)

            # ----------------------- The first degradation process ----------------------- #
            out = filter2D(self.gt_usm, self.kernel1)

            # random resize (but more constrained)
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                resize_scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                resize_scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                resize_scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=resize_scale, mode=mode)

            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)

            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                resize_scale2 = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                resize_scale2 = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                resize_scale2 = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(int(target_h * resize_scale2), int(target_w * resize_scale2)), mode=mode)

            # add noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # Final processing with JPEG and sinc filter
            if np.random.uniform() < 0.5:
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(target_h, target_w), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(target_h, target_w), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # Create final LQ image
            self.lq_orig = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # Interpolate LQ back to GT size for diffusion training
            self.lq = F.interpolate(self.lq_orig, size=(ori_h, ori_w), mode='bicubic', align_corners=False)

            # Random crop both GT and LQ
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size, 1)

            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # For validation - interpolate LQ to GT size
            self.lq_orig = data['lq'].to(self.device)
            self.gt = data['gt'].to(self.device)
            # Interpolate LQ to GT size
            self.lq = F.interpolate(self.lq_orig, size=self.gt.shape[-2:], mode='bicubic', align_corners=False)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(SRFLOW, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    @torch.no_grad()
    def check_ddp_consistency(self):
        """
        Verifies that model weights are identical across all GPUs.
        """
        if not dist.is_initialized():
            return

        # Check Generator weights
        for name, param in self.net_g.named_parameters():
            if param.grad is not None:
                # Gather parameters from all GPUs
                params_list = [torch.zeros_like(param) for _ in range(dist.get_world_size())]
                dist.all_gather(params_list, param)
                
                # Compare all against rank 0
                base_param = params_list[0]
                for i, other_param in enumerate(params_list[1:]):
                    if not torch.allclose(base_param, other_param, atol=1e-6):
                        print(f"CRITICAL WARNING: Rank {i+1} G param '{name}' mismatch with Rank 0!")
                        raise RuntimeError("DDP Desynchronization detected!")
                #break # Checking one parameter is usually enough to detect divergence
                
    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()
        
        # 1. Setup Rectified Flow variables
        b, c, h, w = self.gt.shape
        device = self.gt.device
        
        # Sample time t uniformly [0, 1]
        # We reshape to (b, 1, 1, 1) for correct broadcasting during mixing
        t = torch.rand((b, 1, 1, 1), device=device)
        
        # Sample Gaussian noise (X0)
        noise = torch.randn_like(self.gt)
        
        # 2. Linear Interpolation (The Flow)
        # X_t = t * GT + (1 - t) * Noise
        # At t=0, xt is pure noise. At t=1, xt is the clean GT.
        xt = t * self.gt + (1.0 - t) * noise
        
        # 3. Prepare Model Input
        # Concat the noisy image (xt) and the LR condition (lq) 
        # This results in a tensor with (C_xt + C_lq) channels (usually 3+3=6)
        model_input = torch.cat([xt, self.lq], dim=1)
        
        # 4. Forward Pass
        with autocast('cuda', dtype=torch.bfloat16):
            # The model predicts the velocity 'v'
            # We pass the concatenated input and the time t (flattened for the embedding layer)
            v_pred = self.net_g(model_input, t.view(b))
            
            # 5. Calculate Target and Loss
            # Target velocity for Rectified Flow is: GT - Noise
            target = self.gt - noise
            
            # Standard Flow Matching loss is MSE on velocity
            l_total = self.cri_pix(v_pred, target)
    
        # 6. Backward and Optimize
        scaled_loss = l_total / float(self.gradient_accumulation_steps)
        scaled_loss.backward()
        
        self._accum_steps += 1
        if self._accum_steps >= self.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=1.0)
            self.optimizer_g.step()
            self.optimizer_g.zero_grad()
            self._accum_steps = 0
    
            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)
    
        # Log the flow loss
        loss_dict['l_flow'] = l_total.detach()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if current_iter % 1000 == 0:
            self.check_ddp_consistency()



    def test(self):
        lq = self.lq
        sampling_steps = 5  # Renamed to avoid conflict with 'steps' tensor

        batch_size, _, h, w = lq.shape
        device = lq.device
        
        # 1. Start with pure Gaussian noise (X0)
        # In your training: t=0 is noise, t=1 is clean (GT)
        xt = torch.randn_like(lq)
        
        # 2. Define timesteps (from 0.0 up to 1.0)
        # We start at 0 (noise) and add velocity to reach 1 (clean)
        time_steps = torch.linspace(0.0, 1.0, sampling_steps + 1).to(device)
        dt = 1.0 / sampling_steps 
        
        # Select model (EMA preferred)
        model = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        model.eval()

        with torch.no_grad():
            # We loop through sampling_steps (e.g., 0 to 4 if steps=5)
            for i in range(sampling_steps):
                t = time_steps[i].expand(batch_size)
                
                # 3. Prepare 6-channel input
                model_input = torch.cat([xt, lq], dim=1)
                
                # 4. Predict velocity (v)
                # In Flow Matching, v = GT - Noise
                with autocast('cuda', dtype=torch.bfloat16):
                    v_pred = model(model_input, t)
                
                # 5. Euler Step: move xt TOWARD the clean image
                # Since we go from t=0 to t=1, we ADD the velocity
                # x_{t+dt} = x_t + (dt * v_pred)
                xt = xt + dt * v_pred
        
        # Reset model to training mode if necessary
        if self.is_train:
            self.net_g.train()

        # Clamp output to valid range
        self.output = torch.clamp(xt, 0.0, 1.0)