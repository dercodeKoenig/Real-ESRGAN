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
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY

import torch.distributed as dist

# idk claude says it is faster
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


@MODEL_REGISTRY.register()
class RealESRGANModel1R(SRGANModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealESRGANModel1R, self).__init__(opt)

        if(opt['train'].get("use_compile", True)):
            print("using model compile")
            #self.net_g.compile(mode="max-autotune", dynamic=False, fullgraph=True)
            #self.net_d.compile(mode="max-autotune", dynamic=False, fullgraph =True)
            self.net_g.compile(dynamic=False) # unfortunately, the noise injection layer is incompatible with max-autotune
            self.net_d.compile(dynamic=False)

        self.jpeger = DiffJPEG(differentiable=False).to(self.device)  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().to(self.device)  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

        self.min_scale = opt['train'].get('min_scale', 1.0)
        self.max_scale = opt['train'].get('max_scale', 4.0)
        print("scales:", self.min_scale, self.max_scale)

        self.d_guessing_threshold = opt['train'].get('d_guessing_threshold', 999999)
        print ("d_guessing_threshold:", self.d_guessing_threshold)
        
        self.d_loss_threshold = opt['train'].get('d_loss_threshold',
                                                 0.6)  # When D loss < this, slow down D updates. When D loss < this / 2, skip the update
        self.d_slow_iters = opt['train'].get('d_slow_iters', 11)  # Update D every x steps when loss is low
        self.d_normal_iters = opt['train'].get('d_normal_iters', 1)  # Normal D update frequency

        self.gan_warmup_iters = opt['train'].get('gan_warmup_iters', 0)  # warmup steps before gan training
        self.percept_warmup_iters = opt['train'].get('percept_warmup_iters', 0)  # warmup steps before adding vgg loss

        self.enable_gan = opt['train'].get('enable_gan', True)
        print("gan enabled", self.enable_gan)

        self.gradient_accumulation_steps = opt['train'].get('gradient_accumulation_steps', 1)
        self._accum_steps = 0  # internal counter for gradient accumulation
        print("gradient_accumulation_steps:",  self.gradient_accumulation_steps)

        self.blur_pixel_loss = opt['train'].get('blur_pixel_loss', False)
        if self.blur_pixel_loss:
            self.pixel_blur_transform = torchvision.transforms.GaussianBlur(
                kernel_size=9, 
                sigma=1.5
            ).to(self.device)
        print("blur_pixel_loss:",  self.blur_pixel_loss)

        # Track discriminator update counter and cached values
        self.cached_d_loss_value = -1.0

        # Cache tensor values instead of floats - initialize as tensors
        self.cached_d_real = torch.tensor(0.0, device=self.device)
        self.cached_d_fake = torch.tensor(0.0, device=self.device)
        self.cached_out_d_real = torch.tensor(0.0, device=self.device)
        self.cached_out_d_fake = torch.tensor(0.0, device=self.device)

        print(f"D loss threshold: {self.d_loss_threshold}")
        print(f"D slow update interval: {self.d_slow_iters}")
        print(f"D normal update interval: {self.d_normal_iters}")
        print("")
        print("gan_warmup_iters:", self.gan_warmup_iters)
        print("percept_warmup_iters:", self.percept_warmup_iters)

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
        super(RealESRGANModel1R, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def _should_update_discriminator(self, current_iter):
        """Determine if discriminator should be updated based on adaptive strategy using cached loss."""
        if not self.enable_gan:
            return False

        # Adaptive strategy based on cached discriminator loss from previous iteration
        if self.cached_d_loss_value < self.d_loss_threshold and self.cached_d_loss_value > 0:
            # D is too strong, update less frequently
            return current_iter % self.d_slow_iters == 0
        else:
            # D loss is reasonable, update normally
            return current_iter % self.d_normal_iters == 0
    
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

        # Check Discriminator weights (Crucial for your adaptive logic)
        for name, param in self.net_d.named_parameters():
            if param.requires_grad:
                params_list = [torch.zeros_like(param) for _ in range(dist.get_world_size())]
                dist.all_gather(params_list, param)
                
                base_param = params_list[0]
                for i, other_param in enumerate(params_list[1:]):
                    if not torch.allclose(base_param, other_param, atol=1e-6):
                        print(f"CRITICAL WARNING: Rank {i+1} D param '{name}' mismatch with Rank 0!")
                        raise RuntimeError("DDP Desynchronization detected!")
                #break
                
    def optimize_parameters(self, current_iter):
        num_refine = self.opt.get('num_refine_steps', 1)  # e.g. set in YAML or default

        for refine_step in range(num_refine):
            # --- choose GTs depending on usm options ---
            l1_gt = self.gt_usm if self.opt['l1_gt_usm'] else self.gt
            percep_gt = self.gt_usm if self.opt['percep_gt_usm'] else self.gt
            gan_gt = self.gt_usm if self.opt['gan_gt_usm'] else self.gt

            with autocast('cuda', dtype=torch.bfloat16):
                self.output = self.net_g(self.lq)
                if self.cri_ldl:
                    self.output_ema = self.net_g_ema(self.lq)

                        

            should_update_d = self._should_update_discriminator(current_iter)
            if should_update_d:

                output_for_d = self.output.detach().clone()
                
                for p in self.net_d.parameters():
                    p.requires_grad = True
                
                self.optimizer_d.zero_grad()
                with autocast('cuda', dtype=torch.bfloat16):
                    real_d_pred = self.net_d(gan_gt)
                    fake_d_pred = self.net_d(output_for_d)
                    
                l_d_real = self.cri_gan(real_d_pred.float(), True, is_disc=True) 
                l_d_fake = self.cri_gan(fake_d_pred.float(), False, is_disc=True)
                d_total_loss_tensor = (l_d_real + l_d_fake)
                
                if dist.is_initialized():
                    loss_to_sync = d_total_loss_tensor.detach().clone()
                    dist.all_reduce(loss_to_sync, op=dist.ReduceOp.AVG)
                    d_total_loss = loss_to_sync.item()
                else:
                    d_total_loss = d_total_loss_tensor.item()

                if d_total_loss >= self.d_loss_threshold * 0.5:
                    d_total_loss_tensor.backward()
                    torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), max_norm=1.0)
                    self.optimizer_d.step()
                
                if self.cached_d_loss_value < 0:
                    self.cached_d_loss_value = d_total_loss
                else: # smooth update to make the decision of gan weight and if it should update d more stable and not just depend on 1 batch that has noise
                    self.cached_d_loss_value = self.cached_d_loss_value * 0.99 + d_total_loss * 0.01
                self.cached_d_real = l_d_real.detach()
                self.cached_d_fake = l_d_fake.detach()
                self.cached_out_d_real = torch.mean(real_d_pred.detach())
                self.cached_out_d_fake = torch.mean(fake_d_pred.detach())
            
            
            for p in self.net_d.parameters():
                p.requires_grad = False


            with autocast('cuda', dtype=torch.bfloat16):
                
                l_g_total = 0
                loss_dict = OrderedDict()

                # pixel / ldl / perceptual / gan losses as in your code
                if self.cri_pix:
                    if(self.blur_pixel_loss):
                        output_blur = self.pixel_blur_transform(self.output)
                        l1_gt_blur = self.pixel_blur_transform(l1_gt)
                        l_g_pix = self.cri_pix(output_blur, l1_gt_blur)
                    else:    
                        l_g_pix = self.cri_pix(self.output, l1_gt)
                    l_g_total += l_g_pix
                    loss_dict['l_g_pix'] = l_g_pix

                if self.cri_ldl:
                    pixel_weight = get_refined_artifact_map(self.gt, self.output, self.output_ema, 7)
                    l_g_ldl = self.cri_ldl(
                        torch.mul(pixel_weight, self.output),
                        torch.mul(pixel_weight, self.gt)
                    )
                    l_g_total += l_g_ldl
                    loss_dict['l_g_ldl'] = l_g_ldl

                if self.cri_perceptual and current_iter > self.percept_warmup_iters:
                    l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                    if l_g_percep is not None:
                        l_g_total += l_g_percep
                        loss_dict['l_g_percep'] = l_g_percep
                    if l_g_style is not None:
                        l_g_total += l_g_style
                        loss_dict['l_g_style'] = l_g_style

                if current_iter > self.gan_warmup_iters and self.enable_gan and self.cached_d_loss_value < self.d_guessing_threshold :
                    fake_g_pred = self.net_d(self.output)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    l_g_total += l_g_gan
                    loss_dict['l_g_gan'] = l_g_gan
                else:
                    loss_dict['l_g_gan'] = torch.tensor(0.0, device=self.device)

            # --- Backward pass for generator ---
            scaled_loss = l_g_total / float(self.gradient_accumulation_steps)
            scaled_loss.backward()
            self._accum_steps += 1

            # Step optimizer only when accumulated enough
            if self._accum_steps >= self.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=1.0)
                self.optimizer_g.step()
                self.optimizer_g.zero_grad()
                self._accum_steps = 0  # reset counter

                if self.ema_decay > 0:
                    self.model_ema(decay=self.ema_decay)


            loss_dict['l_d_real'] = self.cached_d_real
            loss_dict['l_d_fake'] = self.cached_d_fake
            loss_dict['out_d_real'] = self.cached_out_d_real
            loss_dict['out_d_fake'] = self.cached_out_d_fake
            loss_dict['d_total_loss'] = torch.tensor(self.cached_d_loss_value, device=self.cached_d_real.device)

            self.log_dict = self.reduce_loss_dict(loss_dict)

            # --- Feed generator output as new LQ for next refinement ---
            self.lq = self.output.detach()  # use model output as next input

            if current_iter % 1000 == 0:
                self.check_ddp_consistency()
