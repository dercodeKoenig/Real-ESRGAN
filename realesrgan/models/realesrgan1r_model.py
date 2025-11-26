import numpy as np
import random
import torch
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
            self.net_g.compile(mode="max-autotune", dynamic=False, fullgraph=True)
            #self.net_d.compile(mode="max-autotune", dynamic=False, fullgraph =True)
            #self.net_g.compile(dynamic=False)
            self.net_d.compile(dynamic=False)

        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

        self.min_scale = opt['train'].get('min_scale', 1.0)
        self.max_scale = opt['train'].get('max_scale', 4.0)
        print("scales:", self.min_scale, self.max_scale)

        # Initialize AMP components
        self.scaler_g = GradScaler()
        self.scaler_d = GradScaler()

        self.d_loss_threshold = opt['train'].get('d_loss_threshold',
                                                 0.6)  # When D loss < this, slow down D updates. When D loss < this / 2, skip the update
        self.d_slow_iters = opt['train'].get('d_slow_iters', 11)  # Update D every x steps when loss is low
        self.d_normal_iters = opt['train'].get('d_normal_iters', 1)  # Normal D update frequency

        self.gan_warmup_iters = opt['train'].get('gan_warmup_iters', 0)  # warmup steps before gan training
        self.percept_warmup_iters = opt['train'].get('percept_warmup_iters', 0)  # warmup steps before adding vgg loss

        # Track discriminator update counter and cached values
        self.cached_d_loss_value = float('inf')  # Initialize with high value to ensure first update

        # Cache tensor values instead of floats - initialize as tensors
        self.cached_d_real = torch.tensor(0.0, device='cuda')
        self.cached_d_fake = torch.tensor(0.0, device='cuda')
        self.cached_out_d_real = torch.tensor(0.0, device='cuda')
        self.cached_out_d_fake = torch.tensor(0.0, device='cuda')

        print(f"D loss threshold: {self.d_loss_threshold}")
        print(f"D slow update interval: {self.d_slow_iters}")
        print(f"D normal update interval: {self.d_normal_iters}")
        print("")
        print("gan_warmup_iters:", self.gan_warmup_iters)
        print("percept_warmup_iters:", self.percept_warmup_iters)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

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

            # Training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
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
        # Always skip during initial discriminator training period
        if current_iter <= self.gan_warmup_iters:
            return False

        # Adaptive strategy based on cached discriminator loss from previous iteration
        if self.cached_d_loss_value < self.d_loss_threshold:
            # D is too strong, update less frequently
            return current_iter % self.d_slow_iters == 0
        else:
            # D loss is reasonable, update normally
            return current_iter % self.d_normal_iters == 0

    def optimize_parameters(self, current_iter):
        num_refine = self.opt.get('num_refine_steps', 2)  # e.g. set in YAML or default

        for refine_step in range(num_refine):
            # --- 1. choose GTs depending on usm options ---
            l1_gt = self.gt_usm if self.opt['l1_gt_usm'] else self.gt
            percep_gt = self.gt_usm if self.opt['percep_gt_usm'] else self.gt
            gan_gt = self.gt_usm if self.opt['gan_gt_usm'] else self.gt

            # --- 2. freeze discriminator and optimize generator ---
            for p in self.net_d.parameters():
                p.requires_grad = False
            self.optimizer_g.zero_grad()

            with autocast('cuda'):
                self.output = self.net_g(self.lq)
                if self.cri_ldl:
                    self.output_ema = self.net_g_ema(self.lq)

                l_g_total = 0
                loss_dict = OrderedDict()

                # pixel / ldl / perceptual / gan losses as in your code
                if self.cri_pix:
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

                if current_iter > self.gan_warmup_iters:
                    fake_g_pred = self.net_d(self.output)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    l_g_total += l_g_gan
                    loss_dict['l_g_gan'] = l_g_gan
                else:
                    loss_dict['l_g_gan'] = torch.tensor(0.0, device='cuda')

            # backward and update
            self.scaler_g.scale(l_g_total).backward()
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()

            output_for_d = self.output.detach().clone()

            # --- 3. optimize discriminator as usual ---
            for p in self.net_d.parameters():
                p.requires_grad = True

            should_update_d = self._should_update_discriminator(current_iter)
            if should_update_d:
                self.optimizer_d.zero_grad()
                with autocast('cuda'):
                    real_d_pred = self.net_d(gan_gt)
                    fake_d_pred = self.net_d(output_for_d)
                    l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                    l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                d_total_loss = (l_d_real + l_d_fake).item()

                if d_total_loss >= self.d_loss_threshold * 0.5:
                    self.scaler_d.scale(l_d_real).backward()
                    self.scaler_d.scale(l_d_fake).backward()
                    self.scaler_d.unscale_(self.optimizer_d)
                    torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), 1.0)
                    self.scaler_d.step(self.optimizer_d)
                    self.scaler_d.update()

                self.cached_d_loss_value = d_total_loss
                self.cached_d_real = l_d_real.detach()
                self.cached_d_fake = l_d_fake.detach()
                self.cached_out_d_real = torch.mean(real_d_pred.detach())
                self.cached_out_d_fake = torch.mean(fake_d_pred.detach())

            loss_dict['l_d_real'] = self.cached_d_real
            loss_dict['l_d_fake'] = self.cached_d_fake
            loss_dict['out_d_real'] = self.cached_out_d_real
            loss_dict['out_d_fake'] = self.cached_out_d_fake
            loss_dict['d_total_loss'] = torch.tensor(self.cached_d_loss_value, device=self.cached_d_real.device)

            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)

            self.log_dict = self.reduce_loss_dict(loss_dict)

            # --- 4. Feed generator output as new LQ for next refinement ---
            self.lq = self.output.detach()  # use model output as next input

