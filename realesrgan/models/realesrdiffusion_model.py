import numpy as np
import random
import torch
from collections import OrderedDict

from basicsr import get_root_logger, build_network
from torch.nn import functional as F
from torch.amp import autocast, GradScaler
import math

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from tqdm import tqdm

# idk claude says it is faster
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


@MODEL_REGISTRY.register()
class DiffusionSRModel(SRModel):
    """Diffusion Super-Resolution Model for training diffusion-based super-resolution.

    It performs:
    1. randomly synthesize LQ images in GPU tensors with dynamic scaling
    2. add noise to GT images for diffusion training
    3. train the network to predict noise
    """

    def __init__(self, opt):
        super(DiffusionSRModel, self).__init__(opt)

        self.net_g.compile(mode="max-autotune", dynamic=False, fullgraph=True)
        #self.net_g.compile(dynamic=False, fullgraph=True)

        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.queue_size = opt.get('queue_size', 180)

        # Initialize AMP components
        self.scaler_g = GradScaler()

        # Diffusion parameters
        self.min_scale = opt['train'].get('min_scale', 1.0)
        self.max_scale = opt['train'].get('max_scale', 4.0)

        self.test_steps = opt['val'].get("test_steps", 20)
        self.test_step_size = opt['val'].get("test_step_size", 0.1)

        target_loss = opt['train'].get('loss', '')
        if target_loss == 'mse':
            self.loss_fn = F.mse_loss
        elif target_loss == 'l1':
            self.loss_fn = F.l1_loss
        else:
            raise "no loss provided"

        print(f"Scale range: {self.min_scale}-{self.max_scale}x")
        print("test steps:", self.test_steps)
        print("test step size:", self.test_step_size)
        print("use loss:", self.loss_fn)

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """Training pair pool for increasing diversity in a batch."""
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:
            # shuffle and dequeue
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]

            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()

            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader and add degradations with dynamic scaling."""
        if self.is_train and self.opt.get('high_order_degradation', True):
            self.gt = data['gt'].to(self.device)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # Random scale factor between min_scale and max_scale
            scale_factor = random.uniform(self.min_scale, self.max_scale)
            target_h = int(ori_h / scale_factor)
            target_w = int(ori_w / scale_factor)

            # ----------------------- The first degradation process ----------------------- #
            out = filter2D(self.gt, self.kernel1)

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
            self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, scale=1)

            # Training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()
        else:
            # For validation - interpolate LQ to GT size
            self.lq_orig = data['lq'].to(self.device)
            self.gt = data['gt'].to(self.device)
            # Interpolate LQ to GT size
            self.lq = F.interpolate(self.lq_orig, size=self.gt.shape[-2:], mode='bicubic', align_corners=False)

    def optimize_parameters(self, current_iter):
        """Optimize parameters using diffusion training."""
        self.optimizer_g.zero_grad()

        with autocast('cuda'):
            # Sample random timesteps for each sample in batch
            batch_size = self.gt.size(0)
            weights = torch.rand((batch_size,), device=self.device)

            # Sample noise
            noise = torch.randn_like(self.gt)

            # Add noise to GT images
            w = weights.view(-1, 1, 1, 1)
            noisy_gt =  self.gt + w * noise

            # Concatenate noisy GT with interpolated LQ as input
            model_input = torch.cat([noisy_gt, self.lq], dim=1)  # [B, 6, H, W] if RGB

            # Predict noise
            predicted_noise = self.net_g(model_input)

            # Compute loss
            loss = loss_fn(predicted_noise, w * noise)

        # Backward pass
        self.scaler_g.scale(loss).backward()

        # Gradient clipping
        self.scaler_g.unscale_(self.optimizer_g)
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=1.0)

        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()

        # Logging
        loss_dict = OrderedDict()
        loss_dict['l_diffusion'] = loss.detach()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    @torch.no_grad()
    def sample_naive(self, lq, model):
        """Naive denoising: iteratively subtract a fraction of predicted noise."""

        batch_size, _, h, w = lq.shape

        # Start from pure noise
        x = torch.randn(batch_size, 3, h, w, device=self.device) + lq

        for _ in range(self.test_steps):
            # Predict noise
            model_input = torch.cat([x, lq], dim=1)
            predicted_noise = model(model_input)

            # Update rule: subtract a fraction of predicted noise
            x = x - self.test_step_size * predicted_noise

        # Optionally clamp to [0,1] if your model outputs images in that range
        return torch.clamp(x, 0, 1)

    def test(self):
        """Test function for inference."""
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.sample_naive(self.lq, self.net_g_ema)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.sample_naive(self.lq, self.net_g)
            self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Validation function."""
        self.is_train = False
        super(DiffusionSRModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True