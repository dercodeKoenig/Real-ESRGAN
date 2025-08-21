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
class RealESRGANModel2(SRGANModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealESRGANModel2, self).__init__(opt)

        self.net_g.compile(mode="max-autotune", dynamic=False, fullgraph=True)
        self.net_d.compile(mode="max-autotune", dynamic=False, fullgraph=True)
        #self.net_g.compile(dynamic=False, fullgraph=True)
        #self.net_d.compile(dynamic=False, fullgraph=True)

        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

        # Initialize AMP components
        self.scaler_g = GradScaler()
        self.scaler_d = GradScaler()

        # Gradient accumulation parameters
        self.gradient_accumulation_steps = opt['train'].get('gradient_accumulation_steps', 1)

        self.percept_warmup_iters = opt['train'].get('percept_warmup_iters', 0)  # warmup steps before adding vgg loss
        self.gan_warmup_iters = opt['train'].get('gan_warmup_iters', 0)  # warmup steps before adding gan loss

        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")

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
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
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
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
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

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealESRGANModel2, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def optimize_parameters(self, current_iter):

        is_accumulation_update_step = ((current_iter+1) % self.gradient_accumulation_steps == 0)
        is_first_accumulation_step = (current_iter % self.gradient_accumulation_steps == 0)

        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        # Only zero gradients at the start of accumulation cycle
        if is_first_accumulation_step:
            self.optimizer_g.zero_grad()

        with autocast('cuda'):
            self.output = self.net_g(self.lq)
            if self.cri_ldl:
                self.output_ema = self.net_g_ema(self.lq)

            l_g_total = 0
            loss_dict = OrderedDict()

            # pixel loss
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
            # perceptual loss
            if self.cri_perceptual and current_iter > self.percept_warmup_iters:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            if current_iter > self.gan_warmup_iters:
                fake_g_pred = self.net_d(self.output)
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan'] = l_g_gan
            else:
                loss_dict['l_g_gan'] = torch.tensor(0.0, device='cuda')

            # Scale loss by accumulation steps for proper averaging
            l_g_total = l_g_total / self.gradient_accumulation_steps

        self.scaler_g.scale(l_g_total).backward()

        # Only step optimizer and scaler when accumulation cycle is complete
        if is_accumulation_update_step:
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()

        # Clone output tensor to avoid CUDA graph issues
        output_for_d = self.output.clone().detach()

        # optimize net_d with adaptive strategy
        for p in self.net_d.parameters():
            p.requires_grad = True

        loss_dict['l_d_real'] =  torch.tensor(0.0, device='cuda')
        loss_dict['l_d_fake'] =  torch.tensor(0.0, device='cuda')
        loss_dict['out_d_real'] =  torch.tensor(0.0, device='cuda')
        loss_dict['out_d_fake'] =  torch.tensor(0.0, device='cuda')

        if current_iter > self.gan_warmup_iters:
            if is_first_accumulation_step:
                self.optimizer_d.zero_grad()

            # Process real samples
            with autocast('cuda'):
                real_d_pred = self.net_d(gan_gt)
                fake_d_pred = self.net_d(output_for_d)

                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)

                loss_dict['l_d_real'] = l_d_real.detach()
                loss_dict['l_d_fake'] = l_d_fake.detach()

                # Scale discriminator losses by accumulation steps
                l_d_real = l_d_real / self.gradient_accumulation_steps
                l_d_fake = l_d_fake / self.gradient_accumulation_steps

            self.scaler_d.scale(l_d_real).backward()
            self.scaler_d.scale(l_d_fake).backward()

            if is_accumulation_update_step:
                self.scaler_d.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), max_norm=1.0)

                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()

            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        # Only update EMA when accumulation cycle is complete
        if self.ema_decay > 0 and is_accumulation_update_step:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)