import torch
from torch import sqrt
from torch import nn
from torch.special import expm1

from tqdm import tqdm
from einops import repeat

from src.utils.diffusion_utils import (
    logsnr_schedule_cosine,
    logsnr_schedule_shifted,
    logsnr_schedule_interpolated,
    unnormalize_to_zero_to_one,
    right_pad_dims_to,
)
from src.utils.utils import exists, default

# main gaussian diffusion class


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        loss_fn,
        image_size,
        num_frames,
        channels=3,
        pred_objective="v",
        noise_schedule=logsnr_schedule_cosine,
        noise_d=None,
        noise_d_low=None,
        noise_d_high=None,
        num_sample_steps=500,
        clip_sample_denoised=True,
        # To be implemented
        inference_model=None,
        augment_pipe=None,
        unconditional_percent=None,
        add_conditional_noise=None,
        use_latent=False,
    ):
        super().__init__()
        assert pred_objective in {"v", "eps"}, "whether to predict v-space (progressive distillation paper) or noise"

        self.model = model
        self.loss_fn = loss_fn

        # image dimensions

        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames

        # training objective

        self.pred_objective = pred_objective

        # noise schedule

        assert not all(
            [*map(exists, (noise_d, noise_d_low, noise_d_high))]
        ), "you must either set noise_d for shifted schedule, or noise_d_low and noise_d_high for shifted and interpolated schedule"

        # determine shifting or interpolated schedules

        self.log_snr = noise_schedule

        if exists(noise_d):
            self.log_snr = logsnr_schedule_shifted(self.log_snr, image_size, noise_d)

        if exists(noise_d_low) or exists(noise_d_high):
            assert exists(noise_d_low) and exists(noise_d_high), "both noise_d_low and noise_d_high must be set"

            self.log_snr = logsnr_schedule_interpolated(self.log_snr, image_size, noise_d_low, noise_d_high)

        # sampling

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next, cond=None):

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, " -> b", b=x.shape[0])
        pred = self.model(x, batch_log_snr, cond=cond)

        if self.pred_objective == "v":
            x_start = alpha * x - sigma * pred

        elif self.pred_objective == "eps":
            x_start = (x - sigma * pred) / alpha

        x_start.clamp_(-1.0, 1.0)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    # sampling related functions

    @torch.no_grad()
    def p_sample(self, x, time, time_next, cond=None, cond_scale=1.0):

        model_mean, model_variance = self.p_mean_variance(x=x, time=time, time_next=time_next, cond=cond)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None, cond_scale=1.0):

        img = torch.randn(shape, device=self.device)
        steps = torch.linspace(1.0, 0.0, self.num_sample_steps + 1, device=self.device)

        for i in tqdm(range(self.num_sample_steps), desc="sampling loop time step", total=self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next, cond=cond, cond_scale=cond_scale)

        img.clamp_(-1.0, 1.0)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, cond=None, cond_scale=1.0, batch_size=16, **kwargs):
        batch_size = cond.shape[0] if exists(cond) else batch_size
        return self.p_sample_loop(
            (batch_size, self.channels, self.num_frames, self.image_size, self.image_size), cond=cond
        )

    # training related functions - noise prediction

    def q_sample(self, x_start, times, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised = x_start * alpha + noise * sigma

        return x_noised, log_snr

    def p_losses(self, x_start, times, cond=None, noise=None, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)
        model_out = self.model(x, log_snr, cond=cond, **kwargs)

        loss_dict = {}

        if self.pred_objective == "v":
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x_start

        elif self.pred_objective == "eps":
            target = noise

        loss = self.loss_fn(model_out, target)

        return loss, loss_dict

    def diffusion_step(self, x, *args, **kwargs):

        times = torch.zeros((x.shape[0],), device=self.device).float().uniform_(0, 1)

        return self.p_losses(x, times, *args, **kwargs)
