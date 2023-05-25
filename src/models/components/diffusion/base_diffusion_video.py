# from collections import namedtuple
# from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.diffusion_utils import (
    cosine_beta_schedule,
    linear_beta_schedule,
    extract,
    unnormalize_to_zero_to_one,
)
from src.utils.utils import default, exists, check_shape
from einops import reduce, rearrange
from tqdm import tqdm
from functools import partial


def identity(t, *args, **kwargs):
    return t


class BaseDiffusionVid(nn.Module):
    def __init__(
        self,
        denoise_fn,
        loss_fn,
        image_size,
        num_frames,
        channels=3,
        timesteps=1000,
        sampling_timesteps=None,
        beta_schedule="cosine",
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.9,
        ddim_sampling_eta=0.0,
        inference_model=None,
        use_latent=False,
        # To be implemented
        augment_pipe=None,
        unconditional_percent=None,
        add_conditional_noise=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        if inference_model is None:
            self.inference_model = denoise_fn
        else:
            self.inference_model = inference_model

        # Precomputing constant parameters
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_fn = loss_fn

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))  # noqa

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

        self.unnormalize_img = unnormalize_to_zero_to_one if not use_latent else lambda x: x

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1.0):

        try:
            noise = self.inference_model.ema_model.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        except AttributeError:
            noise = self.inference_model.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)

        if clip_denoised:
            s = 1.0
            if self.use_dynamic_thres:
                s = torch.quantile(rearrange(x_recon, "b ... -> b (...)").abs(), self.dynamic_thres_percentile, dim=-1)

                s.clamp_(min=1.0)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond=None, cond_scale=1.0, clip_denoised=True):
        b, *_, _ = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale
        )
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None, cond_scale=1.0):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(
            reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond=cond, cond_scale=cond_scale
            )

        return self.unnormalize_img(img)

    @torch.no_grad()
    def ddim_sample(self, shape, cond=None, cond_scale=1.0, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            # self_cond = x_start if self.self_condition else None
            # pred_noise, x_start, *_ = self.forward(img, time_cond, id_cond, clip_x_start=clip_denoised)
            # pred_noise = self.denoise_fn.forward_with_cond_scale(img, time_cond, cond=cond, cond_scale=cond_scale)
            try:
                pred_noise = self.inference_model.ema_model.forward_with_cond_scale(
                    img, time_cond, cond=cond, cond_scale=cond_scale
                )
            except AttributeError:
                pred_noise = self.inference_model.forward_with_cond_scale(
                    img, time_cond, cond=cond, cond_scale=cond_scale
                )
            x_start = self.predict_start_from_noise(img, t=time_cond, noise=pred_noise)
            maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_denoised else identity
            x_start = maybe_clip(x_start)
            # if clip_denoised:
            #     s = 1.0
            #     if self.use_dynamic_thres:
            #         s = torch.quantile(rearrange(x_start, "b ... -> b (...)").abs(), self.dynamic_thres_percentile, dim=-1)

            #         s.clamp_(min=1.0)
            #         s = s.view(-1, *((1,) * (x_start.ndim - 1)))

            #     # clip by threshold, depending on whether static or dynamic
            #     x_start = x_start.clamp(-s, s) / s
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        img = self.unnormalize_img(img)
        return img

    @torch.no_grad()
    def sample(self, cond=None, cond_scale=1.0, batch_size=16, **kwargs):
        # device = next(self.denoise_fn.parameters()).device

        # if is_list_str(cond):
        #     cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, num_frames, image_size, image_size), cond=cond, cond_scale=cond_scale)
        # return self.p_sample_loop(
        #     (batch_size, channels, num_frames, image_size, image_size), cond=cond, cond_scale=cond_scale
        # )

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc="interpolation sample time step", total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond=None, noise=None, **kwargs):
        # b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if is_list_str(cond):
        #     cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls)
        #     cond = cond.to(device)

        x_recon = self.denoise_fn(x_noisy, t, cond=cond, **kwargs)

        loss_dict = {}

        loss = self.loss_fn(noise, x_recon, reduction="none")

        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        loss_dict["loss_diffusion"] = loss
        # loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        return loss, loss_dict

    def diffusion_step(self, x, *args, **kwargs):
        b, device, img_size, = (
            x.shape[0],
            x.device,
            self.image_size,
        )
        check_shape(x, "b c f h w", c=self.channels, f=self.num_frames, h=img_size, w=img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # x = normalize_img(x)
        # TODO: check if this is correct
        # mask_empty_frames = x
        return self.p_losses(x, t, *args, **kwargs)
