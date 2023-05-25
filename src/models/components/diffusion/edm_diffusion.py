"""
From paper: Elucidating the Design Space of Diffusion-Based Generative Models (https://arxiv.org/abs/2206.00364)
And implementation: https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/elucidated_imagen.py

"""
from math import sqrt
from random import random
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from einops import rearrange, reduce

from src.utils.diffusion_utils import (
    unnormalize_to_zero_to_one,
    is_float_dtype,
)
from src.utils.utils import default, exists
from src.models.components.unets.elucided_unet_wrapper import EDMPrecond
from src.models.components.modules.losses import VGGLoss

# For conditional noise
from src.utils.diffusion_utils import linear_beta_schedule, extract


def gaussian_noise(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    noise = default(noise, lambda: torch.randn_like(x_start))

    return (
        extract(sqrt_alphas_cumprod, t, x_start.shape).to(x_start.device) * x_start
        + extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape).to(x_start.device) * noise
    )


# constants

Hparams_fields = [
    "num_sample_steps",
    "sigma_min",
    "sigma_max",
    "sigma_data",
    "rho",
    "P_mean",
    "P_std",
    "S_churn",
    "S_tmin",
    "S_tmax",
    "S_noise",
]

Hparams = namedtuple("Hparams", Hparams_fields)


# main class
class EDMDiffusion(nn.Module):
    def __init__(
        self,
        unet,
        loss_fn,
        image_size,
        num_frames,
        channels=3,
        inference_model=None,
        cond_drop_prob=0.1,
        dynamic_thresholding=True,
        dynamic_thresholding_percentile=0.95,  # unsure what this was based on perusal of paper
        num_sample_steps=32,  # number of sampling steps
        sigma_min=0.002,  # min noise level
        sigma_max=80,  # max noise level
        sigma_data=0.5,  # standard deviation of data distribution
        rho=7,  # controls the sampling schedule
        P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std=1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn=80,  # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin=0.05,
        S_tmax=50,
        S_noise=1.003,
        augment_pipe=None,
        null_cond_prob=0,  # probability of passing null condition
        # Other losses
        vgg_weight=0,
        smooth_weight=0,
        # RNN related
        unconditional_percent=None,
        # Conditional noise
        add_conditional_noise=None,
    ):
        super().__init__()

        self.unconditional_percent = unconditional_percent

        edm_kwargs = dict(
            dynamic_thresholding_percentile=dynamic_thresholding_percentile,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            dynamic_threshold=dynamic_thresholding,
            concat_in_time=unconditional_percent is not None,
        )
        self.null_cond_prob = null_cond_prob
        self.edm_precond = partial(EDMPrecond, **edm_kwargs)
        self.unet = self.edm_precond(unet)
        if inference_model is None:
            self.inference_model = self.unet
        else:
            self.inference_model = self.edm_precond(inference_model)
        self.loss_fn = loss_fn
        self.image_size = image_size
        self.num_frames = num_frames
        self.channels = channels

        self.right_pad_dims_to_datatype = partial(rearrange, pattern=("b -> b 1 1 1 1"))

        # classifier free guidance

        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.0

        # normalize and unnormalize image functions

        self.unnormalize_img = unnormalize_to_zero_to_one
        self.init_noise_sigma = 1.0

        # elucidating parameters

        hparams = [
            num_sample_steps,
            sigma_min,
            sigma_max,
            sigma_data,
            rho,
            P_mean,
            P_std,
            S_churn,
            S_tmin,
            S_tmax,
            S_noise,
        ]

        self.hparams = Hparams(*hparams)

        self.self_cond = self.unet.model.self_cond if hasattr(self.unet.model, "self_cond") else False

        # Augmentation
        self.augment_pipe = augment_pipe

        self.vgg_weight = vgg_weight
        if vgg_weight > 0:
            self.vgg_loss = VGGLoss(reduction="none")
        self.smooth_weight = smooth_weight

        # Add conditional noise
        self.add_conditional_noise = add_conditional_noise
        if add_conditional_noise == "noise":
            timesteps = 100
            betas = linear_beta_schedule(timesteps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, axis=0)
            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()
            self.training_cond_noise = [
                partial(
                    gaussian_noise,
                    t=torch.tensor([i]),
                    sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                    sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                )
                for i in range(10)
            ]
            self.inference_cond_noise = partial(
                gaussian_noise,
                t=torch.tensor([3]),
                sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            )

        elif add_conditional_noise == "blur":
            self.training_cond_noise = [T.GaussianBlur(kernel_size=(3, 3))]
            self.inference_cond_noise = T.GaussianBlur(kernel_size=(3, 3), sigma=1)

    def update_inf_model(self, model):
        self.inference_model = self.edm_precond(model)

    # sample schedule
    def sample_schedule(self, num_sample_steps, rho, sigma_min, sigma_max, device="cuda"):
        N = num_sample_steps
        inv_rho = 1 / rho

        steps = torch.arange(num_sample_steps, device=device, dtype=torch.float32)
        sigmas = (sigma_max**inv_rho + steps / (N - 1) * (sigma_min**inv_rho - sigma_max**inv_rho)) ** rho

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def sample(
        self,
        batch_size=16,
        cond=None,
        audio=None,
        clamp=True,
        dynamic_threshold=True,
        cond_scale=1.0,
        use_tqdm=True,
        init_images=None,
        skip_steps=None,
        autoregressive_passes=1,
        **kwargs,
    ):
        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        shape = (batch_size, channels, num_frames, image_size, image_size)

        if audio is not None:
            total_frames = num_frames * autoregressive_passes
            assert (
                audio.shape[1] >= total_frames
            ), f"audio must be at least as long as the number of frames we want to generate, audio shape: {audio.shape}, num_frames: {num_frames}, autoregressive_passes: {autoregressive_passes}"
            audio_list = [audio[:, i : i + num_frames] for i in range(0, audio.shape[1], num_frames)]

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma

        sigmas = self.sample_schedule(
            self.hparams.num_sample_steps,
            self.hparams.rho,
            self.hparams.sigma_min,
            self.hparams.sigma_max,
            device=cond.device if exists(cond) else "cuda",
        )

        gammas = torch.where(
            (sigmas >= self.hparams.S_tmin) & (sigmas <= self.hparams.S_tmax),
            min(self.hparams.S_churn / self.hparams.num_sample_steps, sqrt(2) - 1),
            0.0,
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # images is noise at the beginning

        init_sigma = sigmas[0]

        images = init_sigma * torch.randn(shape, device=cond.device if exists(cond) else "cuda")

        # initializing with an image

        if exists(init_images):
            images += init_images

        # keeping track of x0, for self conditioning if needed

        x_start = None
        states = None

        # unet kwargs
        if self.add_conditional_noise is not None:
            cond = self.inference_cond_noise(cond)

        unet_kwargs = dict(
            clamp=clamp, dynamic_threshold=dynamic_threshold, cond_scale=cond_scale, cond=cond, **kwargs
        )

        # gradually denoise

        initial_step = default(skip_steps, 0)
        sigmas_and_gammas = sigmas_and_gammas[initial_step:]

        total_steps = len(sigmas_and_gammas)

        full_sequence = []

        previous_states = [None] * len(sigmas_and_gammas)

        for i_reg in range(autoregressive_passes):
            current_states = []
            for i, (sigma, sigma_next, gamma) in tqdm(
                enumerate(sigmas_and_gammas), total=total_steps, desc="sampling time step", disable=not use_tqdm
            ):
                unet_kwargs["audio"] = None if audio is None else audio_list[i_reg]

                sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

                eps = self.hparams.S_noise * torch.randn(
                    shape, device=cond.device if exists(cond) else "cuda"
                )  # stochastic sampling

                sigma_hat = sigma + gamma * sigma
                added_noise = sqrt(sigma_hat**2 - sigma**2) * eps

                images_hat = images + added_noise

                self_cond = x_start if self.self_cond else None

                states_in = previous_states[i]
                model_output, states = self.inference_model.forward_with_cond_scale(
                    images_hat,
                    sigma_hat,
                    self_cond=self_cond,
                    states=states_in,
                    latent_self_cond=states,  # If RIN is used, this is the latent self conditioning
                    **unet_kwargs,
                )
                assert torch.isnan(model_output).sum() == 0, "model output is nan"
                current_states.append(states)

                denoised_over_sigma = (images_hat - model_output) / sigma_hat

                images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

                # second order correction, if not the last timestep
                has_second_order_correction = sigma_next != 0

                if has_second_order_correction:
                    self_cond = model_output if self.self_cond else None

                    states_in = current_states[-1]
                    model_output_next, states = self.inference_model.forward_with_cond_scale(
                        images_next,
                        sigma_next,
                        states=states_in,
                        self_cond=self_cond,
                        latent_self_cond=states,  # If RIN is used, this is the latent self conditioning
                        **unet_kwargs,
                    )
                    current_states[-1] = states

                    denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                    images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (
                        denoised_over_sigma + denoised_prime_over_sigma
                    )

                images = images_next

                x_start = (
                    model_output if not has_second_order_correction else model_output_next
                )  # save model output for self conditioning

            previous_states = current_states
            images = images.clamp(-1.0, 1.0)
            full_sequence.append(images)

            if self.add_conditional_noise is not None:
                cond = self.inference_cond_noise(images[:, :, -1])
            else:
                cond = images[:, :, -1]
            unet_kwargs["cond"] = cond  # use last image as cond for next pass

        if len(full_sequence) == 1:
            full_sequence = full_sequence[0]
        else:
            full_sequence = torch.concatenate(full_sequence, dim=2)

        return self.unnormalize_img(full_sequence)

    # training
    def select_random_frames(self, tensor, n):
        B, C, T, H, W = tensor.size()
        selected_frames = torch.zeros((B, C, n, H, W), device=tensor.device, dtype=tensor.dtype)

        for i in range(B):
            frame_indices = torch.randperm(T)[:n]
            selected_frames[i] = tensor[i, :, frame_indices, :, :]

        return selected_frames

    def loss_weight(self, sigma_data, sigma):
        return (sigma**2 + sigma_data**2) * (sigma * sigma_data) ** -2

    def noise_distribution(self, P_mean, P_std, batch_size, device="cuda"):
        return (P_mean + P_std * torch.randn((batch_size,), device=device)).exp()

    def variable_length_loss(self, x, y, lengths, loss_func, n_frames_loss=None):
        if n_frames_loss is not None:
            # Select n_frames_loss indexes to compute the loss on
            x_sub = self.select_random_frames(x, n_frames_loss)
            y_sub = self.select_random_frames(y, n_frames_loss)
        else:
            x_sub = x
            y_sub = y

        batch_size = x.size(0)
        loss = []
        if lengths is None:
            return reduce(loss_func(x_sub, y_sub, reduction="none"), "b ... -> b", "mean")
        for i in range(batch_size):
            loss += [loss_func(x_sub[i, :, : lengths[i]], y_sub[i, :, : lengths[i]], reduction="mean")]
        return torch.stack(loss)  # B * 1

    def variable_smooth_loss(self, x, lengths, loss_func, n_frames_loss=None):
        batch_size = x.size(0)
        loss = []

        if n_frames_loss is not None:
            # Select n_frames_loss indexes to compute the loss on
            x_sub = self.select_random_frames(x, n_frames_loss)
        else:
            x_sub = x

        def smooth_loss(x, reduction="mean"):
            return loss_func(x[:, :-1], x[:, 1:], reduction=reduction)

        if lengths is None:
            return reduce(smooth_loss(x_sub, "none"), "b ... -> b", "mean")

        for i in range(batch_size):
            if lengths[i] > 1:
                loss += [smooth_loss(x_sub[i, :, : lengths[i]])]
        assert len(loss) > 0
        return torch.stack(loss)

    def diffusion_step(
        self,
        images,
        cond_image=None,
        audio=None,
        lengths=None,
        n_frames_loss=None,
        **kwargs,
    ):
        assert is_float_dtype(images.dtype), f"images tensor needs to be floats but {images.dtype} dtype found instead"

        batch_size, num_frames = images.shape[0], images.shape[2]

        labels = None
        if self.augment_pipe is not None:
            cat_images = torch.cat([images, cond_image.unsqueeze(2)], dim=2) if exists(cond_image) else images
            cat_images, labels = self.augment_pipe(cat_images)
            images, cond_image = cat_images[:, :, :num_frames], cat_images[:, :, -1]

        # get the sigmas
        sigmas = self.noise_distribution(self.hparams.P_mean, self.hparams.P_std, batch_size, device=images.device)
        padded_sigmas = self.right_pad_dims_to_datatype(sigmas)

        # noise
        noise = torch.randn_like(images, device=images.device)
        noised_images = images + padded_sigmas * noise  # alphas are 1. in the paper

        # unet kwargs
        if self.add_conditional_noise is not None:
            training_cond_noise = np.random.choice(self.training_cond_noise)
            cond_image = training_cond_noise(cond_image)

        unet_kwargs = dict(
            cond=cond_image,
            audio=audio,
            cond_drop_prob=self.cond_drop_prob,
            augment_labels=labels,
            null_cond_prob=self.null_cond_prob,
            lengths=lengths,
            unconditional_percent=self.unconditional_percent,
            **kwargs,
        )

        # self conditioning - https://arxiv.org/abs/2208.04202 - training will be 25% slower
        self_cond = self.self_cond

        if self_cond and random() < 0.5:
            with torch.no_grad():
                pred_x0, states = self.unet(noised_images, sigmas, **unet_kwargs)
                pred_x0 = pred_x0.detach()
                states = states.detach()

            unet_kwargs = {**unet_kwargs, "self_cond": pred_x0, "latent_self_cond": states}

        # get prediction
        denoised_images, _ = self.unet(noised_images, sigmas, **unet_kwargs)

        loss_dict = {}

        # losses
        losses = self.variable_length_loss(denoised_images, images, lengths, self.loss_fn, n_frames_loss=n_frames_loss)

        # loss weighting
        losses = losses * self.loss_weight(self.hparams.sigma_data, sigmas)
        losses = losses.mean()  # return average loss
        loss_dict["loss_diffusion"] = losses

        if self.smooth_weight > 0:
            smooth_loss = self.smooth_weight * self.variable_smooth_loss(
                denoised_images, lengths, self.loss_fn, n_frames_loss=n_frames_loss
            )
            smooth_loss = smooth_loss * self.loss_weight(self.hparams.sigma_data, sigmas)
            smooth_loss = smooth_loss.mean()
            loss_dict["loss_smooth"] = smooth_loss
            losses += smooth_loss

        if self.vgg_weight > 0:
            vgg_loss = (
                self.variable_length_loss(
                    unnormalize_to_zero_to_one(denoised_images),
                    unnormalize_to_zero_to_one(images),
                    lengths,
                    self.vgg_loss,
                    n_frames_loss=n_frames_loss,
                )
                * self.vgg_weight
            )
            vgg_loss = vgg_loss * self.loss_weight(self.hparams.sigma_data, sigmas)
            vgg_loss = vgg_loss.mean()
            losses += vgg_loss
            loss_dict["loss_vgg"] = vgg_loss

        return losses, loss_dict
