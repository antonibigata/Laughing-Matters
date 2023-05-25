import torch
from einops import rearrange
from random import random


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        model,
        dynamic_thresholding_percentile=0.95,
        dynamic_threshold=True,
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        concat_in_time=False,
    ):
        super().__init__()

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.dynamic_threshold = dynamic_threshold
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile
        self.model = model
        self.concat_in_time = concat_in_time

        # dynamic thresholding

    def threshold_x_start(self, x_start):
        if not self.dynamic_threshold:
            return x_start.clamp(-1.0, 1.0)

        s = torch.quantile(rearrange(x_start, "b ... -> b (...)").abs(), self.dynamic_thresholding_percentile, dim=-1)

        s.clamp_(min=1.0)
        s = right_pad_dims_to(x_start, s)
        return x_start.clamp(-s, s) / s

    # derived preconditioning params - Table 1

    def c_skip(self, sigma_data, sigma):
        return (sigma_data**2) / (sigma**2 + sigma_data**2)

    def c_out(self, sigma_data, sigma):
        return sigma * sigma_data * (sigma_data**2 + sigma**2) ** -0.5

    def c_in(self, sigma_data, sigma):
        return 1 * (sigma**2 + sigma_data**2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    def forward(self, noised_images, sigma, *, clamp=False, cond=None, unconditional_percent=None, **kwargs):
        batch, device = noised_images.shape[0], noised_images.device

        if unconditional_percent is not None and self.concat_in_time:
            # Either ground truth or unconditional
            if random() < unconditional_percent:
                cond = None
            else:
                # cond_image = images[:, :, 0]
                noised_images = noised_images[:, :, 1:]

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b () () () ()")

        net_out = self.model(
            self.c_in(self.sigma_data, padded_sigma) * noised_images, self.c_noise(sigma), cond=cond, **kwargs
        )
        states = None
        if isinstance(net_out, tuple):
            states = net_out[1]
            net_out = net_out[0]

        if noised_images.shape[2] < net_out.shape[2]:
            noised_images = torch.cat([cond.unsqueeze(2), noised_images], dim=2)

        out = (
            self.c_skip(self.sigma_data, padded_sigma) * noised_images
            + self.c_out(self.sigma_data, padded_sigma) * net_out
        )

        if not clamp:
            return out, states
        return self.threshold_x_start(out), states

    def forward_with_cond_scale(self, noised_images, sigma, *, cond=None, clamp=False, **kwargs):
        batch, device = noised_images.shape[0], noised_images.device

        if self.concat_in_time and cond is not None:
            noised_images = noised_images[:, :, 1:]

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b () () () ()")

        try:  # TODO: make this cleaner
            net_out = self.model.ema_model.forward_with_cond_scale(
                self.c_in(self.sigma_data, padded_sigma) * noised_images, self.c_noise(sigma), cond=cond, **kwargs
            )
        except AttributeError:
            net_out = self.model.forward_with_cond_scale(
                self.c_in(self.sigma_data, padded_sigma) * noised_images, self.c_noise(sigma), cond=cond, **kwargs
            )
        states = None
        if isinstance(net_out, tuple):
            states = net_out[1]
            net_out = net_out[0]

        if noised_images.shape[2] < net_out.shape[2]:
            noised_images = torch.cat([cond.unsqueeze(2), noised_images], dim=2)

        out = (
            self.c_skip(self.sigma_data, padded_sigma) * noised_images
            + self.c_out(self.sigma_data, padded_sigma) * net_out
        )

        if not clamp:
            return out, states
        return self.threshold_x_start(out), states

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
