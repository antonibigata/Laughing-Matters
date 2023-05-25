import torch
from torchmetrics import Metric
from copy import deepcopy
import numpy as np
import scipy
from typing import Any
from torch import Tensor
from torch.autograd import Function
import torch.nn as nn
from torchmetrics.utilities import rank_zero_info
from torchvision.transforms import transforms

from einops import rearrange

# import sys

# sys.path.append("/vol/paramonos2/projects/antoni/code/generating_laugh")
from src.utils.open_url import open_url


# def _compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
#     mu_gen, sigma_gen = compute_stats(feats_fake)
#     mu_real, sigma_real = compute_stats(feats_real)

#     m = np.square(mu_gen - mu_real).sum()
#     s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
#     fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

#     return float(fid)


# def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     mu = feats.mean(axis=0)  # [d]
#     sigma = np.cov(feats, rowvar=False)  # [d, d]

#     return mu, sigma


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    All credit to `Square Root of a Positive Definite Matrix`_
    """

    @staticmethod
    def forward(ctx: Any, input_data: Tensor) -> Tensor:
        # TODO: update whenever pytorch gets an matrix square root function
        # Issue: https://github.com/pytorch/pytorch/issues/9983
        m = input_data.detach().cpu().numpy().astype(np.float_)
        scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)

        sqrtm = torch.from_numpy(scipy_res.real).to(input_data)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


def _compute_fvd(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps: float = 1e-6) -> Tensor:
    r"""Adjusted version of `Fid Score`_
    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).
    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant - used if sigma_1 @ sigma_2 matrix is singular
    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2

    covmean = sqrtm(sigma1.mm(sigma2))
    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        rank_zero_info(f"FID calculation produces singular product; adding {eps} to diagonal of covariance estimates")
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

    tr_covmean = torch.trace(covmean)
    print(diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


class FVD(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    def __init__(self, reset_real_features: bool = False, resize=None):
        super().__init__()
        self.reset_real_features = reset_real_features
        detector_url = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
        self.detector_kwargs = dict(
            rescale=True, resize=True, return_features=True
        )  # Return raw features before the softmax layer.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_used = device
        with open_url(detector_url, verbose=False) as f:
            self.detector = torch.jit.load(f).eval().to(device)
        for param in self.detector.parameters():
            param.requires_grad = False

        self.resize = resize

        self.transform = transforms.Resize((self.resize, self.resize)) if self.resize else nn.Identity()

        dummy_input = torch.randn(1, 3, 32, 224, 224).to(device)
        feat_dummy = self.detector(dummy_input, **self.detector_kwargs)  # Warmup.
        num_features = feat_dummy.shape[-1]
        mx_nb_feets = (num_features, num_features)
        self.add_state("real_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum", torch.zeros(mx_nb_feets).double(), dist_reduce_fx="sum")
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state("fake_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum", torch.zeros(mx_nb_feets).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, video: torch.Tensor, real: bool):
        """Update the state with extracted features.
        Args:
            video: tensor with video feed to the feature extractor (batch, channels, time, height, width)
            video in range [0, 1]
            real: bool indicating if ``video`` belong to the real or the fake distribution
        """
        # video = rearrange(video, "b c t h w -> b t c h w")
        video = self._to_detector_input(video).contiguous()
        # print(video.shape, video.dtype, video.min(), video.max())
        features = self.detector(video.to(self.device_used), **self.detector_kwargs)

        # self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += video.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += video.shape[0]

    def _to_detector_input(self, video: torch.Tensor) -> torch.Tensor:
        """Convert video tensor to the input format of the feature extractor."""

        video *= 255
        T = video.shape[2]
        video = self.transform(rearrange(video, "b c t h w -> (b t) c h w"))

        return rearrange(video, "(b t) c h w -> b c t h w", t=T).clamp(0, 255)

    def compute(self):
        """Calculate FVD score based on accumulated extracted features from the two distributions."""
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)

        fvd = _compute_fvd(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake)
        # print(fvd)
        if not torch.isfinite(fvd).all():
            mu1 = mean_real.squeeze(0)
            mu2 = mean_fake.squeeze(0)
            sigma1 = cov_real
            sigma2 = cov_fake
            eps = 1e-6
            print(
                torch.isfinite(cov_real).all(),
                torch.isfinite(cov_fake).all(),
                torch.isfinite(mean_real).all(),
                torch.isfinite(mean_fake).all(),
            )
            print(self.fake_features_num_samples)
            print(self.real_features_num_samples)
            print(torch.isfinite(cov_real_num).all(), torch.isfinite(cov_fake_num).all())
            print("--" * 100)
            diff = mu1 - mu2

            covmean = sqrtm(sigma1.mm(sigma2))
            # Product might be almost singular
            if not torch.isfinite(covmean).all():
                rank_zero_info(
                    f"FID calculation produces singular product; adding {eps} to diagonal of covariance estimates"
                )
                offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
                covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

            tr_covmean = torch.trace(covmean)
            fvd = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
            print(
                torch.isfinite(diff).all(),
                torch.isfinite(covmean).all(),
                torch.isfinite(tr_covmean).all(),
                torch.isfinite(fvd).all(),
            )
            raise ValueError("FVD is not finite")

        return fvd

    def reset(self) -> None:
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()


if __name__ == "__main__":

    fvd = FVD()
    video_true = torch.randn(1, 3, 32, 224, 224)
    video_fake = torch.randn(1, 3, 32, 224, 224)
    fvd.update(video_true, True)
    fvd.update(video_fake, False)
    fvd.update(video_true + 0.5, True)
    fvd.update(video_fake + 0.4, False)
    print(fvd.compute())
