from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from azula.denoise import Denoiser, GaussianDenoiser, GaussianPosterior
from azula.noise import RectifiedSchedule, Schedule
from torch import BoolTensor, Tensor, nn


class FlexGaussianDenoiser(Denoiser):
    """Flexible denoiser wrapper.

    - Trains backbone against different targets: x0, ε, v, score.
    - Always returns x0_hat in `forward`, so samplers work unchanged.
    """

    def __init__(
        self,
        backbone: nn.Module,
        schedule: Schedule,
        target: Literal["data", "noise", "v", "score"] = "data",
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        self.backbone = backbone
        self.schedule = schedule
        self.target = target
        self.reduction = reduction

    def _raw_to_x_hat(
        self, x_t: Tensor, raw: Tensor, alpha_t: Tensor, sigma_t: Tensor
    ) -> Tensor:
        match self.target:
            case "data":
                x0_hat = raw
            case "noise":
                eps_hat = raw
                x0_hat = (x_t - sigma_t * eps_hat) / alpha_t
            case "v":
                v_hat = raw
                den = alpha_t**2 + sigma_t**2
                x0_hat = (alpha_t * x_t - sigma_t * v_hat) / den
            case "score":
                score_hat = raw
                x0_hat = (x_t + sigma_t**2 * score_hat) / alpha_t
            case "u":
                if not isinstance(self.schedule, RectifiedSchedule):
                    raise ValueError("u target requires a RectifiedSchedule")
                d_alpha_t, d_sigma_t = -1, 1
                u_hat = raw
                x0_hat = (
                    1
                    / (alpha_t - d_alpha_t * (sigma_t / d_sigma_t))
                    * (x_t - (sigma_t / d_sigma_t) * u_hat)
                )
            case _:
                raise ValueError(f"Invalid target: {self.target}")
        return x0_hat

    def _expand(self, t: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Expand schedule outputs to match x_t shape."""
        alpha_t, sigma_t = self.schedule(t)
        while alpha_t.ndim < x.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]
        return alpha_t, sigma_t

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> GaussianPosterior:
        """Convert backbone output into x0_hat, regardless of target."""
        alpha_t, sigma_t = self._expand(t, x_t)
        raw = self.backbone(x_t, t, **kwargs)
        x0_hat = self._raw_to_x_hat(x_t=x_t, raw=raw, alpha_t=alpha_t, sigma_t=sigma_t)
        c_var = sigma_t**2 / (alpha_t**2 + sigma_t**2)
        return GaussianPosterior(mean=x0_hat, var=c_var)

    def _raw_target(self, x: Tensor, z: Tensor, t: Tensor, x_t: Tensor) -> Tensor:
        alpha_t, sigma_t = self._expand(t, x)

        match self.target:
            case "data":
                return x
            case "noise":
                return z
            case "v":
                return alpha_t * z - sigma_t * x
            case "score":
                return -(x_t - alpha_t * x) / (sigma_t**2)
            case "u":
                if not isinstance(self.schedule, RectifiedSchedule):
                    raise ValueError("u target requires a RectifiedSchedule")
                return z - x
            case _:
                raise ValueError(f"Invalid target: {self.target}")

    def loss(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        """MSE loss between backbone prediction and regression target."""
        alpha_t, sigma_t = self._expand(t, x)
        z = torch.randn_like(x)
        x_t = alpha_t * x + sigma_t * z

        raw_target = self._raw_target(x=x, z=z, t=t, x_t=x_t)
        raw_pred = self.backbone(x_t, t, **kwargs)
        loss = F.mse_loss(raw_pred, raw_target, reduction=self.reduction)
        return loss


class InpaintDenoiser(Denoiser):
    def __init__(
        self,
        denoiser: GaussianDenoiser,
        y: Tensor = None,
        mask: BoolTensor = None,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.register_buffer(
            "y", torch.as_tensor(y) if y is not None else torch.tensor([])
        )
        self.register_buffer(
            "mask", torch.as_tensor(mask) if mask is not None else torch.tensor([])
        )

    def set_context(self, y: Tensor, mask: BoolTensor):
        """
        Manually update the context.
        Crucial: we move the input to the device of the existing buffer.
        """
        self.y = torch.as_tensor(y).to(self.y.device)
        self.mask = torch.as_tensor(mask).to(self.mask.device)

    @property
    def schedule(self):
        return self.denoiser.schedule

    def forward(
        self, x_t: Tensor, t: Tensor, independent_noise_dim=None, **kwargs
    ) -> Tensor:
        if self.y.numel() == 0:
            raise ValueError(
                "Context 'y' and 'mask' must be set via set_context() before forward."
            )

        x_s = torch.where(self.mask, self.y, x_t)

        if independent_noise_dim is not None and independent_noise_dim <= 1:
            # Note: k-diffusion uses (2,3,4) for 5D tensors (e.g. video)
            # Ensure your mask has enough dimensions for this mean() call
            t_mask = self.mask.float().mean(dim=tuple(range(2, self.mask.ndim))).bool()
            t = torch.where(t_mask, 0, t)

        return self.denoiser(x_s, t, **kwargs)
