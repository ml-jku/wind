r"""Auto-encoder building blocks."""

__all__ = [
    "AutoEncoder",
    "AutoEncoderLoss",
    "get_autoencoder",
]

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.functional import cosine_similarity

from .nn.dcae import DCDecoder, DCEncoder


class AutoEncoder(nn.Module):
    r"""Creates an auto-encoder module.

    Arguments:
        encoder: An encoder module.
        decoder: A decoder module.
        saturation: The type of latent saturation.
        noise: The latent noise's standard deviation.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        saturation: str = "softclip2",
        noise: float = 0.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.saturation = saturation
        self.noise = noise

    def saturate(self, x: Tensor) -> Tensor:
        if self.saturation is None:
            return x
        elif self.saturation == "softclip":
            return x / (1 + abs(x) / 5)
        elif self.saturation == "softclip2":
            return x * torch.rsqrt(1 + torch.square(x / 5))
        elif self.saturation == "tanh":
            return torch.tanh(x / 5) * 5
        elif self.saturation == "arcsinh":
            return torch.arcsinh(x)
        elif self.saturation == "rmsnorm":
            return x * torch.rsqrt(
                torch.mean(torch.square(x), dim=1, keepdim=True) + 1e-5
            )
        else:
            raise ValueError(f"unknown saturation '{self.saturation}'")

    def encode(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z = self.saturate(z)
        return z

    def decode(self, z: Tensor, noisy: bool = True) -> Tensor:
        if noisy and self.noise > 0:
            z = z + self.noise * torch.randn_like(z)

        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        y = self.decoder(z)
        return y, z


class AutoEncoderLoss(nn.Module):
    r"""Creates a weighted auto-encoder loss module."""

    def __init__(
        self,
        losses: Sequence[str] = ["mse"],  # noqa: B006
        weights: Sequence[float] = [1.0],  # noqa: B006
    ):
        super().__init__()

        assert len(losses) == len(weights)

        self.losses = list(losses)
        self.register_buffer("weights", torch.as_tensor(weights))

    def forward(self, autoencoder: AutoEncoder, x: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            x: A clean tensor :math:`x`, with shape :math:`(B, C, ...)`.
            kwargs: Optional keyword arguments.

        Returns:
            The weighted loss.
        """

        y, z = autoencoder(x, **kwargs)

        values = []

        for loss in self.losses:
            if loss == "mse":
                loss_value = (x - y).square().mean()
            elif loss == "mae":
                loss_value = (x - y).abs().mean()
            elif loss == "vrmse":
                x = rearrange(x, "B C ... -> B C (...)")
                y = rearrange(y, "B C ... -> B C (...)")
                loss_value = (x - y).square().mean(dim=2) / (x.var(dim=2) + 1e-2)
                loss_value = torch.sqrt(loss_value).mean()
            elif loss == "similarity":
                f = rearrange(z, "B ... -> B (...)")
                loss_value = cosine_similarity(f[None, :], f[:, None], dim=-1)
                loss_value = loss_value[
                    *torch.triu_indices(
                        *loss_value.shape, offset=1, device=loss_value.device
                    )
                ]
                loss_value = loss_value.mean()
            else:
                raise ValueError(f"unknown loss '{loss}'.")

            values.append(loss_value)

        values = torch.stack(values)

        return torch.vdot(self.weights, values)


def get_autoencoder(
    pix_channels: int,
    lat_channels: int,
    spatial: int = 2,
    # Arch
    arch: Optional[str] = None,
    saturation: str = "softclip2",
    # Asymmetry
    encoder_only: Dict[str, Any] = {},  # noqa: B006
    decoder_only: Dict[str, Any] = {},  # noqa: B006
    # Noise
    latent_noise: float = 0.0,
    # Ignore
    name: Optional[str] = None,
    loss: Optional[DictConfig] = None,
    # Passthrough
    **kwargs,
) -> AutoEncoder:
    r"""Instantiates an auto-encoder."""

    if arch in (None, "dcae"):
        encoder = DCEncoder(
            in_channels=pix_channels,
            out_channels=lat_channels,
            spatial=spatial,
            **encoder_only,
            **kwargs,
        )

        decoder = DCDecoder(
            in_channels=lat_channels,
            out_channels=pix_channels,
            spatial=spatial,
            **decoder_only,
            **kwargs,
        )
    else:
        raise NotImplementedError()

    autoencoder = AutoEncoder(
        encoder,
        decoder,
        saturation=saturation,
        noise=latent_noise,
    )

    return autoencoder
