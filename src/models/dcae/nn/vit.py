r"""Vision Transformer (ViT) building blocks.

References:
    | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    (Dosovitskiy et al., 2021)
    | https://arxiv.org/abs/2010.11929

    | Scalable Diffusion Models with Transformers (Peebles et al., 2022)
    | https://arxiv.org/abs/2212.09748
"""

__all__ = [
    "ViTBlock",
    "ViT",
]

import math
from typing import Hashable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .attention import MultiheadSelfAttention, xfa
from .embedding import SineEncoding
from .layers import Patchify, Unpatchify


class ViTBlock(nn.Module):
    r"""Creates a ViT block module.

    Arguments:
        channels: The number of channels :math:`C`.
        mod_features: The number of modulating features :math:`D`.
        ffn_factor: The channel factor in the FFN.
        spatial: The number of spatial dimensinons :math:`N`.
        rope: Whether to use rotary positional embedding (RoPE) or not.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
        kwargs: Keyword arguments passed to :class:`MultiheadSelfAttention`.
    """

    def __init__(
        self,
        channels: int,
        mod_features: int = 0,
        ffn_factor: int = 4,
        spatial: int = 2,
        rope: bool = True,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.checkpointing = checkpointing

        # Ada-LN Zero
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)

        if mod_features > 0:
            self.ada_zero = nn.Sequential(
                nn.Linear(mod_features, mod_features),
                nn.SiLU(),
                nn.Linear(mod_features, 4 * channels),
                Rearrange("... (n C) -> n ... 1 C", n=4),
            )

            self.ada_zero[-2].weight.data.mul_(1e-2)
        else:
            self.ada_zero = nn.Parameter(torch.randn(4, channels))
            self.ada_zero.data.mul_(1e-2)

        # MSA
        self.msa = MultiheadSelfAttention(channels, **kwargs)

        # Rotary PE
        if rope:
            amplitude = 1e2 ** -torch.rand(channels // 2)
            direction = torch.nn.functional.normalize(
                torch.randn(spatial, channels // 2), dim=0
            )

            self.theta = nn.Parameter(amplitude * direction)
        else:
            self.theta = None

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_factor * channels),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            nn.Linear(ffn_factor * channels, channels),
        )

    def _forward(
        self,
        x: Tensor,
        mod: Optional[Tensor] = None,
        coo: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        skip: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tokens :math:`x`, with shape :math:`(*, L, C)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(*, D)`.
            coo: The postition coordinates, with shape :math:`(*, L, N)`.
            mask: The attention mask, with shape :math:`(*, L, L)`.
            skip: A skip connection, with shape :math:`(*, L, C)`.

        Returns:
            The ouput tokens :math:`y`, with shape :math:`(*, L, C)`.
        """

        if self.theta is None:
            theta = None
        else:
            theta = torch.einsum("...ij,jk", coo, self.theta)

        if torch.is_tensor(self.ada_zero):
            a, b, c, d = self.ada_zero
        else:
            a, b, c, d = self.ada_zero(mod)

        y = (a + 1) * self.norm(x) + b
        y = y + self.msa(y, theta, mask)
        y = self.ffn(y)
        y = (x + c * y) * torch.rsqrt(1 + c * c)

        if skip is not None:
            y = (y + d * skip) * torch.rsqrt(1 + d * d)

        return y

    def forward(
        self,
        x: Tensor,
        mod: Optional[Tensor] = None,
        coo: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        skip: Optional[Tensor] = None,
    ) -> Tensor:
        if self.checkpointing:
            return checkpoint(
                self._forward, x, mod, coo, mask, skip, use_reentrant=False
            )
        else:
            return self._forward(x, mod, coo, mask, skip)


class ViT(nn.Module):
    r"""Creates a modulated ViT-like module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        cond_channels: The number of condition channels :math:`C_c`.
        mod_features: The number of modulating features :math:`D`.
        hid_channels: The numbers of hidden token channels.
        hid_blocks: The number of hidden transformer blocks.
        spatial: The number of spatial dimensions :math:`N`.
        patch_size: The patch size.
        unpatch_size: The unpatch size.
        window_size: The local attention window size.
        kwargs: Keyword arguments passed to :class:`ViTBlock`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 0,
        mod_features: int = 0,
        hid_channels: int = 1024,
        hid_blocks: int = 3,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        unpatch_size: Union[int, Sequence[int], None] = None,
        window_size: Union[int, Sequence[int], None] = None,
        **kwargs,
    ):
        super().__init__()

        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial

        if unpatch_size is None:
            unpatch_size = patch_size
        elif isinstance(unpatch_size, int):
            unpatch_size = [unpatch_size] * spatial

        self.patch = Patchify(patch_size, channel_last=True)
        self.unpatch = Unpatchify(unpatch_size, channel_last=True)

        self.in_proj = nn.Linear(
            math.prod(patch_size) * (in_channels + cond_channels), hid_channels
        )
        self.out_proj = nn.Linear(hid_channels, math.prod(patch_size) * out_channels)

        self.positional_embedding = nn.Sequential(
            SineEncoding(hid_channels),
            Rearrange("... N C -> ... (N C)"),
            nn.Linear(spatial * hid_channels, hid_channels),
        )

        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    channels=hid_channels,
                    mod_features=mod_features,
                    spatial=spatial,
                    **kwargs,
                )
                for _ in range(hid_blocks)
            ]
        )

        self.spatial = spatial

        if window_size is None:
            self.window_size = None
        elif isinstance(window_size, int):
            self.window_size = (window_size,) * spatial
        else:
            self.window_size = tuple(window_size)

    @staticmethod
    def coo_and_mask(
        shape: Sequence[int],
        spatial: int,
        window_size: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        r"""Returns the token coordinates and attention mask for a given input shape
        and window size."""

        assert isinstance(shape, Hashable)
        assert isinstance(window_size, Hashable)

        coo = (torch.arange(size, device=device) for size in shape)
        coo = torch.cartesian_prod(*coo)
        coo = torch.reshape(coo, shape=(-1, spatial))

        if window_size is None:
            mask = None
        else:
            delta = torch.abs(coo[:, None] - coo[None, :])
            delta = torch.minimum(delta, delta.new_tensor(shape) - delta)

            mask = torch.all(delta <= coo.new_tensor(window_size) // 2, dim=-1)

            if xfa._has_cpp_library:
                mask = xfa.SparseCS(mask, device=mask.device)._mat

        return coo.to(dtype=dtype), mask

    def forward(
        self,
        x: Tensor,
        mod: Optional[Tensor] = None,
        cond: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, L_1, ..., L_N)`.
            mod: The modulation vector, with shape :math:`(D)` or :math:`(B, D)`.
            cond: The condition tensor, with :math:`(B, C_c, L_1, ..., L_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, L_1, ..., L_N)`.
        """

        if cond is not None:
            x = torch.cat((x, cond), dim=1)

        x = self.patch(x)
        x = self.in_proj(x)

        shape = x.shape[-self.spatial - 1 : -1]
        coo, mask = self.coo_and_mask(
            shape,
            spatial=self.spatial,
            window_size=self.window_size,
            dtype=x.dtype,
            device=x.device,
        )

        x = skip = torch.flatten(x, -self.spatial - 1, -2)
        x = x + self.positional_embedding(coo)

        for block in self.blocks:
            x = block(x, mod, coo=coo, mask=mask, skip=skip)

        x = torch.unflatten(x, sizes=shape, dim=-2)

        x = self.out_proj(x)
        x = self.unpatch(x)

        return x
