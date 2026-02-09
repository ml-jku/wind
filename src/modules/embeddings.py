from typing import List, Union

import einops
import numpy as np
import torch
from torch import nn


# Copied and adapted from
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: List[int]):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or
        [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# Copied and adapted from https://github.com/BenediktAlkin/KappaModules/blob/main/kappamodules/functional/pos_embed.py  # noqa: E501
def get_sincos_1d_from_seqlen(seqlen: int, dim: int, max_wavelength: int = 10000):
    grid = torch.arange(seqlen, dtype=torch.double)
    # Scale with 200.0
    if seqlen != 1:
        grid *= 200.0 / grid.max()
    return get_sincos_1d_from_grid(grid=grid, dim=dim, max_wavelength=max_wavelength)


# Copied and adapted from https://github.com/BenediktAlkin/KappaModules/blob/main/kappamodules/functional/pos_embed.py  # noqa: E501
def get_sincos_1d_from_grid(grid, dim: int, max_wavelength: int = 10000):
    if dim % 2 == 0:
        padding = None
    else:
        padding = torch.zeros(*grid.shape, 1)
        dim -= 1
    # generate frequencies for sin/cos (e.g. dim=8 -> omega = [1.0, 0.1, 0.01, 0.001])
    omega = 1.0 / max_wavelength ** (torch.arange(0, dim, 2, dtype=torch.double) / dim)
    # create grid of frequencies with timesteps
    # Example seqlen=5 dim=8
    # [0, 0.0, 0.00, 0.000]
    # [1, 0.1, 0.01, 0.001]
    # [2, 0.2, 0.02, 0.002]
    # [3, 0.3, 0.03, 0.003]
    # [4, 0.4, 0.04, 0.004]
    # Note: supports cases where grid is more than 1d
    out = grid.unsqueeze(-1) @ omega.unsqueeze(0)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.concat([emb_sin, emb_cos], dim=-1).float()
    if padding is None:
        return emb
    else:
        return torch.concat([emb, padding], dim=-1)


# Copied and adapted from https://github.com/BenediktAlkin/KappaModules/blob/main/kappamodules/layers/continuous_sincos_embed.py  # noqa: E501
class ContinuousSincosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        ndim: int,
        pos_scale: float = 200.0,
        box_size: Union[float, List[float]] = None,
        max_wavelength: int = 10000,
        dtype=torch.float32,
    ):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        if box_size is not None:
            self.register_buffer("pos_scale", 200.0 / torch.tensor(box_size))
        else:
            self.pos_scale = pos_scale
        # if dim is not cleanly divisible -> cut away trailing dimensions
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = max_wavelength
        self.padding = self.ndim_padding + self.sincos_padding * ndim
        effective_dim_per_wave = (self.dim - self.padding) // ndim
        assert effective_dim_per_wave > 0
        self.register_buffer(
            "omega",
            1.0
            / max_wavelength
            ** (
                torch.arange(0, effective_dim_per_wave, 2, dtype=dtype)
                / effective_dim_per_wave
            ),
        )

    def forward(self, coords, x=None):
        out_dtype = coords.dtype
        ndim = coords.shape[-1]
        assert self.ndim == ndim
        coords = coords * self.pos_scale
        out = coords.unsqueeze(-1).to(self.omega.dtype) @ self.omega.unsqueeze(0)
        emb = torch.concat([torch.sin(out), torch.cos(out)], dim=-1)
        if coords.ndim == 3:
            emb = einops.rearrange(
                emb, "bs num_points ndim dim -> bs num_points (ndim dim)"
            )
        elif coords.ndim == 2:
            emb = einops.rearrange(emb, "num_points ndim dim -> num_points (ndim dim)")
        else:
            raise NotImplementedError
        emb = emb.to(out_dtype)
        if self.padding > 0:
            padding = torch.zeros(
                *emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype
            )
            emb = torch.concat([emb, padding], dim=-1)
        if x is not None:
            return x + emb
        else:
            return emb

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{type(self).__name__}(dim={self.dim})"

    @classmethod
    def create_pos_embeddings(cls, grid_size, dim_embed):
        sincos_embedder = cls(dim=dim_embed, ndim=2, box_size=grid_size)
        all_pos = torch.cartesian_prod(
            torch.arange(grid_size[0]), torch.arange(grid_size[1])
        ).float()
        return sincos_embedder(all_pos)
