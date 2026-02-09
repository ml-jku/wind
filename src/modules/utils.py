# Parts based on https://github.com/black-forest-labs/flux

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, einsum, nn


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))  # noqa: E501
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return gelu(x)


class sin(nn.Module):

    def __call__(self, x: Tensor) -> Tensor:
        return torch.sin(x)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec)).chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class ModulationRollingChunked(nn.Module):
    def __init__(self, dim: int, triple: bool = False):
        super().__init__()
        self.is_triple = triple
        self.multiplier = 9 if triple else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, :, :, None].chunk(
            self.multiplier, dim=-1
        )

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_triple else None,
        )


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


class EmbedNDPBC(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        # Time embedding should not be periodic
        emb = [rope(ids[..., 0], self.axes_dim[0], self.theta)]
        for i in range(n_axes)[1:]:
            emb.append(rope_pbc(ids[..., i], self.axes_dim[i], self.theta))
        emb = torch.cat(emb, dim=-3)

        return emb.unsqueeze(1)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pe: Tensor | None = None,
    is_causal: bool = False,
) -> Tensor:
    q, k = apply_rope(q, k, pe) if pe is not None else (q, k)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def attention_with_weights(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pe: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """This function is used to compute the attention weights for the given query, key,
    and value tensors. it is less performant but we can analyze the attention weights.

    q: (B H L D)
    k: (B H L D)
    v: (B H L D)
    pe: (B L D)
    """
    H = q.shape[1]
    q, k = apply_rope(q, k, pe) if pe is not None else (q, k)
    q, k, v = map(lambda t: rearrange(t, "B H L D -> (B H) L D"), (q, k, v))

    sim = einsum("B I D, B J D -> B I J", q, k) * (1 / math.sqrt(q.shape[-1]))
    weights = sim.softmax(dim=-1)
    out = einsum("B I J, B J D -> B I D", weights, v)
    out = rearrange(out, "(B H) L D -> B L (H D)", H=H)
    return out, rearrange(weights, "(B H) ... -> B H ...", H=H)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def rope_pbc(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    # Get the maximum number of different positions
    n_different_pos = pos.max() + 1
    # Calculate the base frequency
    base_freq = 2 * torch.pi / n_different_pos
    scale = torch.arange(0, dim / 2, 1, dtype=torch.float64, device=pos.device)
    omega = (scale + 1) * base_freq
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """Create sinusoidal timestep embeddings.

    :param t: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def flexible_timestep_embedding(t: Tensor, dim: int) -> Tensor:
    """
    Flexible timestep embedding that handles any number of dimensions.

    Args:
        t: Tensor with any shape (e.g., [16, 1, 1, 1] or [16, 10, 1, 1])
        dim: Embedding dimension

    Returns:
        Tensor with original shape but last dimension replaced with embedding dim
        (e.g., [16, 1, 1, 256] or [16, 10, 1, 256])
    """
    original_shape = t.shape
    t_flat = t.flatten()
    t_embed = timestep_embedding(t_flat, dim)
    new_shape = list(original_shape) + [dim]
    return t_embed.view(new_shape)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


def downsample(x: Tensor, downsample_factor: int) -> Tensor:
    b, t, _, h, w = x.shape
    if h % downsample_factor or w % downsample_factor:
        raise ValueError(
            f"downsample_factor={downsample_factor} must divide H={h} and W={w}"
        )
    x = rearrange(x, "b t c h w -> (b t) c h w")
    x = F.avg_pool2d(x, kernel_size=downsample_factor, stride=downsample_factor)
    return rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)


def t_lin(t, n_chunks, n_clean=0, t0=0.0, t1=1.0):
    t = 1 - t
    if t.ndim == 1:
        t = t.unsqueeze(dim=-1)
    ws = repeat(torch.arange(n_chunks, device=t.device), "w -> s w", s=t.shape[0])
    tw_lin = 1 - torch.clamp((ws + t - n_clean) / (n_chunks - n_clean), 0, 1)
    tw_lin = torch.clamp(tw_lin, t0, t1)
    return tw_lin


def t_init(t, n_chunks, n_clean=0, t0=0.0, t1=1.0):
    t = 1 - t
    if t.ndim == 1:
        t = t.unsqueeze(dim=-1)
    a = n_chunks / (n_chunks - n_clean)
    b = -n_clean / (n_chunks - n_clean)
    ws = torch.arange(n_chunks, device=t.device)
    tw_init = 1 - torch.clamp(
        ws / (n_chunks - n_clean) + a * t + b,
        0,
        1,
    )
    tw_init[:, :n_clean] = 1.0
    tw_init = torch.clamp(tw_init, t0, t1)
    return tw_init


def expand_like(a: Tensor, b: Tensor, dim_insert_pos: int = -1) -> Tensor:
    """Function to reshape a to broadcastable dimension of b.

    Args:
      a (Tensor): [batch_dim, ...]
      b (Tensor): [batch_dim, ...]
      dim_insert_pos (int): Position where to insert new dimensions.
                           -1 (default) means at the end,
                           positive values insert at that position.

    Returns:
      Tensor: Reshaped tensor.
    """
    num_new_dims = len(b.size()) - len(a.size())
    if num_new_dims <= 0:
        return a

    # Handle negative indexing properly
    a_ndim = len(a.size())
    if dim_insert_pos < 0:
        dim_insert_pos = a_ndim + dim_insert_pos + 1

    # Create new shape by inserting 1s at the specified position
    new_shape = list(a.shape)
    for _ in range(num_new_dims):
        new_shape.insert(dim_insert_pos, 1)

    a = a.view(*new_shape)
    return a


class CondSamplerWrapper:
    def __init__(self, sampler, denoiser=None):
        self.sampler = sampler
        self.denoiser = denoiser

    def __call__(self, denoiser, y, A):
        if self.denoiser is not None:
            cond_denoiser = self.denoiser(denoiser, y=y, A=A)
            return self.sampler(cond_denoiser)
        else:
            return self.sampler(denoiser, y=y, A=A)


def pad_latlon_asym(
    x: torch.Tensor,
    pad_top: int = 3,
    pad_bottom: int = 4,
    pad_left: int = 8,
    pad_right: int = 8,
) -> torch.Tensor:
    """
    Asymmetrically pad ERA5-style tensors (B, C, H, W)
    with reflect padding in latitude (top/bottom)
    and periodic padding in longitude (left/right).

    Args:
        x: Input tensor of shape (B, C, H, W)
        pad_top: Number of rows to add at the top (north)
        pad_bottom: Number of rows to add at the bottom (south)
        pad_left: Number of columns to add on the left (west)
        pad_right: Number of columns to add on the right (east)
    Returns:
        Padded tensor of shape (B, C, H + pad_top + pad_bottom, W + pad_left + pad_right)
    """
    # Pad latitude with reflection (top/bottom)
    x = F.pad(x, (0, 0, pad_top, pad_bottom), mode="reflect")

    # Pad longitude with periodic wrap (left/right)
    if pad_left > 0:
        left_part = x[..., -pad_left:]
    else:
        left_part = torch.empty_like(x[..., :0])
    if pad_right > 0:
        right_part = x[..., :pad_right]
    else:
        right_part = torch.empty_like(x[..., :0])
    x = torch.cat([left_part, x, right_part], dim=-1)

    return x
