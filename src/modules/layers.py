from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from .utils import (
    EmbedND,
    Modulation,
    QKNorm,
    attention,
    attention_with_weights,
    default,
    exists,
    gelu,
    modulate,
)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, context=None, mask=None):
        x = self.norm(x)

        if exists(self.norm_context):
            context = self.norm_context(context)
            return self.fn(x, context, mask)

        return self.fn(x)


class Attention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, qk_norm=False
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.norm = QKNorm(dim_head) if qk_norm else lambda q, k, v: (q, k)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 1,
        act: nn.Module = nn.GELU,
        input_dim: int = None,
        output_dim: int = None,
    ):
        super().__init__()
        input_dim = default(input_dim, dim)
        output_dim = default(output_dim, dim)
        layers = [nn.Sequential(nn.Linear(input_dim, dim), act())]

        layers = layers + [
            nn.Sequential(nn.Linear(dim, dim), act()) for _ in range(1, depth)
        ]
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        context_dim=None,
        heads: int = 4,
        dim_head: int = 64,
        act=nn.GELU,
        qk_norm=False,
    ):
        super().__init__()
        self.attn = PreNorm(
            dim,
            Attention(
                query_dim=dim,
                context_dim=context_dim,
                heads=heads,
                dim_head=dim_head,
                qk_norm=qk_norm,
            ),
            context_dim=context_dim,
        )
        self.ff = PreNorm(dim, FeedForward(dim, act=act))

    def forward(self, x, context=None, mask=None):
        x = self.attn(x, context=context, mask=mask) + x
        x = self.ff(x) + x
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, scale=None, qk_norm=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = default(scale, dim_head**-0.5)
        self.heads = heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.norm = QKNorm(dim_head) if qk_norm else lambda q, k, _: (q, k)

    def forward(self, x, mask=None):
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        q, k = self.norm(q, k, v)

        if mask is not None:
            mask = repeat(mask, "b j -> (b h) () j", h=h)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int = 64,
        act=nn.GELU,
        scale=None,
        qk_norm=False,
    ):
        super().__init__()
        self.attn = PreNorm(dim, SelfAttention(dim, heads, dim_head, scale, qk_norm))
        self.ff = PreNorm(dim, FeedForward(dim, act=act))

    def forward(self, x, mask=None):
        x = self.attn(x, mask=mask) + x
        x = self.ff(x) + x
        return x


class SelfAttentionWithPE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor | None = None) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


class ParallelMLPAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        is_causal: bool = False,
        act: Callable = gelu,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.act = act

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.is_causal = is_causal

    def forward(
        self, x: Tensor, pe: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        qkv, mlp = torch.split(
            self.linear1(x), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        attn = attention(q, k, v, pe=pe, is_causal=self.is_causal)
        output = self.linear2(torch.cat((attn, self.act(mlp)), 2))
        return output


class ParallelMLPAttentionWithResidual(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        is_causal: bool = False,
        act: Callable = gelu,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.act = act

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.is_causal = is_causal

    def forward(
        self, x: Tensor, pe: Tensor | None = None, v_in: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        qkv, mlp = torch.split(
            self.linear1(x), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        if v_in is not None:
            v = v + rearrange(v_in, "B L (H D) -> B H L D", H=self.num_heads)
        q, k = self.norm(q, k, v)

        attn = attention(q, k, v, pe=pe, is_causal=self.is_causal)
        output = self.linear2(torch.cat((attn, self.act(mlp)), 2))
        return output, rearrange(v, "B H L D -> B L (H D)")


class ParallelMLPAttentionWithWeights(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size

    def forward(self, x: Tensor, pe: Tensor | None = None) -> Tensor:
        qkv, mlp = torch.split(
            self.linear1(x), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        attn, weights = attention_with_weights(q, k, v, pe=pe)
        output = self.linear2(torch.cat((attn, gelu(mlp)), 2))
        return output, weights


class MultiStepTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        is_causal: bool = False,
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)

        self.spatial_block = ParallelMLPAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.temporal_block = ParallelMLPAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            is_causal=is_causal,
        )

    def forward(
        self,
        x: Tensor,
        pe_spatial: EmbedND,
        pe_temporal: EmbedND,
    ) -> Tensor:
        _, T, L, _ = x.size()

        residual = x
        x = rearrange(x, "B T L D -> (B T) L D", L=L)
        x = self.pre_norm(x)
        x = self.spatial_block(x=x, pe=pe_spatial)
        x = rearrange(x, "(B T) L D -> B T L D", T=T)
        x = residual + x

        residual = x
        x = rearrange(x, "B T L D -> (B L) T D", L=L)
        x = self.pre_norm(x)
        x = self.temporal_block(x=x, pe=pe_temporal)
        x = rearrange(x, "(B L) T D -> B T L D", L=L)
        x = residual + x

        return x


class MultiStepTransformerCondLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        is_causal: bool = False,
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.modulation = Modulation(hidden_size, double=True)

        self.spatial_block = ParallelMLPAttention(
            hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio
        )
        self.temporal_block = ParallelMLPAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            is_causal=is_causal,
        )

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        pe_spatial: EmbedND,
        pe_temporal: EmbedND,
    ) -> Tensor:
        _, T, L, _ = x.size()

        mod1, mod2 = self.modulation(y)
        residual = x
        x = modulate(self.pre_norm(x), mod1.shift, mod1.scale)
        x = rearrange(x, "B T L D -> (B T) L D", L=L)
        x = self.spatial_block(x=x, pe=pe_spatial)
        x = rearrange(x, "(B T) L D -> B T L D", T=T)
        x = residual + mod1.gate * x

        residual = x
        x = modulate(self.pre_norm(x), mod2.shift, mod2.scale)
        x = rearrange(x, "B T L D -> (B L) T D", L=L)
        x = self.temporal_block(x=x, pe=pe_temporal)
        x = rearrange(x, "(B L) T D -> B T L D", L=L)
        x = residual + mod2.gate * x

        return x


class MultiStepDitParallelLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        is_temporal_causal: bool = False,
        act: Callable = gelu,
    ):
        super().__init__()
        self.modulation = Modulation(hidden_size, double=True)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.act = act

        self.spatial_block = ParallelMLPAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            act=self.act,
        )
        self.temporal_block = ParallelMLPAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            is_causal=is_temporal_causal,
            act=self.act,
        )

    def forward(
        self, x: Tensor, y: Tensor, pe_spatial: EmbedND, pe_temporal: EmbedND
    ) -> Tensor:
        _, T, L, _ = x.size()

        mod1, mod2 = self.modulation(y)
        residual = x
        x = modulate(self.pre_norm(x), mod1.shift, mod1.scale)
        x = rearrange(x, "B T L D -> (B T) L D", L=L)
        x = self.spatial_block(x=x, pe=pe_spatial)
        x = rearrange(x, "(B T) L D -> B T L D", T=T)
        x = residual + mod1.gate * x

        residual = x
        x = modulate(self.pre_norm(x), mod2.shift, mod2.scale)
        x = rearrange(x, "B T L D -> (B L) T D", L=L)
        x = self.temporal_block(x=x, pe=pe_temporal)
        x = rearrange(x, "(B L) T D -> B T L D", L=L)
        x = residual + mod2.gate * x

        return x


class MultiStepResFormerDitParallelLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        is_temporal_causal: bool = False,
        act: Callable = gelu,
    ):
        super().__init__()
        self.modulation = Modulation(hidden_size, double=True)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.spatial_block = ParallelMLPAttentionWithResidual(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            act=act,
        )
        self.temporal_block = ParallelMLPAttentionWithResidual(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            is_causal=is_temporal_causal,
            act=act,
        )

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        pe_spatial: EmbedND,
        pe_temporal: EmbedND,
        v_spatial: Tensor | None = None,
        v_temporal: Tensor | None = None,
    ) -> Tensor:
        _, T, L, _ = x.size()

        mod1, mod2 = self.modulation(y)
        x = modulate(self.pre_norm(x), mod1.shift, mod1.scale)
        x = rearrange(x, "B T L D -> (B T) L D", L=L)
        if v_spatial is not None:
            v_spatial = rearrange(v_spatial, "B T L D -> (B T) L D", L=L)
        x, v_spatial = self.spatial_block(x=x, pe=pe_spatial, v_in=v_spatial)
        x = rearrange(x, "(B T) L D -> B T L D", T=T)
        x = mod1.gate * x
        if v_spatial is not None:
            v_spatial = rearrange(v_spatial, "(B T) L D -> B T L D", T=T)

        x = modulate(self.pre_norm(x), mod2.shift, mod2.scale)
        x = rearrange(x, "B T L D -> (B L) T D", L=L)
        if v_temporal is not None:
            v_temporal = rearrange(v_temporal, "B T L D -> (B L) T D", L=L)
        x = self.pre_norm(x)
        x, v_temporal = self.temporal_block(x=x, pe=pe_temporal, v_in=v_temporal)
        x = rearrange(x, "(B L) T D -> B T L D", L=L)
        x = mod2.gate * x
        if v_temporal is not None:
            v_temporal = rearrange(v_temporal, "(B L) T D -> B T L D", L=L)

        return x, v_spatial, v_temporal


class MultiStepTimeCondTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        is_temporal_causal: bool = False,
    ):
        super().__init__()
        self.modulation = Modulation(hidden_size, double=True)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.spatial_block = ParallelMLPAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.temporal_block = ParallelMLPAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            is_causal=is_temporal_causal,
        )

    def forward(
        self, x: Tensor, y: Tensor, pe_spatial: EmbedND, pe_temporal: EmbedND
    ) -> Tensor:
        _, T, L, _ = x.size()

        mod1, mod2 = self.modulation(y)
        residual = x
        x = modulate(self.pre_norm(x), mod1.shift, mod1.scale)
        x = rearrange(x, "B T L D -> (B T) L D", L=L)
        x = self.spatial_block(x=x, pe=pe_spatial)
        x = rearrange(x, "(B T) L D -> B T L D", T=T)
        x = residual + mod1.gate * x

        residual = x
        x = modulate(self.pre_norm(x), mod2.shift, mod2.scale)
        x = rearrange(x, "B T L D -> (B L) T D", L=L)
        x = self.temporal_block(x=x, pe=pe_temporal)
        x = rearrange(x, "(B L) T D -> B T L D", L=L)
        x = residual + mod2.gate * x

        return x
