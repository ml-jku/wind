from typing import Callable

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor

from .layers import (
    MultiStepDitParallelLayer,
    MultiStepResFormerDitParallelLayer,
    MultiStepTimeCondTransformerLayer,
    MultiStepTransformerCondLayer,
    MultiStepTransformerLayer,
)
from .utils import (
    EmbedND,
    EmbedNDPBC,
    MLPEmbedder,
    Modulation,
    flexible_timestep_embedding,
    gelu,
    modulate,
    timestep_embedding,
)


class MultiStepTransformerV2(nn.Module):
    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int = 2,
        theta: int = 10_000,
        is_causal: bool = False,
        embed_pos: bool = True,
        embed_time: bool = True,
        axes_dim: list[int] = [16, 56, 56],
        initialize_weights: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.embed_pos = embed_pos
        self.embed_time = embed_time

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads

        self.x_in = nn.Linear(in_dim, hidden_size)
        self.mask_to_emb = nn.Embedding(2, hidden_size)

        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                MultiStepTransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=is_causal,
                )
            )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, self.out_dim)

        if initialize_weights:
            self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.weight is not None:
                    nn.init.ones_(module.weight)

        self.apply(_basic_init)

        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        nn.init.normal_(self.mask_to_emb.weight, std=0.02)

    def forward(
        self,
        x: Tensor,
        x_cond_mask: Tensor,
        pos_ids: Tensor,
    ) -> Tensor:
        B, T, L, _ = x.size()
        x = self.x_in(x) + self.mask_to_emb(x_cond_mask)

        pe_embed = self.pe_embedder(pos_ids)
        pe_embed_spatial = rearrange(pe_embed, "b c (t l) ... -> (b t) c l ...", t=T)
        pe_embed_temporal = rearrange(pe_embed, "b c (t l) ... -> (b l) c t ...", l=L)
        for block in self.blocks:
            x = block(x=x, pe_spatial=pe_embed_spatial, pe_temporal=pe_embed_temporal)

        x = self.pre_norm(x)
        x = self.linear(x)
        return x


class MultiStepTransformerCondV2(nn.Module):

    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int = 2,
        theta: int = 10_000,
        is_causal: bool = False,
        embed_pos: bool = True,
        embed_time: bool = True,
        axes_dim: list[int] = [16, 56, 56],
        vec_in_dim: int | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.embed_pos = embed_pos
        self.embed_time = embed_time

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads

        self.x_in = nn.Linear(in_dim, hidden_size)
        self.mask_to_emb = nn.Embedding(2, hidden_size)

        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                MultiStepTransformerCondLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=is_causal,
                )
            )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, self.out_dim)
        self.vector_in = MLPEmbedder(vec_in_dim, hidden_size)
        self.modulation = Modulation(hidden_size, double=False)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.weight is not None:
                    nn.init.ones_(module.weight)

        self.apply(_basic_init)

        nn.init.normal_(self.mask_to_emb.weight, std=0.02)

        nn.init.trunc_normal_(self.vector_in.in_layer.weight, std=0.02)
        nn.init.trunc_normal_(self.vector_in.out_layer.weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.modulation.lin.weight)
            nn.init.zeros_(block.modulation.lin.bias)

        nn.init.zeros_(self.modulation.lin.weight)
        nn.init.zeros_(self.modulation.lin.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x: Tensor,
        x_cond_mask: Tensor,
        pos_ids: Tensor,
        y: Tensor,
    ) -> Tensor:
        _, t, l, _ = x.size()
        x = self.x_in(x) + self.mask_to_emb(x_cond_mask)
        vec = self.vector_in(y)

        pe_embed = self.pe_embedder(pos_ids)
        pe_embed_spatial = rearrange(pe_embed, "b c (t l) ... -> (b t) c l ...", t=t)
        pe_embed_temporal = rearrange(pe_embed, "b c (t l) ... -> (b l) c t ...", l=l)
        for block in self.blocks:
            x = block(
                x=x,
                y=vec,
                pe_spatial=pe_embed_spatial,
                pe_temporal=pe_embed_temporal,
            )

        mod, _ = self.modulation(vec)
        x = modulate(self.pre_norm(x), mod.shift, mod.scale)
        x = self.linear(x)
        return x


class MutliStepDitParallelV2(nn.Module):
    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int = 2,
        theta: int = 10_000,
        is_causal: bool = False,
        embed_pos: bool = True,
        embed_time: bool = True,
        axes_dim: list[int] = [16, 56, 56],
        vec_in_dim: int | None = None,
        act: Callable = gelu,
        use_pbc_pe: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.embed_pos = embed_pos
        self.embed_time = embed_time

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads

        self.x_in = nn.Linear(in_dim, hidden_size)
        if use_pbc_pe:
            self.pe_embedder = EmbedNDPBC(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        else:
            self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                MultiStepDitParallelLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    is_temporal_causal=is_causal,
                    act=act,
                )
            )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, self.out_dim)
        self.vector_in = MLPEmbedder(vec_in_dim, hidden_size)
        self.modulation = Modulation(hidden_size, double=False)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.weight is not None:
                    nn.init.ones_(module.weight)

        self.apply(_basic_init)

        nn.init.trunc_normal_(self.vector_in.in_layer.weight, std=0.02)
        nn.init.trunc_normal_(self.vector_in.out_layer.weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.modulation.lin.weight)
            nn.init.zeros_(block.modulation.lin.bias)

        nn.init.zeros_(self.modulation.lin.weight)
        nn.init.zeros_(self.modulation.lin.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        pos_ids: Tensor,
        y: Tensor | None = None,
    ) -> Tensor:
        _, T, L, _ = x.size()
        x = self.x_in(x)

        vec = self.time_in(flexible_timestep_embedding(t, 256))
        vec = vec + self.vector_in(y) if y is not None else vec

        pe_embed = self.pe_embedder(pos_ids)
        pe_embed_spatial = rearrange(pe_embed, "b c (t l) ... -> (b t) c l ...", t=T)
        pe_embed_temporal = rearrange(pe_embed, "b c (t l) ... -> (b l) c t ...", l=L)
        for block in self.blocks:
            x = block(
                x=x,
                y=vec,
                pe_spatial=pe_embed_spatial,
                pe_temporal=pe_embed_temporal,
            )

        mod, _ = self.modulation(vec)
        x = modulate(self.pre_norm(x), mod.shift, mod.scale)
        x = self.linear(x)
        return x


class MutliStepDitParallelV3(nn.Module):
    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int = 2,
        theta: int = 10_000,
        is_causal: bool = False,
        embed_pos: bool = True,
        embed_time: bool = True,
        axes_dim: list[int] = [16, 56, 56],
        vec_in_dim: int | None = None,
        act: Callable = gelu,
        use_pbc_pe: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.embed_pos = embed_pos
        self.embed_time = embed_time

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads

        self.x_in = nn.Linear(in_dim, hidden_size)
        self.mask_to_emb = nn.Embedding(2, hidden_size)
        if use_pbc_pe:
            self.pe_embedder = EmbedNDPBC(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        else:
            self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                MultiStepDitParallelLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    is_temporal_causal=is_causal,
                    act=act,
                )
            )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, self.out_dim)
        self.vector_in = MLPEmbedder(vec_in_dim, hidden_size)
        self.modulation = Modulation(hidden_size, double=False)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.weight is not None:
                    nn.init.ones_(module.weight)

        self.apply(_basic_init)

        nn.init.trunc_normal_(self.vector_in.in_layer.weight, std=0.02)
        nn.init.trunc_normal_(self.vector_in.out_layer.weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.modulation.lin.weight)
            nn.init.zeros_(block.modulation.lin.bias)

        nn.init.zeros_(self.modulation.lin.weight)
        nn.init.zeros_(self.modulation.lin.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        pos_ids: Tensor,
        y: Tensor | None = None,
        x_cond_mask: Tensor | None = None,
    ) -> Tensor:
        _, T, L, _ = x.size()
        x = self.x_in(x) + self.mask_to_emb(x_cond_mask)

        vec = self.time_in(flexible_timestep_embedding(t, 256))
        vec = vec + self.vector_in(y) if y is not None else vec

        pe_embed = self.pe_embedder(pos_ids)
        pe_embed_spatial = rearrange(pe_embed, "b c (t l) ... -> (b t) c l ...", t=T)
        pe_embed_temporal = rearrange(pe_embed, "b c (t l) ... -> (b l) c t ...", l=L)
        for block in self.blocks:
            x = block(
                x=x,
                y=vec,
                pe_spatial=pe_embed_spatial,
                pe_temporal=pe_embed_temporal,
            )

        mod, _ = self.modulation(vec)
        x = modulate(self.pre_norm(x), mod.shift, mod.scale)
        x = self.linear(x)
        return x


class MutliStepDitParallelV4(nn.Module):
    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int = 2,
        theta: int = 10_000,
        is_causal: bool = False,
        embed_pos: bool = True,
        embed_time: bool = True,
        axes_dim: list[int] = [16, 56, 56],
        vec_in_dim: int | None = None,
        act: Callable = gelu,
        use_pbc_pe: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.embed_pos = embed_pos
        self.embed_time = embed_time

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads

        self.x_in = nn.Linear(in_dim, hidden_size)
        if use_pbc_pe:
            self.pe_embedder = EmbedNDPBC(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        else:
            self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                MultiStepDitParallelLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    is_temporal_causal=is_causal,
                    act=act,
                )
            )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, self.out_dim)
        self.vector_in = MLPEmbedder(vec_in_dim, hidden_size)
        self.modulation = Modulation(hidden_size, double=False)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.weight is not None:
                    nn.init.ones_(module.weight)

        self.apply(_basic_init)

        nn.init.trunc_normal_(self.vector_in.in_layer.weight, std=0.02)
        nn.init.trunc_normal_(self.vector_in.out_layer.weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.modulation.lin.weight)
            nn.init.zeros_(block.modulation.lin.bias)

        nn.init.zeros_(self.modulation.lin.weight)
        nn.init.zeros_(self.modulation.lin.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x: Tensor,
        pos_ids: Tensor,
        y: Tensor | None = None,
    ) -> Tensor:
        _, T, L, _ = x.size()
        x = self.x_in(x)

        vec = self.vector_in(y)

        pe_embed = self.pe_embedder(pos_ids)
        pe_embed_spatial = rearrange(pe_embed, "b c (t l) ... -> (b t) c l ...", t=T)
        pe_embed_temporal = rearrange(pe_embed, "b c (t l) ... -> (b l) c t ...", l=L)
        for block in self.blocks:
            x = block(
                x=x,
                y=vec,
                pe_spatial=pe_embed_spatial,
                pe_temporal=pe_embed_temporal,
            )

        mod, _ = self.modulation(vec)
        x = modulate(self.pre_norm(x), mod.shift, mod.scale)
        x = self.linear(x)
        return x


class MultiStepResFormerDitParallel(nn.Module):
    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int = 2,
        theta: int = 10_000,
        is_causal: bool = False,
        embed_pos: bool = True,
        embed_time: bool = True,
        axes_dim: list[int] = [16, 56, 56],
        vec_in_dim: int | None = None,
        act: Callable = gelu,
        use_pbc_pe: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.embed_pos = embed_pos
        self.embed_time = embed_time
        self.act = act

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads

        self.x_in = nn.Linear(in_dim, hidden_size)
        if use_pbc_pe:
            self.pe_embedder = EmbedNDPBC(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        else:
            self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                MultiStepResFormerDitParallelLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    is_temporal_causal=is_causal,
                    act=self.act,
                )
            )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, self.out_dim)
        self.vector_in = MLPEmbedder(vec_in_dim, hidden_size)
        self.modulation = Modulation(hidden_size, double=False)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.weight is not None:
                    nn.init.ones_(module.weight)

        self.apply(_basic_init)

        nn.init.trunc_normal_(self.vector_in.in_layer.weight, std=0.02)
        nn.init.trunc_normal_(self.vector_in.out_layer.weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.modulation.lin.weight)
            nn.init.zeros_(block.modulation.lin.bias)

        nn.init.zeros_(self.modulation.lin.weight)
        nn.init.zeros_(self.modulation.lin.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        pos_ids: Tensor,
        y: Tensor | None = None,
    ) -> Tensor:
        B, T, L, _ = x.size()
        x = self.x_in(x)

        vec = self.time_in(flexible_timestep_embedding(t, 256))
        vec = vec + self.vector_in(y) if y is not None else vec

        pe_embed = self.pe_embedder(pos_ids)
        pe_embed_spatial = rearrange(pe_embed, "b c (t l) ... -> (b t) c l ...", t=T)
        pe_embed_temporal = rearrange(pe_embed, "b c (t l) ... -> (b l) c t ...", l=L)
        v_spatial, v_temporal = None, None
        for block in self.blocks:
            x, v_spatial, v_temporal = block(
                x=x,
                y=vec,
                pe_spatial=pe_embed_spatial,
                pe_temporal=pe_embed_temporal,
                v_spatial=v_spatial,
                v_temporal=v_temporal,
            )

        mod, _ = self.modulation(vec)
        x = modulate(self.pre_norm(x), mod.shift, mod.scale)
        x = self.linear(x)
        return x


class MutliStepDitParallel(nn.Module):
    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int = 2,
        theta: int = 10_000,
        is_causal: bool = False,
        embed_pos: bool = True,
        embed_time: bool = True,
        axes_dim: list[int] = [16, 56, 56],
        vec_in_dim: int | None = None,
        act: Callable = gelu,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.embed_pos = embed_pos
        self.embed_time = embed_time
        self.act = act

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads

        self.x_in = nn.Linear(in_dim, hidden_size)
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                MultiStepDitParallelLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    is_temporal_causal=is_causal,
                    act=self.act,
                )
            )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, self.out_dim)
        self.vector_in = MLPEmbedder(vec_in_dim, hidden_size)
        self.modulation = Modulation(hidden_size, double=False)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.weight is not None:
                    nn.init.ones_(module.weight)

        self.apply(_basic_init)

        nn.init.trunc_normal_(self.vector_in.in_layer.weight, std=0.02)
        nn.init.trunc_normal_(self.vector_in.out_layer.weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.modulation.lin.weight)
            nn.init.zeros_(block.modulation.lin.bias)

        nn.init.zeros_(self.modulation.lin.weight)
        nn.init.zeros_(self.modulation.lin.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def temporal_rope_embedding(
        self, B: int, T: int, L: int, device: torch.device
    ) -> Tensor:
        if not self.embed_time:
            return None
        ids = torch.arange(T, device=device)
        emb = self.pe_embedder(rearrange(ids, "T -> 1 T 1"))
        return repeat(emb, "1 ... -> (1 B L) ...", B=B, L=L)

    def spatial_rope_embedding(
        self, B: int, T: int, L: int, device: torch.device
    ) -> Tensor:
        if not self.embed_pos:
            return None
        ids = torch.arange(L, device=device)
        emb = self.pe_embedder(rearrange(ids, "L -> 1 L 1"))
        return repeat(emb, "1 ... -> (1 B T) ...", B=B, T=T)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor | None = None,
    ) -> Tensor:
        B, T, L, _ = x.size()
        x = self.x_in(x)

        vec = self.time_in(flexible_timestep_embedding(t, 256))
        vec = vec + self.vector_in(y) if y is not None else vec

        pe_spatial = self.spatial_rope_embedding(B, T, L, x.device)
        pe_temporal = self.temporal_rope_embedding(B, T, L, x.device)
        for block in self.blocks:
            x = block(
                x=x,
                y=vec,
                pe_spatial=pe_spatial,
                pe_temporal=pe_temporal,
            )

        mod, _ = self.modulation(vec)
        x = modulate(self.pre_norm(x), mod.shift, mod.scale)
        x = self.linear(x)
        return x


class MultiStepTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int = 2,
        theta: int = 10_000,
        is_causal: bool = False,
        embed_pos: bool = True,
        embed_time: bool = True,
        reset_parameters: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.embed_pos = embed_pos
        self.embed_time = embed_time

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads

        self.x_in = nn.Linear(in_dim, hidden_size)
        self.mask_to_emb = nn.Embedding(2, hidden_size)
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=[pe_dim])
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                MultiStepTransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=is_causal,
                )
            )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, self.out_dim)

    def temporal_rope_embedding(
        self, B: int, T: int, L: int, device: torch.device
    ) -> Tensor:
        if not self.embed_time:
            return None
        ids = torch.arange(T, device=device)
        emb = self.pe_embedder(rearrange(ids, "T -> 1 T 1"))
        return repeat(emb, "1 ... -> (1 B L) ...", B=B, L=L)

    def spatial_rope_embedding(
        self, B: int, T: int, L: int, device: torch.device
    ) -> Tensor:
        if not self.embed_pos:
            return None
        ids = torch.arange(L, device=device)
        emb = self.pe_embedder(rearrange(ids, "L -> 1 L 1"))
        return repeat(emb, "1 ... -> (1 B T) ...", B=B, T=T)

    def forward(
        self,
        x: Tensor,
        x_cond_mask: Tensor,
    ) -> Tensor:
        B, T, L, _ = x.size()
        x = self.x_in(x) + self.mask_to_emb(x_cond_mask)

        pe_spatial = self.spatial_rope_embedding(B, T, L, x.device)
        pe_temporal = self.temporal_rope_embedding(B, T, L, x.device)
        for block in self.blocks:
            x = block(x=x, pe_spatial=pe_spatial, pe_temporal=pe_temporal)

        x = self.pre_norm(x)
        x = self.linear(x)
        return x


class MultiStepTimeCondTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int = 2,
        theta: int = 10_000,
        is_causal: bool = False,
        embed_pos: bool = True,
        embed_time: bool = True,
        reset_parameters: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = self.in_dim
        self.embed_pos = embed_pos
        self.embed_time = embed_time

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads

        self.x_in = nn.Linear(in_dim, hidden_size)
        self.cond_to_emb = nn.Linear(in_dim, hidden_size)
        self.mask_to_emb = nn.Embedding(2, hidden_size)
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=[pe_dim])
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                MultiStepTimeCondTransformerLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    is_temporal_causal=is_causal,
                )
            )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, self.out_dim)

    def temporal_rope_embedding(
        self, B: int, T: int, L: int, device: torch.device
    ) -> Tensor:
        if not self.embed_time:
            return None
        ids = torch.arange(T, device=device)
        emb = self.pe_embedder(rearrange(ids, "T -> 1 T 1"))
        return repeat(emb, "1 ... -> (1 B L) ...", B=B, L=L)

    def spatial_rope_embedding(
        self, B: int, T: int, L: int, device: torch.device
    ) -> Tensor:
        if not self.embed_pos:
            return None
        ids = torch.arange(L, device=device)
        emb = self.pe_embedder(rearrange(ids, "L -> 1 L 1"))
        return repeat(emb, "1 ... -> (1 B T) ...", B=B, T=T)

    def forward(
        self,
        x: Tensor,
        x_cond: Tensor,
        x_cond_mask: Tensor,
        t: Tensor,
    ) -> Tensor:
        B, T, L, _ = x.size()
        x = self.x_in(x) + self.cond_to_emb(x_cond) + self.mask_to_emb(x_cond_mask)

        t_embed = self.time_in(timestep_embedding(t, 256))
        t_embed = rearrange(t_embed, "(B T) ... -> B T ...", T=T)

        pe_spatial = self.spatial_rope_embedding(B, T, L, x.device)
        pe_temporal = self.temporal_rope_embedding(B, T, L, x.device)
        for block in self.blocks:
            x = block(x=x, y=t_embed, pe_spatial=pe_spatial, pe_temporal=pe_temporal)

        shift, scale = self.adaLN_modulation(t_embed)[:, :, None].chunk(2, dim=-1)
        x = modulate(self.pre_norm(x), shift, scale)
        x = self.linear(x)
        return x
