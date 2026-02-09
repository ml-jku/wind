from functools import partial
from typing import Tuple

import torch
from einops import rearrange, repeat
from omegaconf import DictConfig
from timm.layers import trunc_normal_
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.models.efficientvit.models.efficientvit.dc_ae import (
    build_downsample_block,
    build_upsample_block,
)
from src.modules.azula_tools.utils import expand_like
from src.modules.utils import MLPEmbedder, flexible_timestep_embedding
from src.utils.torch_utils import zero_bias

from ..base_backbone import BaseBackbone
from ..dit.dit_base import SinusoidalPositionalEmbedding
from ..modules.embeddings import RotaryEmbedding3D
from .uvit_blocks import (
    AxialRotaryEmbedding,
    EmbedInput,
    ProjectOutput,
    ResBlock,
    ResBlockWithTemporalAttention,
    TransformerBlock,
)


class UViT3DV2(BaseBackbone):
    """
    A U-ViT backbone from the following papers:
    - Simple diffusion: End-to-end diffusion for high resolution images
    (https://arxiv.org/abs/2301.11093)
    - Simpler Diffusion (SiD2): 1.5 FID on ImageNet512 with pixel-space diffusion
    (https://arxiv.org/abs/2410.19324)
    - We more closely follow SiD2's Residual U-ViT, where blockwise skip-connections
    are removed, and only a single skip-connection is used per downsampling operation.
    """

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        additional_inputs: int,
        max_tokens: int,
        use_causal_mask=True,
        remove_noise_cond: bool = False,
        identity_init: bool = False,
        init_weights: bool = False,
        spatial_dims: int = 2,
        drop_bias: bool = False,
    ):
        # ------------------------------- Configuration --------------------------------
        # these configurations closely follow the notation in the SiD2 paper
        channels = cfg.channels
        self.emb_dim = cfg.emb_channels
        patch_size = cfg.patch_size
        block_types = cfg.block_types
        block_dropouts = cfg.block_dropouts
        num_updown_blocks = cfg.num_updown_blocks
        num_mid_blocks = cfg.num_mid_blocks
        num_heads = cfg.num_heads
        self.pos_emb_type = cfg.pos_emb_type
        self.num_levels = len(channels)
        self.resolution_h = x_shape[1]
        self.resolution_w = x_shape[2]
        self.is_transformers = [
            block_type != "ResBlock" and block_type != "ResBlockWithTemporalAttention"
            for block_type in block_types
        ]
        self.use_checkpointing = list(cfg.use_checkpointing)
        self.temporal_length = max_tokens
        self.remove_noise_cond = remove_noise_cond
        self.identity_init = identity_init
        self.spatial_dims = spatial_dims

        # ------------------------------ Initialization --------------------------------

        super().__init__(cfg, x_shape, max_tokens, None, use_causal_mask)

        self.time_in = (
            nn.Identity()
            if self.remove_noise_cond
            else MLPEmbedder(in_dim=self.noise_level_dim, hidden_dim=self.emb_dim)
        )

        # -------------- Initial downsampling and final upsampling layers --------------
        # This enables avoiding high-resolution feature maps and speeds up the network

        self.embed_input = EmbedInput(
            in_channels=x_shape[0] + additional_inputs,
            dim=channels[0],
            patch_size=patch_size,
        )
        self.project_output = ProjectOutput(
            dim=channels[0],
            out_channels=x_shape[0],
            patch_size=patch_size,
        )

        # --------------------------- Positional embeddings ----------------------------
        # We use a 1D learnable positional embedding or RoPE for every level with
        # transformers
        assert self.pos_emb_type in [
            "learned_1d",
            "rope",
        ], f"Positional embedding type {self.pos_emb_type} not supported."

        self.pos_embs = nn.ModuleDict({})
        for i_level, channel in enumerate(channels):
            if not self.is_transformers[i_level]:
                continue
            pos_emb_cls, dim = None, None
            if self.pos_emb_type == "rope":
                pos_emb_cls = (
                    RotaryEmbedding3D
                    if block_types[i_level] == "TransformerBlock"
                    else AxialRotaryEmbedding
                )
                dim = channel // num_heads
            else:
                pos_emb_cls = partial(SinusoidalPositionalEmbedding, learnable=True)
                dim = channel
            level_resolution_h = self.resolution_h // patch_size // (2**i_level)
            level_resolution_w = self.resolution_w // patch_size // (2**i_level)
            self.pos_embs[f"{i_level}"] = pos_emb_cls(
                dim,
                (self.temporal_length, level_resolution_h, level_resolution_w),
            )

        def _rope_kwargs(i_level: int):
            return (
                {"rope": self.pos_embs[f"{i_level}"]}
                if self.pos_emb_type == "rope" and self.is_transformers[i_level]
                else {}
            )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        block_type_to_cls = {
            "ResBlock": partial(ResBlock, emb_dim=self.emb_dim),
            "ResBlockWithTemporalAttention": partial(
                ResBlockWithTemporalAttention,
                emb_dim=self.emb_dim,
                temporal_length=self.temporal_length,
                heads=num_heads,
            ),
            "TransformerBlock": partial(
                TransformerBlock, emb_dim=self.emb_dim, heads=num_heads
            ),
            "AxialTransformerBlock": partial(
                TransformerBlock,
                emb_dim=self.emb_dim,
                heads=num_heads,
                use_axial=True,
                ax1_len=self.temporal_length,
            ),
        }

        # ---------------------------- Down-sampling blocks ----------------------------
        for i_level, (num_blocks, ch, block_type, block_dropout) in enumerate(
            zip(
                num_updown_blocks,
                channels[:-1],
                block_types[:-1],
                block_dropouts[:-1],
            )
        ):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        block_type_to_cls[block_type](
                            ch, dropout=block_dropout, **_rope_kwargs(i_level)
                        )
                        for _ in range(num_blocks)
                    ]
                    + [
                        build_downsample_block(
                            block_type="ConvPixelUnshuffle",
                            in_channels=ch,
                            out_channels=channels[i_level + 1],
                            shortcut="averaging",
                        )
                    ],
                )
            )

        # ------------------------------ Middle blocks ---------------------------------
        self.mid_blocks = nn.ModuleList(
            [
                block_type_to_cls[block_types[-1]](
                    channels[-1],
                    dropout=block_dropouts[-1],
                    **_rope_kwargs(self.num_levels - 1),
                )
                for _ in range(num_mid_blocks)
            ]
        )

        # ---------------------------- Up-sampling blocks ------------------------------
        for _i_level, (num_blocks, ch, block_type, block_dropout) in enumerate(
            zip(
                reversed(num_updown_blocks),
                reversed(channels[:-1]),
                reversed(block_types[:-1]),
                reversed(block_dropouts[:-1]),
            )
        ):
            i_level = self.num_levels - 2 - _i_level
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        build_upsample_block(
                            block_type="ConvPixelShuffle",
                            in_channels=channels[i_level + 1],
                            out_channels=ch,
                            shortcut="duplicating",
                        )
                    ]
                    + [
                        block_type_to_cls[block_type](
                            ch,
                            dropout=block_dropout,
                            **_rope_kwargs(i_level),
                        )
                        for _ in range(num_blocks)
                    ]
                )
            )

        if init_weights:
            self.apply(self._init_weights)

        if self.remove_noise_cond:
            self.learned_noise_cond = nn.Parameter(
                torch.randn(1, 1, self.noise_level_emb_dim),
                requires_grad=True,
            )

        if drop_bias:
            zero_bias(self)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def noise_level_dim(self) -> int:
        return 256

    @property
    def noise_level_emb_dim(self) -> int:
        return self.emb_dim

    @property
    def external_cond_emb_dim(self) -> int:
        return self.emb_dim

    def _rearrange_and_add_pos_emb_if_transformer(
        self, x: Tensor, emb: Tensor, i_level: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Rearrange input tensor to be compatible with transformer blocks, if necessary.
        Args:
            x: Input tensor of shape (B * T, C, H, W).
            emb: Embedding tensor of shape (B * T, C).
            i_level: Index of the current level.
        Returns:
            x and emb of shape (B, T * H * W, C).
        """
        is_transformer = self.is_transformers[i_level]
        if not is_transformer:
            return x, emb
        h, w = x.shape[-2:]
        x = rearrange(x, "(b t) c h w -> b (t h w) c", t=self.temporal_length)
        if self.pos_emb_type == "learned_1d":
            x = self.pos_embs[f"{i_level}"](x)
        emb = repeat(emb, "(b t) c -> b (t h w) c", t=self.temporal_length, h=h, w=w)
        return x, emb

    def _unrearrange_if_transformer(
        self, x: Tensor, i_level: int, h=None, w=None
    ) -> Tensor:
        """
        Rearrange input tensor back to its original shape, if necessary.
        Args:
            x: Input tensor of shape (B, T * H * W, C).
            i_level: Index of the current level.
        Returns:
            x of shape (B, T, C, H, W).
        """
        is_transformer = self.is_transformers[i_level]
        if not is_transformer:
            return x
        x = rearrange(x, "b (t h w) c -> (b t) c h w", t=self.temporal_length, h=h, w=w)
        return x.contiguous()

    @staticmethod
    def _checkpointed_forward(
        module: nn.Module, *args, use_checkpointing: bool = False
    ) -> Tensor:
        if use_checkpointing:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def _run_level_blocks(
        self, x: Tensor, emb: Tensor, i_level: int, is_up: bool = False
    ) -> Tensor:
        """
        Run the blocks (except up/downsampling blocks) for a given level.
        Gradient checkpointing is used optionally, with self.checkpoints[i_level]
        segments.
        """
        use_checkpointing = self.use_checkpointing[i_level]

        blocks = (
            self.mid_blocks
            if i_level == self.num_levels - 1
            else (
                self.up_blocks[self.num_levels - 2 - i_level][1:]
                if is_up
                else self.down_blocks[i_level][:-1]
            )
        )

        for block in blocks:
            x = self._checkpointed_forward(
                block,
                x,
                emb,
                use_checkpointing=use_checkpointing,
            )
        return x.contiguous()

    def y_cond_embedding(self, y: Tensor, mask_y: Tensor = None) -> Tensor:
        """
        Embed the y conditioning.
        """
        emb = self.vector_in(y)
        if mask_y is not None:
            emb = emb * expand_like(mask_y, emb)
        if emb.ndim < 3:
            emb = repeat(emb, "b d -> b t d", t=self.temporal_length)
        return emb

    def t_cond_embedding(self, t: Tensor, mask_t: Tensor = None) -> Tensor:
        """
        Embed the t conditioning.
        """

        if self.remove_noise_cond:
            emb_t = self.learned_noise_cond
        else:
            while t.ndim < 2:
                t = t[..., None]

            emb_t = self.time_in(flexible_timestep_embedding(t, self.noise_level_dim))
            if mask_t is not None:
                emb_t = emb_t * expand_like(mask_t, emb_t)
        return emb_t

    def _run_level(
        self,
        x: Tensor,
        emb: Tensor,
        i_level: int,
        is_up: bool = False,
        h=None,
        w=None,
    ) -> Tensor:
        """
        Run the blocks (except up/downsampling blocks) for a given level, accompanied
        by reshaping operations before and after.
        """
        x, emb = self._rearrange_and_add_pos_emb_if_transformer(x, emb, i_level)
        x = self._run_level_blocks(x, emb, i_level, is_up)
        x = self._unrearrange_if_transformer(x, i_level, h=h, w=w)
        return x

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass of the U-ViT backbone.
        Args:
            x: Input tensor of shape (B, T, C, H, W).
            noise_levels: Noise level tensor of shape (B, T).
            external_cond: External conditioning tensor of shape (B, T, C).
        Returns:
            Output tensor of shape (B, T, C, H, W).
        """
        orig_resolution_h = x.shape[3]
        orig_resolution_w = x.shape[4]
        assert (
            x.shape[1] == self.temporal_length
        ), "Temporal length of U-ViT is set to {self.temporal_length}, but input has "
        f"temporal length {x.shape[1]}."

        emb_t = self.t_cond_embedding(t, None)

        if self.remove_noise_cond:
            emb = repeat(
                emb_t, "1 1 d -> (b t) d", t=self.temporal_length, b=x.shape[0]
            )
        else:
            emb = rearrange(emb_t, "b t d -> (b t) d")

        # Add additional inputs
        additional_inputs = kwargs["additional_inputs"]
        if additional_inputs.ndim == 4:
            additional_inputs = repeat(
                additional_inputs, "b c h w -> b t c h w", t=self.temporal_length
            )
        x = torch.cat([x, additional_inputs], dim=2)

        # -------------- Initial downsampling and final upsampling layers --------------
        x = rearrange(x, "b t c h w -> (b t) c h w")

        # bilinear interpolation of x
        x = torch.nn.functional.interpolate(
            x,
            size=(self.resolution_h, self.resolution_w),
            mode="bilinear",
            align_corners=False,
        )

        x = self.embed_input(x)

        hs_before = []  # hidden states before downsampling
        hs_after = []  # hidden states after downsampling

        # Down-sampling blocks
        for i_level, down_block in enumerate(
            self.down_blocks,
        ):
            x = self._run_level(x, emb, i_level, h=x.shape[2], w=x.shape[3])
            hs_before.append(x)
            x = down_block[-1](x)
            hs_after.append(x)

        # Middle blocks
        x = self._run_level(x, emb, self.num_levels - 1, h=x.shape[2], w=x.shape[3])

        # Up-sampling blocks
        for _i_level, up_block in enumerate(self.up_blocks):
            i_level = self.num_levels - 2 - _i_level
            x = x - hs_after.pop()
            x = up_block[0](x) + hs_before.pop()
            x = self._run_level(x, emb, i_level, is_up=True, h=x.shape[2], w=x.shape[3])

        x = self.project_output(x)
        # bilinear interpolation of x to the original resolution
        x = torch.nn.functional.interpolate(
            x,
            size=(orig_resolution_h, orig_resolution_w),
            mode="bilinear",
            align_corners=False,
        )
        return rearrange(x, "(b t) c h w -> b t c h w", t=self.temporal_length)
