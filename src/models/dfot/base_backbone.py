from abc import ABC, abstractmethod
from typing import Optional

import torch
from omegaconf import DictConfig
from torch import nn


class BaseBackbone(ABC, nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        use_causal_mask=True,
    ):

        super().__init__()

        self.cfg = cfg
        self.external_cond_dim = external_cond_dim
        self.use_causal_mask = use_causal_mask
        self.x_shape = x_shape

    @property
    def noise_level_dim(self):
        return max(self.noise_level_emb_dim // 4, 32)

    @property
    @abstractmethod
    def noise_level_emb_dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def external_cond_emb_dim(self):
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError
