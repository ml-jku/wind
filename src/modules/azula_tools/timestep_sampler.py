from abc import ABC, abstractmethod

import torch
from torch import Tensor


class TimestepSampler(ABC):
    """Protocol for timestep samplers."""

    def sample(self, x: Tensor, independent_noise_dim: int = 0) -> Tensor:
        """Sample from the timestep distribution.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The sampled tensor.
        """
        if independent_noise_dim + 1 > x.ndim:
            raise ValueError(
                f"independent_noise_dim={independent_noise_dim} is too large for "
                f"x.shape={x.shape}"
            )
        dims = x.shape[: independent_noise_dim + 1]
        return self._sample(dims, device=x.device)

    @abstractmethod
    def _sample(self, dims: tuple, device: torch.device) -> Tensor:
        raise NotImplementedError


class UniformTimestepSampler(TimestepSampler):
    """Uniform timestep sampler."""

    def _sample(self, dims, device: torch.device) -> Tensor:
        """Sample from the uniform distribution.

        Args:
            dims (tuple): The dimensions to sample from.
            device (torch.device): The device to sample from.

        Returns:
            Tensor: The sampled tensor.
        """
        return torch.rand(dims, device=device)
