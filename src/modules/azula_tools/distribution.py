from abc import ABC, abstractmethod
from math import sqrt

import torch
from torch import Tensor


class InitialDistribution(ABC):
    """Protocol for initial distributions."""

    @abstractmethod
    def sample(self, x: Tensor) -> Tensor:
        """Sample from the initial distribution.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The sampled tensor.
        """
        ...


class GaussianInitialDistribution(InitialDistribution):
    """Gaussian initial distribution."""

    mean: float
    std: float

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """Initialize the Gaussian initial distribution.

        Args:
            mean (float): The mean of the Gaussian distribution.
            std (float): The standard deviation of the Gaussian distribution.
        """
        self.mean = mean
        self.std = std

    def sample(self, x: Tensor) -> Tensor:
        """Sample from the Gaussian initial distribution.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The sampled tensor.
        """
        return torch.randn_like(x) * self.std + self.mean


# https://arxiv.org/abs/2305.10474
class TemporallyCorrelatedGaussianInitialDistribution(InitialDistribution):
    """Temporally correlated Gaussian initial distribution."""

    mean: float
    std: float
    alpha: float
    temporal_dim: int

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        alpha: float = 1.0,
        temporal_dim: int = 1,
    ):
        self.mean = mean
        self.std = std
        self.alpha = alpha
        self.temporal_dim = temporal_dim

    def sample(self, x: Tensor) -> Tensor:
        noise_shared_shape = (
            x.shape[: self.temporal_dim] + x.shape[self.temporal_dim + 1 :]
        )
        noise_shared = torch.randn(noise_shared_shape, device=x.device) * sqrt(
            self.alpha**2 / (1 + self.alpha**2)
        )
        noise_independent = torch.randn_like(x) * sqrt(1 / (1 + self.alpha**2))
        noise = noise_shared.unsqueeze(self.temporal_dim) + noise_independent
        return noise
