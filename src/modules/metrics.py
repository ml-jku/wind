from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

from src.utils.spectral import rapsd_torch


class MetricBase(Metric):
    def __init__(self, n_spatial_dims: int, **kwargs):
        super().__init__(**kwargs)
        self.n_spatial_dims = n_spatial_dims
        self.add_state("cum_error", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def compute(self) -> Tensor:
        if self.cum_error.numel() == 0:
            return torch.tensor(0.0)
        return self.cum_error / self.total


class RelativeError(Metric):
    """Mean relative error: mean(|pred - target| / max(|target|, eps)).

    Args:
        eps: Small constant to avoid divide-by-zero when target == 0.
        reduction: 'mean' (default) or 'sum'.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, eps: float = 1e-8, reduction: str = "mean", **kwargs):
        super().__init__(**kwargs)
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.eps = float(eps)
        self.reduction = reduction

        self.add_state("cum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and targets must have the same shape")

        preds = preds.to(torch.float32)
        targets = targets.to(torch.float32)

        denom = torch.clamp(targets.abs(), min=self.eps)
        rel = (preds - targets).abs() / denom

        self.cum_error += rel.sum()
        self.total += rel.numel()

    def compute(self) -> Tensor:
        if self.reduction == "sum":
            return self.cum_error
        # mean
        return self.cum_error / torch.clamp(self.total.to(self.cum_error.dtype), min=1)


class PearsonR(MetricBase):
    """
    Pearson Correlation Coefficient.
    Implementation based on
    https://github.com/PolymathicAI/the_well/blob/master/the_well/benchmark/metrics/spatial.py
    """

    def __init__(self, n_spatial_dims: int, eps: float = 1e-7, **kwargs):
        super().__init__(n_spatial_dims=n_spatial_dims, **kwargs)
        self.eps = eps

    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Pearson Correlation Coefficient

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions.

        Returns:
            Pearson correlation coefficient between x and y.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        # For layout (b, t, c, h, w) with n_spatial_dims=2, flatten over (h, w)
        x_flat = torch.flatten(x, start_dim=-n_spatial_dims, end_dim=-1)
        y_flat = torch.flatten(y, start_dim=-n_spatial_dims, end_dim=-1)

        # Calculate means along flattened spatial axis
        x_mean = torch.mean(x_flat, dim=-1, keepdim=True)
        y_mean = torch.mean(y_flat, dim=-1, keepdim=True)

        # Calculate covariance
        covariance = torch.mean((x_flat - x_mean) * (y_flat - y_mean), dim=-1)
        # Calculate standard deviations
        std_x = torch.std(x_flat, dim=-1)
        std_y = torch.std(y_flat, dim=-1)

        # Calculate Pearson correlation coefficient
        correlation = covariance / (std_x * std_y + eps)
        return correlation

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        # Get per-channel errors with shape (b, t, c)
        errors = self.eval(
            x=preds, y=targets, n_spatial_dims=self.n_spatial_dims, eps=self.eps
        )

        # Sum over batch and time dimensions, keep channels: (b, t, c) -> (c,)
        channel_errors = errors.sum(dim=(0, 1))

        # Initialize cum_error if empty
        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(channel_errors)

        self.cum_error += channel_errors
        self.total += preds.shape[0] * preds.shape[1]  # batch_size * time_steps

    def compute(self) -> Tensor:
        return self.cum_error / self.total


class MSE(MetricBase):
    """
    Mean Squared Error.
    Implementation based on
    https://github.com/PolymathicAI/the_well/blob/master/the_well/benchmark/metrics/spatial.py
    """

    def __init__(self, n_spatial_dims: int, **kwargs):
        super().__init__(n_spatial_dims=n_spatial_dims, **kwargs)

    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
        weighting: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions.

        Returns:
            Mean squared error between x and y.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        # For layout (b, t, c, h, w), reduce over last n_spatial_dims: (h, w)
        spatial_dims = tuple(range(-n_spatial_dims, 0))
        if weighting is not None:
            if weighting.device != x.device:
                weighting = weighting.to(x.device)
            mse = torch.mean((x - y) ** 2 * weighting, dim=spatial_dims)
        else:
            mse = torch.mean((x - y) ** 2, dim=spatial_dims)
        return mse

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        # Get per-channel errors with shape (b, t, c)
        errors = self.eval(x=preds, y=targets, n_spatial_dims=self.n_spatial_dims)

        # Sum over batch and time dimensions, keep channels: (b, t, c) -> (c,)
        channel_errors = errors.sum(dim=(0, 1))

        # Initialize cum_error if empty
        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(channel_errors)

        self.cum_error += channel_errors
        self.total += preds.shape[0] * preds.shape[1]  # batch_size * time_steps


class MAE(MetricBase):
    """
    Mean Absolute Error.
    Implementation based on
    https://github.com/PolymathicAI/the_well/blob/master/the_well/benchmark/metrics/spatial.py
    """

    def __init__(self, n_spatial_dims: int, **kwargs):
        super().__init__(n_spatial_dims=n_spatial_dims, **kwargs)

    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
    ) -> torch.Tensor:
        """
        Mean Absolute Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions.

        Returns:
            Mean absolute error between x and y.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        # For layout (b, t, c, h, w), reduce over last n_spatial_dims: (h, w)
        spatial_dims = tuple(range(-n_spatial_dims, 0))
        return torch.mean((x - y).abs(), dim=spatial_dims)

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        # Get per-channel errors with shape (b, t, c)
        errors = self.eval(x=preds, y=targets, n_spatial_dims=self.n_spatial_dims)

        # Sum over batch and time dimensions, keep channels: (b, t, c) -> (c,)
        channel_errors = errors.sum(dim=(0, 1))

        # Initialize cum_error if empty
        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(channel_errors)

        self.cum_error += channel_errors
        self.total += preds.shape[0] * preds.shape[1]  # batch_size * time_steps


class NMSE(MetricBase):
    """
    Normalized Mean Squared Error.
    Implementation based on
    https://github.com/PolymathicAI/the_well/blob/master/the_well/benchmark/metrics/spatial.py
    """

    def __init__(
        self,
        n_spatial_dims: int,
        eps: float = 1e-7,
        norm_mode: str = "norm",
        **kwargs,
    ):
        super().__init__(n_spatial_dims=n_spatial_dims, **kwargs)
        self.eps = eps
        self.norm_mode = norm_mode

    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
        eps: float = 1e-7,
        norm_mode: str = "norm",
    ) -> torch.Tensor:
        """
        Normalized Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            meta: Metadata for the dataset.
            eps: Small value to avoid division by zero. Default is 1e-7.
            norm_mode:
                Mode for computing the normalization factor. Can be 'norm' or 'std'.
                Default is 'norm'.

        Returns:
            Normalized mean squared error between x and y.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        # For layout (b, t, c, h, w), reduce over last n_spatial_dims: (h, w)
        spatial_dims = tuple(range(-n_spatial_dims, 0))
        if norm_mode == "norm":
            norm = torch.mean(y**2, dim=spatial_dims)
        elif norm_mode == "std":
            norm = torch.std(y, dim=spatial_dims) ** 2
        else:
            raise ValueError(f"Invalid norm_mode: {norm_mode}")
        return MSE.eval(x=x, y=y, n_spatial_dims=n_spatial_dims) / (norm + eps)

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        errors = self.eval(
            x=preds,
            y=targets,
            n_spatial_dims=self.n_spatial_dims,
            eps=self.eps,
            norm_mode=self.norm_mode,
        )

        channel_errors = errors.sum(dim=(0, 1))
        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(channel_errors)

        self.cum_error += channel_errors
        self.total += preds.shape[0] * preds.shape[1]  # batch_size * time_steps


class RMSE(MetricBase):
    """
    Root Mean Squared Error.
    Implementation based on
    https://github.com/PolymathicAI/the_well/blob/master/the_well/benchmark/metrics/spatial.py
    """

    def __init__(
        self,
        n_spatial_dims: int,
        **kwargs,
    ):
        super().__init__(n_spatial_dims=n_spatial_dims, **kwargs)

    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
    ) -> torch.Tensor:
        """
        Root Mean Squared Error

        Args:
            x: torch.Tensor | np.ndarray
                Input tensor.
            y: torch.Tensor | np.ndarray
                Target tensor.
            n_spatial_dims: Number of spatial dimensions.

        Returns:
            Root mean squared error between x and y.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        return torch.sqrt(MSE.eval(x=x, y=y, n_spatial_dims=n_spatial_dims))

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        errors = self.eval(x=preds, y=targets, n_spatial_dims=self.n_spatial_dims)

        channel_errors = errors.sum(dim=(0, 1))
        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(channel_errors)

        self.cum_error += channel_errors
        self.total += preds.shape[0] * preds.shape[1]  # batch_size * time_steps


class NRMSE(MetricBase):
    """
    Normalized Root Mean Squared Error.
    Implementation based on
    https://github.com/PolymathicAI/the_well/blob/master/the_well/benchmark/metrics/spatial.py
    """

    def __init__(
        self,
        n_spatial_dims: int,
        eps: float = 1e-7,
        norm_mode: str = "norm",
        **kwargs,
    ):
        super().__init__(n_spatial_dims=n_spatial_dims, **kwargs)
        self.eps = eps
        self.norm_mode = norm_mode

    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
        eps: float = 1e-7,
        norm_mode: str = "norm",
    ) -> torch.Tensor:
        """
        Normalized Root Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions.
            eps: Small value to avoid division by zero. Default is 1e-7.
            norm_mode : Mode for computing the normalization factor.
                Can be 'norm' or 'std'. Default is 'norm'.

        Returns:
            Normalized root mean squared error between x and y.

        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        return torch.sqrt(
            NMSE.eval(
                x=x, y=y, n_spatial_dims=n_spatial_dims, eps=eps, norm_mode=norm_mode
            )
        )

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        # Get per-channel errors with shape (b, t, c)
        errors = self.eval(
            preds, targets, self.n_spatial_dims, self.eps, self.norm_mode
        )

        # Sum over batch and time dimensions, keep channels: (b, t, c) -> (c,)
        channel_errors = errors.sum(dim=(0, 1))

        # Initialize cum_error if empty
        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(channel_errors)

        self.cum_error += channel_errors
        self.total += preds.shape[0] * preds.shape[1]  # batch_size * time_steps


# TODO: Combine with NRMSE class.
class NRMSETime(NRMSE):
    """
    Normalized Root Mean Squared Error for time series over timesteps.
    """

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        errors = self.eval(
            preds, targets, self.n_spatial_dims, self.eps, self.norm_mode
        )

        channel_errors = errors.sum(dim=0)

        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(channel_errors)

        self.cum_error += channel_errors
        self.total += preds.shape[0]  # batch_size

    def plot(
        self,
        val: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        ax: Optional[_AX_TYPE] = None,  # type: ignore
        log_y: bool = False,
        channel_names: Optional[Sequence[str]] = None,
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        val = val if val is not None else self.compute()

        name = self.__class__.__name__

        if isinstance(val, Tensor):
            val = val.detach().cpu().numpy()

        n_channels = val.shape[1]

        fig = go.Figure()

        for channel in range(n_channels):
            fig.add_trace(
                go.Scatter(
                    x=list(range(val.shape[0])),
                    y=val[:, channel],
                    name=(
                        channel_names[channel]
                        if channel_names
                        else f"channel {channel}"
                    ),
                    mode="lines+markers",
                    line=dict(width=2),
                )
            )

        if n_channels > 1:
            mean_val = val.mean(axis=1)
            fig.add_trace(
                go.Scatter(
                    x=list(range(val.shape[0])),
                    y=mean_val,
                    name="mean",
                    mode="lines+markers",
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            title=f"{name} - Time Series",
            xaxis_title="Timestep",
            yaxis_title="Error",
            yaxis_type="log" if log_y else "linear",
            xaxis=dict(
                autorange=True,
            ),
            yaxis=dict(
                autorange=True,
            ),
        )

        return fig


class VMSE(MetricBase):
    """
    Variance Scaled Mean Squared Error.
    Implementation based on
    https://github.com/PolymathicAI/the_well/blob/master/the_well/benchmark/metrics/spatial.py
    """

    def __init__(self, n_spatial_dims: int, **kwargs):
        super().__init__(n_spatial_dims=n_spatial_dims, **kwargs)

    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
    ) -> torch.Tensor:
        """
        Variance Scaled Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions.

        Returns:
            Variance mean squared error between x and y.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        return NMSE.eval(x=x, y=y, n_spatial_dims=n_spatial_dims, norm_mode="std")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        errors = self.eval(x=preds, y=targets, n_spatial_dims=self.n_spatial_dims)
        channel_errors = errors.sum(dim=(0, 1))

        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(channel_errors)

        self.cum_error += channel_errors
        self.total += preds.shape[0] * preds.shape[1]  # batch_size * time_steps


class VRMSE(MetricBase):
    """
    Root Variance Scaled Mean Squared Error.
    Implementation based on
    https://github.com/PolymathicAI/the_well/blob/master/the_well/benchmark/metrics/spatial.py
    """

    def __init__(self, n_spatial_dims: int, eps: float = 1e-7, **kwargs):
        super().__init__(n_spatial_dims=n_spatial_dims, **kwargs)
        self.eps = eps

    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Root Variance Scaled Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions.
            eps: Small value to avoid division by zero. Default is 1e-7.

        Returns:
            Root variance mean squared error between x and y.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        return NRMSE.eval(
            x=x, y=y, n_spatial_dims=n_spatial_dims, eps=eps, norm_mode="std"
        )

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        errors = self.eval(
            x=preds, y=targets, n_spatial_dims=self.n_spatial_dims, eps=self.eps
        )
        channel_errors = errors.sum(dim=(0, 1))

        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(channel_errors)

        self.cum_error += channel_errors
        self.total += preds.shape[0] * preds.shape[1]  # batch_size * time_steps


class VRMSETime(VRMSE):
    """
    Root Variance Scaled Mean Squared Error for time series over timesteps.
    """

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        errors = self.eval(
            preds,
            targets,
            n_spatial_dims=self.n_spatial_dims,
            eps=self.eps,
        )

        channel_errors = errors.sum(dim=0)

        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(channel_errors)

        self.cum_error += channel_errors
        self.total += preds.shape[0]  # batch_size

    def plot(
        self,
        val: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        ax: Optional[_AX_TYPE] = None,  # type: ignore
        log_y: bool = False,
        channel_names: Optional[Sequence[str]] = None,
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        val = val if val is not None else self.compute()

        name = self.__class__.__name__

        if isinstance(val, Tensor):
            val = val.detach().cpu().numpy()

        n_channels = val.shape[1]

        fig = go.Figure()

        for channel in range(n_channels):
            fig.add_trace(
                go.Scatter(
                    x=list(range(val.shape[0])),
                    y=val[:, channel],
                    name=(
                        channel_names[channel]
                        if channel_names
                        else f"channel {channel}"
                    ),
                    mode="lines+markers",
                    line=dict(width=2),
                )
            )

        if n_channels > 1:
            mean_val = val.mean(axis=1)
            fig.add_trace(
                go.Scatter(
                    x=list(range(val.shape[0])),
                    y=mean_val,
                    name="mean",
                    mode="lines+markers",
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            title=f"{name} - Time Series",
            xaxis_title="Timestep",
            yaxis_title="Error",
            yaxis_type="log" if log_y else "linear",
            xaxis=dict(
                autorange=True,
            ),
            yaxis=dict(
                autorange=True,
            ),
        )

        return fig


class LInfinity(MetricBase):
    """
    L-Infinity Norm.
    Implementation based on
    https://github.com/PolymathicAI/the_well/blob/master/the_well/benchmark/metrics/spatial.py
    """

    def __init__(self, n_spatial_dims: int, **kwargs):
        super().__init__(n_spatial_dims=n_spatial_dims, **kwargs)

    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
    ) -> torch.Tensor:
        """
        L-Infinity Norm

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions.

        Returns:
            L-Infinity norm between x and y.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        # For layout (b, t, c, h, w), flatten over last n_spatial_dims: (h, w)
        return torch.max(
            torch.abs(x - y).flatten(start_dim=-n_spatial_dims, end_dim=-1), dim=-1
        ).values

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        errors = self.eval(x=preds, y=targets, n_spatial_dims=self.n_spatial_dims)

        channel_errors = errors.sum(dim=(0, 1))

        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(channel_errors)

        self.cum_error += channel_errors
        self.total += preds.shape[0] * preds.shape[1]  # batch_size * time_steps


class RAPSD(Metric):
    """
    Spectral Error.
    """

    def __init__(self, n_spatial_dims: int, **kwargs):
        super().__init__(**kwargs)

        self.n_spatial_dims = n_spatial_dims
        self.add_state("rapsd_pred", default=[], dist_reduce_fx="cat")
        self.add_state("rapsd_target", default=[], dist_reduce_fx="cat")
        self.add_state("raspd_error", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
    ) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        if n_spatial_dims != 2:
            raise ValueError("RAPSD is only supported for 2D fields")

        raspsd_x, _ = rapsd_torch(x)
        raspsd_y, _ = rapsd_torch(y)

        return torch.abs(raspsd_x - raspsd_y), raspsd_x, raspsd_y

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        errors, raspsd_pred, raspsd_target = self.eval(
            x=preds, y=targets, n_spatial_dims=self.n_spatial_dims
        )

        self.raspd_error.append(errors)
        self.rapsd_pred.append(raspsd_pred)
        self.rapsd_target.append(raspsd_target)

        self.total += preds.shape[0]  # batch_size * time_steps

    def compute(self) -> tuple[Tensor, Tensor, Tensor]:
        agg_raspsd_error = dim_zero_cat(self.raspd_error).sum(dim=(0)) / self.total
        agg_raspsd_pred = dim_zero_cat(self.rapsd_pred).sum(dim=(0)) / self.total
        agg_raspsd_target = dim_zero_cat(self.rapsd_target).sum(dim=(0)) / self.total

        return agg_raspsd_error, agg_raspsd_pred, agg_raspsd_target


class MSETime(MSE):
    """
    Root Variance Scaled Mean Squared Error for time series over timesteps.
    """

    def __init__(self, n_spatial_dims: int, weighting: torch.Tensor = None, **kwargs):
        super().__init__(n_spatial_dims=n_spatial_dims, **kwargs)
        self.weighting = weighting

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        errors = self.eval(
            preds,
            targets,
            n_spatial_dims=self.n_spatial_dims,
            weighting=self.weighting,
        )

        channel_errors = errors.sum(dim=0)

        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(channel_errors)

        self.cum_error += channel_errors
        self.total += preds.shape[0]  # batch_size

    def plot(
        self,
        val: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        ax: Optional[_AX_TYPE] = None,  # type: ignore
        log_y: bool = False,
        channel_names: Optional[Sequence[str]] = None,
    ) -> _PLOT_OUT_TYPE:  # type: ignore
        val = val if val is not None else self.compute()

        name = self.__class__.__name__

        if isinstance(val, Tensor):
            val = val.detach().cpu().numpy()

        n_channels = val.shape[1]

        fig = go.Figure()

        for channel in range(n_channels):
            fig.add_trace(
                go.Scatter(
                    x=list(range(val.shape[0])),
                    y=val[:, channel],
                    name=(
                        channel_names[channel]
                        if channel_names
                        else f"channel {channel}"
                    ),
                    mode="lines+markers",
                    line=dict(width=2),
                )
            )

        if n_channels > 1:
            mean_val = val.mean(axis=1)
            fig.add_trace(
                go.Scatter(
                    x=list(range(val.shape[0])),
                    y=mean_val,
                    name="mean",
                    mode="lines+markers",
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            title=f"{name} - Time Series",
            xaxis_title="Timestep",
            yaxis_title="Error",
            yaxis_type="log" if log_y else "linear",
            xaxis=dict(
                autorange=True,
            ),
            yaxis=dict(
                autorange=True,
            ),
        )

        return fig


class CRPSTimeOptimized(MetricBase):
    """
    Continuous Ranked Probability Score (CRPS).

    Supports both:
    - 'erdm': Unbiased estimator (divides spread by M(M-1)). Matches `CRPSTime`.
    - 'gencast': Biased estimator (divides spread by M^2). Matches GenCast/GraphCast.

    preds:   (b, m, t, c, h, w)   # m = ensemble members
    targets: (b,    t, c, h, w)
    Returns: (b, t, c)  (reduced over spatial dims only)
    """

    def __init__(
        self,
        n_spatial_dims: int,
        weighting: torch.Tensor = None,
        estimator: str = "erdm",  # Options: 'erdm' or 'gencast'
        **kwargs,
    ):
        super().__init__(n_spatial_dims=n_spatial_dims, **kwargs)
        self.weighting = weighting

        if estimator not in ["erdm", "gencast"]:
            raise ValueError(f"estimator must be 'erdm' or 'gencast', got {estimator}")
        self.estimator = estimator

    def eval(
        self,
        preds: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
    ) -> torch.Tensor:
        """
        CRPS reduced over spatial dims only.
        Returns (b, t, c).
        """
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds).float()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float()

        # Expected shapes: preds (b, m, t, c, ...spatial), targets (b, t, c, ...spatial)
        if preds.dim() < 5:
            raise ValueError("preds must be (b, m, t, c, ...spatial)")
        if targets.dim() != preds.dim() - 1:
            raise ValueError(
                "targets must be (b, t, c, ...spatial) with the same spatial dims as preds"
            )

        b, m = preds.shape[0], preds.shape[1]

        # Check for M < 2 only if using unbiased estimator which divides by M-1
        if self.estimator == "erdm" and m < 2:
            raise ValueError(
                "CRPS (unbiased/erdm) requires at least 2 ensemble members"
            )

        spatial_dims = tuple(range(-self.n_spatial_dims, 0))

        # --- Term 1: Mean Absolute Error ---
        # (1/M) * sum |x - y|
        term1 = torch.abs(preds - targets.unsqueeze(1)).mean(dim=1)

        # --- Term 2: Spread (Optimized) ---
        # Uses sorting to avoid O(M^2) loop

        # 1. Sort ensemble members: O(M log M)
        preds_sorted, _ = torch.sort(preds, dim=1)

        # 2. Generate weights for the linear combination
        # The sum of pairwise diffs is exactly: 2 * sum_{k=0}^{m-1} (2k + 1 - m) * x_{(k)}
        # We compute the weighted sum term here.
        k = torch.arange(m, device=preds.device, dtype=preds.dtype)
        weights = (2 * k + 1 - m).view(1, m, 1, 1, *([1] * self.n_spatial_dims))

        weighted_sum = (preds_sorted * weights).sum(dim=1)

        # 3. Apply normalization based on estimator type
        # term2 = (1 / denominator) * sum_{i,j} |x_i - x_j|
        # Since weighted_sum = (1/2) * sum_{i,j} |x_i - x_j|, we divide accordingly.

        if self.estimator == "erdm":
            # Unbiased: Divide by 2 * M * (M - 1)
            # term2 = weighted_sum / (M * (M - 1))
            term2 = weighted_sum / (m * (m - 1))
        else:
            # GenCast (Biased): Divide by 2 * M * M
            # term2 = weighted_sum / (M * M)
            term2 = weighted_sum / (m**2)

        # --- Final CRPS ---
        crps = term1 - term2

        # Apply spatial weighting
        if self.weighting is not None:
            if self.weighting.device != crps.device:
                self.weighting = self.weighting.to(crps.device)
            crps = crps * self.weighting

        # Reduce over spatial dims
        return crps.mean(dim=spatial_dims)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        # Basic shape checks
        if preds.shape[0] != targets.shape[0]:
            raise ValueError("Batch size must match between preds and targets")
        if preds.shape[2:] != targets.shape[1:]:
            raise ValueError(
                "Non-ensemble dims of preds (t,c,spatial) must match targets (t,c,spatial)"
            )

        # (b, t, c)
        crps_btc = self.eval(
            preds=preds, targets=targets, n_spatial_dims=self.n_spatial_dims
        ).sum(dim=0)

        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(crps_btc)

        self.cum_error += crps_btc
        self.total += preds.shape[0]  # batch_size


class CRPSTime(MetricBase):
    """
    Continuous Ranked Probability Score (unbiased).
    preds:   (b, m, t, c, h, w)   # m = ensemble members
    targets: (b,    t, c, h, w)
    Returns: (b, t, c)  (reduced over spatial dims only)

    Per-position weights (e.g., grid-cell areas) can be provided via
    `spatial_weights` with shape matching the spatial dims, e.g. (h, w) or
    broadcastable to them. If not provided, ones are used.
    """

    def __init__(
        self,
        n_spatial_dims: int,
        weighting: torch.Tensor = None,
        **kwargs,
    ):
        super().__init__(n_spatial_dims=n_spatial_dims, **kwargs)
        self.weighting = weighting

    def eval(
        self,
        preds: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
    ) -> torch.Tensor:
        """
        Unbiased CRPS reduced over spatial dims only.
        Returns (b, t, c).
        """
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds).float()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float()

        # Expected shapes: preds (b, m, t, c, ...spatial), targets (b, t, c, ...spatial)
        if preds.dim() < 5:
            raise ValueError("preds must be (b, m, t, c, ...spatial)")
        if targets.dim() != preds.dim() - 1:
            raise ValueError(
                "targets must be (b, t, c, ...spatial) with the same spatial dims as preds"
            )

        b, m = preds.shape[0], preds.shape[1]
        if m < 2:
            raise ValueError("CRPS (unbiased) requires at least 2 ensemble members")

        spatial_dims = tuple(range(-n_spatial_dims, 0))
        spatial_shape = tuple(preds.shape[d] for d in spatial_dims)
        t = preds.shape[2]
        c = preds.shape[3]

        # term1 = (1/M) sum_m |y_hat_m - y|
        abs_err = torch.abs(preds - targets.unsqueeze(1))  # (b, m, t, c, *S)
        term1 = abs_err.mean(dim=1)  # (b, t, c, *S)

        # term2 = 1 / [2 M (M-1)] sum_m sum_n |y_hat_m - y_hat_n|
        # TODO: Currently done with for loops, can be done with a matrix multiplication
        # Consumes lot of memory though
        diffs = torch.zeros(b, t, c, *spatial_shape, device=preds.device)
        for first_sum_idx in range(m):
            for second_sum_idx in range(m):
                diffs += torch.abs(preds[:, first_sum_idx] - preds[:, second_sum_idx])

        term2 = diffs / (2.0 * m * (m - 1))  # (b, t, c, *S)

        crps = term1 - term2  # (b, t, c, *S)
        if self.weighting is not None:
            if self.weighting.device != crps.device:
                self.weighting = self.weighting.to(crps.device)
            crps = crps * self.weighting
        crps = crps.mean(dim=spatial_dims)

        return crps

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        # Basic shape checks
        if preds.shape[0] != targets.shape[0]:
            raise ValueError("Batch size must match between preds and targets")
        if preds.shape[2:] != targets.shape[1:]:
            raise ValueError(
                "Non-ensemble dims of preds (t,c,spatial) must match targets (t,c,spatial)"
            )

        # (b, t, c)
        crps_btc = self.eval(
            preds=preds, targets=targets, n_spatial_dims=self.n_spatial_dims
        ).sum(dim=0)

        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(crps_btc)

        self.cum_error += crps_btc
        self.total += preds.shape[0]  # batch_size


class CRPSGenCastTime(MetricBase):
    """
    Continuous Ranked Probability Score (unbiased).
    preds:   (b, m, t, c, h, w)   # m = ensemble members
    targets: (b,    t, c, h, w)
    Returns: (b, t, c)  (reduced over spatial dims only)

    Per-position weights (e.g., grid-cell areas) can be provided via
    `spatial_weights` with shape matching the spatial dims, e.g. (h, w) or
    broadcastable to them. If not provided, ones are used.
    """

    def __init__(
        self,
        n_spatial_dims: int,
        weighting: torch.Tensor = None,
        **kwargs,
    ):
        super().__init__(n_spatial_dims=n_spatial_dims, **kwargs)
        self.weighting = weighting

    def eval(
        self,
        preds: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
    ) -> torch.Tensor:
        """
        Unbiased CRPS reduced over spatial dims only.
        Returns (b, t, c).
        """
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds).float()
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float()

        # Expected shapes: preds (b, m, t, c, ...spatial), targets (b, t, c, ...spatial)
        if preds.dim() < 5:
            raise ValueError("preds must be (b, m, t, c, ...spatial)")
        if targets.dim() != preds.dim() - 1:
            raise ValueError(
                "targets must be (b, t, c, ...spatial) with the same spatial dims as preds"
            )

        b, m = preds.shape[0], preds.shape[1]
        spatial_dims = tuple(range(-self.n_spatial_dims, 0))

        # --- Term 1: Mean Absolute Error ---
        # (1/M) * sum |x - y|
        term1 = torch.abs(preds - targets.unsqueeze(1)).mean(dim=1)

        # --- Term 2: Spread (Optimized) ---
        # 1/(2M^2) * sum |x_i - x_j|
        # We avoid the O(M^2) double loop by sorting the ensemble.
        # The sum of pairwise diffs is equivalent to a weighted sum of sorted members.

        # 1. Sort ensemble members: O(M log M)
        preds_sorted, _ = torch.sort(preds, dim=1)

        # 2. Generate weights for the linear combination
        # For x_sorted[k] (0-indexed), the weight is (2k + 1 - M)
        # Shape: (1, m, 1, 1, 1, 1) to broadcast
        k = torch.arange(m, device=preds.device, dtype=preds.dtype)
        weights = (2 * k + 1 - m).view(1, m, 1, 1, *([1] * self.n_spatial_dims))

        # 3. Compute Term 2 using the linear combination identity
        # This replaces the double loop entirely
        term2 = (preds_sorted * weights).sum(dim=1) / (m**2)

        # --- Final CRPS ---
        crps = term1 - term2

        # Apply spatial weighting
        if self.weighting is not None:
            crps = crps * self.weighting

        # Reduce over spatial dims
        return crps.mean(dim=spatial_dims)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        # Basic shape checks
        if preds.shape[0] != targets.shape[0]:
            raise ValueError("Batch size must match between preds and targets")
        if preds.shape[2:] != targets.shape[1:]:
            raise ValueError(
                "Non-ensemble dims of preds (t,c,spatial) must match targets (t,c,spatial)"
            )

        # (b, t, c)
        crps_btc = self.eval(
            preds=preds, targets=targets, n_spatial_dims=self.n_spatial_dims
        ).sum(dim=0)

        if self.cum_error.numel() == 0:
            self.cum_error = torch.zeros_like(crps_btc)

        self.cum_error += crps_btc
        self.total += preds.shape[0]  # batch_size


def calc_spread_skill_ratio(
    preds: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    n_spatial_dims: int = 2,
    weighting: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    """
    Calculates Spread-Skill Ratio preserving the time dimension.

    Memory optimized: loops over time steps to prevent large intermediate allocations.
    Returns: (t, c) -- Averaged over Batch and Spatial dims, preserving Time and Channel.
    """
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).float()
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets).float()

    # Shape checks
    if preds.dim() < 5:
        raise ValueError("preds must be (b, m, t, c, ...spatial)")

    b, m, t, c = preds.shape[:4]

    # We will store the Mean Squared Error (for Skill) and Variance (for Spread)
    # for each time step individually.
    skill_parts = []
    spread_parts = []

    spatial_dims = tuple(range(-n_spatial_dims, 0))

    # Loop over Time (T)
    for i in range(t):
        # Slice current time step:
        # preds: (b, m, c, ...spatial)
        # targets: (b, c, ...spatial)
        p_t = preds[:, :, i]
        tgt_t = targets[:, i]

        # 1. Calculate Ensemble Mean for this step
        mean_t = p_t.mean(dim=1)  # (b, c, ...spatial)

        # 2. Skill Component: (target - mean)^2
        sq_err = (tgt_t - mean_t) ** 2

        # 3. Spread Component: Variance over members
        # Re-use mean_t to avoid re-scanning the tensor
        diff = p_t - mean_t.unsqueeze(1)
        var_t = (diff**2).sum(dim=1) / (m - 1)

        # Apply weighting (Assuming weighting is spatial-only, e.g. HxW)
        if weighting is not None:
            sq_err = sq_err * weighting
            var_t = var_t * weighting

        # Reduce over Spatial Dims
        # Shape becomes (b, c)
        step_skill = sq_err.mean(dim=spatial_dims)
        step_spread = var_t.mean(dim=spatial_dims)

        # Reduce over Batch Dim (matching your original code's logic)
        # Shape becomes (c,)
        skill_parts.append(step_skill.mean(dim=0))
        spread_parts.append(step_spread.mean(dim=0))

    # Stack results back into (t, c) tensors
    # Expected Variance and Expected MSE
    avg_skill_mse = torch.stack(skill_parts, dim=0)  # (t, c)
    avg_spread_var = torch.stack(spread_parts, dim=0)  # (t, c)

    # Final Calculation: Sqrt(Mean(Var)) and Sqrt(Mean(MSE))
    skill = torch.sqrt(avg_skill_mse)
    spread = torch.sqrt(avg_spread_var)

    # Apply Correction Factor (M+1)/M
    correction = torch.sqrt(torch.tensor((m + 1) / m, device=preds.device))

    return (spread / skill) * correction


def calc_spread_skill_ratio_erdm(
    preds: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    n_spatial_dims: int = 2,
    weighting: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    """
    Calculates Spread-Skill Ratio preserving the time dimension.

    Memory optimized: loops over time steps to prevent large intermediate allocations.
    Returns: (t, c) -- Averaged over Batch and Spatial dims, preserving Time and Channel.
    """
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).float()
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets).float()

    # Shape checks
    if preds.dim() < 5:
        raise ValueError("preds must be (b, m, t, c, ...spatial)")

    b, m, t, c = preds.shape[:4]

    # We will store the Mean Squared Error (for Skill) and Variance (for Spread)
    # for each time step individually.
    skill_parts = []
    spread_parts = []

    spatial_dims = tuple(range(-n_spatial_dims, 0))

    # Loop over Time (T)
    for i in range(t):
        # Slice current time step:
        # preds: (b, m, c, ...spatial)
        # targets: (b, c, ...spatial)
        p_t = preds[:, :, i]
        tgt_t = targets[:, i]

        # 1. Calculate Ensemble Mean for this step
        mean_t = p_t.mean(dim=1)  # (b, c, ...spatial)

        # 2. Skill Component: (target - mean)^2
        sq_err = (tgt_t - mean_t) ** 2

        # 3. Spread Component: Variance over members
        # Re-use mean_t to avoid re-scanning the tensor
        diff = p_t - mean_t.unsqueeze(1)
        var_t = (diff**2).sum(dim=1) / (m - 1)

        # Apply weighting (Assuming weighting is spatial-only, e.g. HxW)
        if weighting is not None:
            sq_err = sq_err * weighting
            var_t = var_t * weighting

        # Reduce over Spatial Dims
        # Shape becomes (b, c)
        step_skill = sq_err.mean(dim=spatial_dims)
        step_spread = var_t.mean(dim=spatial_dims)

        # Reduce over Batch Dim (matching your original code's logic)
        # Shape becomes (c,)
        skill_parts.append(torch.sqrt(step_skill).mean(dim=0))
        spread_parts.append(torch.sqrt(step_spread).mean(dim=0))

    # Stack results back into (t, c) tensors
    # Expected Variance and Expected MSE
    avg_skill_mse = torch.stack(skill_parts, dim=0)  # (t, c)
    avg_spread_var = torch.stack(spread_parts, dim=0)  # (t, c)

    # Final Calculation: Sqrt(Mean(Var)) and Sqrt(Mean(MSE))
    skill = avg_skill_mse
    spread = avg_spread_var

    # Apply Correction Factor (M+1)/M
    correction = torch.sqrt(torch.tensor((m + 1) / m, device=preds.device))

    return (spread / skill) * correction
