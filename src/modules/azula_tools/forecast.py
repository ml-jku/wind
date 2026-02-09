from abc import ABC, abstractmethod
from typing import Tuple

import torch
from azula.sample import Sampler
from torch import Tensor, nn
from tqdm import tqdm

from src.modules.azula_tools.custom_denoiser import InpaintDenoiser
from src.modules.azula_tools.distribution import InitialDistribution
from src.modules.azula_tools.utils import context_from_mask
from src.utils.torch_utils import expand_like

from .utils import compose_forecast, compute_n_forecasts, extract_frames


class ForecastBase(ABC):

    def __init__(
        self,
        window_size: int = 5,
        num_pre_ready: int = 1,
        overlap: int = 1,
        forecast_size: int = 20,
        seed: int = 42,
        eps: float = 1e-3,
        initial_distribution: InitialDistribution = None,
    ):
        self.window_size = window_size
        self.num_pre_ready = num_pre_ready
        self.overlap = overlap
        self.forecast_size = forecast_size
        self.seed = seed
        self.eps = eps
        self.initial_distribution = initial_distribution

    @abstractmethod
    def create_wrapped_model(self, model: nn.Module, **kwargs) -> nn.Module:
        pass

    @abstractmethod
    def prepare_forecast(
        self, preds: Tensor, overlap: int, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def solve(
        self,
        model: nn.Module,
        **model_extras,
    ) -> tuple[Tensor, Tensor | None]:
        pass

    def sample(
        self,
        model: nn.Module,
        x_conds: Tensor,
        forecast_timesteps: int,
        **kwargs,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Execute autoregressive sampling with clean separation of concerns.

        Args:
            velocity_model: The velocity prediction model
            x_conds: Conditioning tensor
            **model_extras: Additional model arguments

        Returns:
            Sampled tensor of target length
        """
        # Setup
        wrapped_model = self.create_wrapped_model(model, **kwargs)
        n_forecasts = self._compute_forecast_schedule(forecast_timesteps)
        overlap = self.num_pre_ready

        preds = x_conds[:, : self.window_size].clone()

        # Not exactly perfect, but it's the best we can do for now
        if "additional_inputs" in kwargs:
            all_additional_inputs = kwargs["additional_inputs"].clone()
        else:
            all_additional_inputs = None

        forecast_frames = []
        forecast_sampling_steps = {}
        start_idx = 0
        for forecast_idx in tqdm(range(n_forecasts), desc="Shifted window sampling"):
            context, context_mask = self.prepare_forecast(
                preds=preds,
                overlap=overlap,
                **kwargs,
            )
            if "additional_inputs" in kwargs:
                kwargs["additional_inputs"] = all_additional_inputs[
                    :, start_idx : start_idx + self.window_size
                ]
            else:
                kwargs["additional_inputs"] = None
            preds, sampling_steps = self.solve(
                wrapped_model,
                context=context,
                context_mask=context_mask,
                **kwargs,
            )

            if sampling_steps is not None:
                forecast_sampling_steps[f"forecast_{forecast_idx}"] = sampling_steps

            frames = extract_frames(
                prediction=preds,
                is_first=(forecast_idx == 0),
                overlap=overlap,
            )
            forecast_frames.append(frames.cpu())

            overlap = self.overlap
            preds = torch.roll(preds, overlap - self.window_size, dims=1)

            start_idx += self.window_size - overlap

        return (
            compose_forecast(forecast_frames, forecast_timesteps),
            forecast_sampling_steps,
        )

    def _compute_forecast_schedule(self, forecast_timesteps: int) -> int:
        return compute_n_forecasts(
            window_size=self.window_size,
            n_steps=forecast_timesteps,
            n_cond_steps=self.overlap,
        )


class InpaintForecast(ForecastBase):
    """Forecast strategy that uses Azula via a sampling strategy."""

    def __init__(self, sampler: Sampler, **kwargs):
        super().__init__(**kwargs)
        self.sampler = sampler

    def prepare_forecast(
        self,
        preds: Tensor,
        overlap: int = 1,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        context_mask = torch.zeros_like(preds, dtype=torch.bool)
        context_mask[:, :overlap] = True
        context = context_from_mask(preds, context_mask).clone()
        return context, context_mask

    def create_wrapped_model(self, model: nn.Module, **kwargs) -> nn.Module:
        return model

    def solve(
        self,
        model: nn.Module,
        **kwargs,
    ) -> Tensor:
        context = kwargs["context"]
        mask = expand_like(kwargs["context_mask"], context)
        cond_denoiser = InpaintDenoiser(
            denoiser=model.backbone,
            y=context,
            mask=mask,
        )
        cond_sampler = self.sampler(denoiser=cond_denoiser)
        if self.initial_distribution is not None:
            x_1 = self.initial_distribution.sample(context)
        else:
            x_1 = cond_sampler.init(context.shape, device=context.device)
        out = cond_sampler(x_1, **kwargs)

        if isinstance(out, tuple):
            x_0, steps = out
        else:
            x_0 = out
            steps = None

        x_0 = torch.where(mask, context, x_0)
        return x_0, steps
