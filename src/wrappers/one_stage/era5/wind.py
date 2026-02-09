from typing import Dict, Optional

import torch
from azula.denoise import GaussianDenoiser
from einops import repeat
from einops._torch_specific import allow_ops_in_compiled_graph
from hydra.utils import instantiate
from torch import Tensor, nn

from src.modules.azula_tools.forecast import ForecastBase
from src.modules.azula_tools.timestep_sampler import (
    TimestepSampler,
    UniformTimestepSampler,
)
from src.modules.metrics import CRPSTime, MSETime
from src.utils import (
    get_era5_area_weighting,
    get_era5_channel_weighting,
    get_era5_field_names,
    load_stats,
)
from src.wrappers.base import RollingWrapperBase

allow_ops_in_compiled_graph()


class Wind(RollingWrapperBase):
    """Wrapper that coordinates model, sampling, and training."""

    backbone: nn.Module
    forecast_strategy: ForecastBase
    denoiser: GaussianDenoiser
    timestep_sampler: TimestepSampler

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.area_weights = get_era5_area_weighting()
        self.channel_weights = get_era5_channel_weighting(self.hparams.channel_names)
        self.stats = load_stats(
            self.hparams.stats_path,
            self.hparams.channel_names,
            self.hparams.diff_var_channel_weights,
        )
        for split in ["val"]:
            self._setup_metrics(split)
        backbone = torch.compile(
            instantiate(self.hparams.backbone),
            disable=not self.hparams.compile,
            fullgraph=True,
        )
        self.backbone = instantiate(self.hparams.denoiser)(backbone=backbone)
        self.forecast_strategy = instantiate(self.hparams.forecast_strategy)
        self.timestep_sampler = (
            instantiate(self.hparams.timestep_sampler)
            if hasattr(self.hparams, "timestep_sampler")
            else UniformTimestepSampler()
        )

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
        self.area_weights = self.area_weights.to(self.device)
        self.channel_weights = self.channel_weights.to(self.device)
        if self.hparams.diff_var_channel_weights is not None:
            self.diff_var_channel_weights = (
                (self.stats["std"] / self.stats["diff_std"]) ** (2.0)
            ).to(self.device)
            self.diff_var_channel_weights /= self.diff_var_channel_weights.mean()

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> Tensor:
        return self.backbone(x_t, t, **kwargs)

    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        loss = self.model_step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["target_fields"].size(0),
        )
        return loss

    def model_step(self, batch: Dict[str, Tensor]) -> tuple[Dict[str, Tensor], Tensor]:
        x = batch["target_fields"]
        mask_t = getattr(batch["model_kwargs"], "mask_t", None)
        noise_dim = self.hparams.independent_noise_dim

        t = self.timestep_sampler.sample(x, independent_noise_dim=noise_dim)
        if t.ndim == 1:
            t = repeat(t, "b -> b t", t=x.shape[1]).clone()  # clone is important here
        t = t * mask_t if mask_t is not None else t
        if "n_cond_timesteps" in self.hparams:
            t[:, : self.hparams.n_cond_timesteps] = 0

        loss = self.backbone.loss(x, t, **batch["model_kwargs"])
        if self.hparams.use_area_weights:
            loss = loss * self.area_weights
        if self.hparams.use_channel_weights:
            loss = loss * self.channel_weights.view(
                1, 1, self.channel_weights.shape[0], 1, 1
            )
        if self.hparams.diff_var_channel_weights is not None:
            loss = loss * self.diff_var_channel_weights
        loss = loss.mean()
        return {"loss": loss}

    def _setup_metrics(self, split: str):
        n_spatial_dims = self.hparams.n_spatial_dims
        setattr(
            self,
            f"{split}_crps_time",
            CRPSTime(n_spatial_dims=n_spatial_dims, weighting=self.area_weights),
        )
        setattr(
            self,
            f"{split}_mse_time",
            MSETime(n_spatial_dims=n_spatial_dims, weighting=self.area_weights),
        )

    def _reset_metrics(self, split):
        getattr(self, f"{split}_crps_time").reset()
        getattr(self, f"{split}_mse_time").reset()

    def forecast(
        self,
        x: Tensor,
        model_kwargs: Dict[str, Tensor],
        forecast_timesteps: int,
    ) -> tuple[Tensor | None]:
        all_preds = []
        for _ in range(self.hparams.n_ensembles):
            preds, _ = self.forecast_strategy.sample(
                model=self,
                x_conds=x,
                forecast_timesteps=forecast_timesteps,
                **{
                    "independent_noise_dim": self.hparams.independent_noise_dim,
                    "additional_inputs": model_kwargs["additional_inputs"],
                },
            )
            all_preds.append(preds)
        preds = torch.stack(all_preds, dim=1)
        return preds

    # Override from base because we have model_kwargs
    def on_step(self, split: str, batch: Dict[str, Tensor]):
        forecast_timesteps = self.hparams.n_forecast_timesteps
        targets = batch["target_fields"][:, : self.hparams.n_forecast_timesteps].to(
            self.device
        )
        preds = self.forecast(
            x=batch["target_fields"],
            model_kwargs=batch["model_kwargs"],
            forecast_timesteps=forecast_timesteps,
        ).to(self.device)
        preds = self.trainer.val_dataloaders.dataset.postprocess(preds)[
            :, :, : self.hparams.n_forecast_timesteps
        ]
        targets = self.trainer.val_dataloaders.dataset.postprocess(targets)

        getattr(self, f"{split}_crps_time").update(preds, targets)
        getattr(self, f"{split}_mse_time").update(preds.mean(dim=1), targets)

    def on_epoch_end(self, split: str):
        crps_time = getattr(self, f"{split}_crps_time").compute()
        mse_time = getattr(self, f"{split}_mse_time").compute()

        all_field_names = get_era5_field_names(self.hparams.channel_names)

        for i, field_name in enumerate(all_field_names):
            if field_name in self.hparams.val_field_names:
                self.log(
                    f"{split}/crps_{field_name}",
                    crps_time[:, i].mean(),
                    prog_bar=True,
                    sync_dist=True,
                )
                self.log(
                    f"{split}/mse_{field_name}",
                    mse_time[:, i].mean(),
                    prog_bar=True,
                    sync_dist=True,
                )
        self._reset_metrics(split)
