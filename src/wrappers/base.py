import random
from typing import Any, Dict, Optional, Tuple

import lightning as L
import torch
import torchmetrics
from einops import repeat
from hydra.utils import instantiate
from torch import Tensor

import wandb
from src.modules.metrics import (
    NRMSE,
    RAPSD,
    RMSE,
    VRMSE,
    LInfinity,
    NRMSETime,
    PearsonR,
)
from src.utils.plotting import plot_comparison_traj, plot_raspsd
from src.utils.pylogger import RankedLogger
from src.utils.tensor_utils import tensor_tree_map

log = RankedLogger(__name__, rank_zero_only=True)


class WrapperBase(L.LightningModule):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ema = None
        self.cached_weights = None

    def setup(self, stage: Optional[str] = None):
        if self.hparams.ema is not None:
            self.ema = instantiate(self.hparams.ema)(model=self)

    def load_ema_weights(self):
        # model.state_dict() contains references to model weights rather
        # than copies. Therefore, we need to clone them before calling
        # load_state_dict().
        print("Loading EMA weights")
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.state_dict())
        self.load_state_dict(self.ema.state_dict()["params"])

    def restore_cached_weights(self):
        print("Restoring cached weights")
        if self.cached_weights is not None:
            self.load_state_dict(self.cached_weights)
            self.cached_weights = None

    def on_before_zero_grad(self, *args, **kwargs):
        if self.ema:
            self.ema.update(self)

    def on_train_start(self):
        if self.ema:
            if self.ema.device != self.device:
                self.ema.to(self.device)

    def on_validation_start(self):
        if self.ema:
            if self.ema.device != self.device:
                self.ema.to(self.device)
            if self.cached_weights is None:
                self.load_ema_weights()

    def on_validation_end(self):
        if self.ema:
            self.restore_cached_weights()

    def on_test_start(self):
        if self.ema:
            if self.ema.device != self.device:
                self.ema.to(self.device)
            if self.cached_weights is None:
                self.load_ema_weights()

    def on_test_end(self):
        if self.ema:
            self.restore_cached_weights()

    def on_load_checkpoint(self, checkpoint):
        print(f"Loading EMA state dict from checkpoint {checkpoint['epoch']}")
        if self.hparams.ema is not None:
            self.ema = instantiate(self.hparams.ema)(model=self)
            self.ema.load_state_dict(checkpoint["ema"])

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            if self.cached_weights is not None:
                self.restore_cached_weights()
            checkpoint["ema"] = self.ema.state_dict()

    def configure_optimizers(self) -> Dict[str, Any]:
        self.lr = self.hparams.optimizer.lr
        optimizer = instantiate(self.hparams.optimizer)(
            params=filter(lambda p: p.requires_grad, self.trainer.model.parameters())
        )
        if hasattr(self.hparams, "scheduler") and self.hparams.scheduler is not None:
            scheduler = instantiate(self.hparams.scheduler)(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor,
                    "interval": self.hparams.interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def forward(self, batch: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        return self.backbone(batch, **kwargs)

    def model_step(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        loss, preds = self.loss(model=self, batch=batch)
        return loss, preds

    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        loss = self.model_step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in loss.items()},
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["target_fields"].size(0),
        )
        return loss


class RollingWrapperBase(WrapperBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.val_metrics_scalar = torchmetrics.MetricCollection(
            {
                "nrmse": NRMSE(n_spatial_dims=self.n_spatial_dims),
                "rmse": RMSE(n_spatial_dims=self.n_spatial_dims),
                "vrmse": VRMSE(n_spatial_dims=self.hparams.n_spatial_dims),
                "linfinity": LInfinity(n_spatial_dims=self.n_spatial_dims),
                "pearsonr": PearsonR(n_spatial_dims=self.n_spatial_dims),
            },
            prefix="val/",
        )
        self.test_metrics_scalar = self.val_metrics_scalar.clone(prefix="test/")

        self.val_nrmse_time = NRMSETime(n_spatial_dims=self.n_spatial_dims)
        self.test_nrmse_time = NRMSETime(n_spatial_dims=self.n_spatial_dims)

        self.val_raspsde = RAPSD(n_spatial_dims=self.n_spatial_dims)
        self.test_raspsde = RAPSD(n_spatial_dims=self.n_spatial_dims)

    @property
    def mean(self) -> Dict[str, Tensor]:
        return self.hparams.stats.mean

    @property
    def std(self) -> Dict[str, Tensor]:
        return self.hparams.stats.std

    @property
    def n_spatial_dims(self) -> int:
        return self.hparams.n_spatial_dims

    def on_epoch_start(self, split: str):
        getattr(self, f"{split}_metrics_scalar").reset()
        getattr(self, f"{split}_nrmse_time").reset()
        getattr(self, f"{split}_raspsde").reset()
        setattr(self, f"{split}_forecast_fields", [])
        setattr(self, f"{split}_ground_truth_fields", [])

    def on_step(self, split: str, batch: Dict[str, Tensor]):
        n_forecast_timesteps = self.hparams.n_forecast_timesteps
        target_fields = batch["target_fields"][:, :n_forecast_timesteps].clone()

        preds, _ = self.forecast(
            x=batch["target_fields"],
            y=batch["model_kwargs"]["y"],
            forecast_timesteps=n_forecast_timesteps,
        )

        dataset = getattr(self.trainer, f"{split}_dataloaders").dataset
        target_fields = dataset.postprocess(target_fields, dim=2)
        preds = dataset.postprocess(preds, dim=2)

        getattr(self, f"{split}_metrics_scalar").update(preds, target_fields)
        getattr(self, f"{split}_nrmse_time").update(preds, target_fields)
        getattr(self, f"{split}_raspsde").update(preds, target_fields)
        getattr(self, f"{split}_forecast_fields").append(preds.cpu())
        getattr(self, f"{split}_ground_truth_fields").append(target_fields.cpu())

    def on_epoch_end(self, split: str):
        metrics_scalar = getattr(self, f"{split}_metrics_scalar").compute()
        self.log_dict(metrics_scalar, prog_bar=True, sync_dist=True)

        raspsde, raspsd_pred, raspsd_target = getattr(
            self, f"{split}_raspsde"
        ).compute()
        self.log(f"{split}/raspsde", raspsde.mean(), prog_bar=True, sync_dist=True)

        nrmse_time = getattr(self, f"{split}_nrmse_time").compute()
        self.log(
            f"{split}/nrmse_time_mean",
            nrmse_time.mean(),
            prog_bar=True,
            sync_dist=True,
        )

        if self.trainer.is_global_zero:
            fig = getattr(self, f"{split}_nrmse_time").plot(log_y=True)
            wandb.log({f"{split}/nrmse_time": fig})

            for c in range(raspsd_pred.shape[1]):
                fig = plot_raspsd(
                    raspsd_pred=raspsd_pred,
                    raspsd_target=raspsd_target,
                    channel=c,
                )

                wandb.log(
                    {f"{split}/raspsd/interactive/{c}": fig},
                )

            forecast_fields = getattr(self, f"{split}_forecast_fields")
            ground_truth_fields = getattr(self, f"{split}_ground_truth_fields")

            if len(forecast_fields) > 0:

                forecast_fields = torch.cat(forecast_fields)
                ground_truth_fields = torch.cat(ground_truth_fields)

                n_samples = min(self.hparams.n_traj_forecast_vis, len(forecast_fields))
                sampled_indices = random.sample(
                    range(len(forecast_fields)),
                    n_samples,
                )

                forecast_fields = forecast_fields[sampled_indices]
                ground_truth_fields = ground_truth_fields[sampled_indices]
                error_fields = forecast_fields - ground_truth_fields

                traj_comparison = plot_comparison_traj(
                    forecast=forecast_fields,
                    ground_truth=ground_truth_fields,
                    residual_error=error_fields,
                    cmap="twilight",
                    error_cmap="coolwarm",
                    use_sym_colormap=False,
                )

                wandb.log(
                    {f"{split}/forecast": wandb.Html(traj_comparison._repr_html_())},
                )

    def on_validation_epoch_start(self):
        self.on_epoch_start(split="val")

    def validation_step(self, batch: Dict[str, Tensor]) -> Tensor:
        self.on_step(split="val", batch=batch)

    def on_validation_epoch_end(self):
        self.on_epoch_end(split="val")

    def on_test_epoch_start(self):
        self.on_epoch_start(split="test")

    def test_step(self, batch: Dict[str, Tensor]) -> Tensor:
        self.on_step("test", batch)

    def on_test_epoch_end(self):
        self.on_epoch_end("test")

    def forecast(
        self,
        x: Tensor,
        y: Tensor,
        forecast_timesteps: int,
        num_samples: int = 1,
    ) -> tuple[Tensor | None]:
        x, y = map(
            lambda x: repeat(x, "b ... -> (b n) ...", n=num_samples).contiguous(),
            (x, y),
        )
        return self.forecast_strategy.sample(
            model=self,
            x_conds=x,
            forecast_timesteps=forecast_timesteps,
            **{
                "independent_noise_dim": self.hparams.independent_noise_dim,
                "y": y,
            },
        )
