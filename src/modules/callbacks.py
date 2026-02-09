import os
from collections import defaultdict

import lightning as pl
import torch
from lightning import Callback

import wandb


class ConfigLRScheduler(Callback):
    """Count up every gradient update step rather than every epoch."""

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if len(trainer.lr_scheduler_configs) > 0:
            self.scheduler = trainer.lr_scheduler_configs[0].scheduler
            assert self.scheduler.__class__.__name__ == "LinearWarmupCosineAnnealingLR"
            self.scheduler.set_steps_per_epoch(
                len(trainer.train_dataloader) // trainer.accumulate_grad_batches
            )


class ForecastStatistics(Callback):
    """Forecast statistics callback (e.g. for evaluation)."""

    def __init__(self, n_spatial_dims: int, n_steps: int, **kwargs):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.n_steps = n_steps


class ExtractForecastData(Callback):
    def __init__(self, dirpath="forecast_data"):
        super().__init__()

    def on_test_start(self, trainer, pl_module):
        dl = trainer.datamodule.forecast_dataloader()
        collector = BatchCollector()
        for batch in dl:
            batch = [data.to(pl_module.device) for data in batch]
            predictions = pl_module.forecast_step(batch)
            collector.update(predictions)

        all_predictions = collector.get_concatenated()
        ckpt_dir = wandb.config.ckpt_path.rsplit("/", 2)[0]
        save_dir = f"{ckpt_dir}/forecast_results"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/{wandb.config.ckpt_path.rsplit('/', 1)[-1].replace('ckpt', 'pt')}"  # noqa: E501
        torch.save(all_predictions, save_path)


class BatchCollector:
    def __init__(self):
        self.storage = defaultdict(list)

    def update(self, batch_dict):
        """Update storage with new batch_dict values."""
        for key, value in batch_dict.items():
            self.storage[key].append(value)

    def get_concatenated(self):
        """Concatenate collected tensors along the batch dimension."""
        return {key: torch.cat(values, dim=0) for key, values in self.storage.items()}

    def reset(self):
        """Reset the storage for a new epoch or evaluation."""
        self.storage.clear()


class PeakMemory(Callback):
    """Get the maximum memory used during training."""

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset stats
        if "cuda" in str(pl_module.device):
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer, pl_module):
        # Log the maximum memory consumption
        if "cuda" in str(pl_module.device):
            max_memory_gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            self.log("train/max_memory", max_memory_gb, prog_bar=True, sync_dist=True)
