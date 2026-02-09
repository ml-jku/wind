import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

torch.set_float32_matmul_precision("high")

OmegaConf.register_new_resolver("eval", eval)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# This is hacky but the problem is when using hydra with ddp and multirun
# that an argument error happens.
if os.environ.get("LOCAL_RANK", "0") != "0":
    filtered_args = []
    for arg in sys.argv:
        if not arg.startswith("hydra."):
            filtered_args.append(arg)
    sys.argv = filtered_args


from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    load_ckpt_path,
    load_class,
    load_run_config_from_wb,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def setup_model(cfg: DictConfig, cfg_wandb: DictConfig) -> L.LightningModule:
    ckpt_path = load_ckpt_path(
        ckpt_dir=cfg_wandb.callbacks.model_checkpoint.dirpath, last=cfg.ckpt_last
    )
    log.info(f"Loading checkpoint from {ckpt_path}")
    model: L.LightningModule = load_class(
        class_string=cfg_wandb.model._target_
    ).load_from_checkpoint(
        checkpoint_path=ckpt_path,
        map_location="cpu",
    )
    model.load_ema_weights()
    model.freeze()
    model.eval()
    return model


def sample_forecast(traj_sampler, parsed_traj, n_forecasts, start_frame):
    conditioning_timesteps = traj_sampler.model.hparams.cond_idx[1]
    cond_traj = {k: v[:conditioning_timesteps].clone() for k, v in parsed_traj.items()}
    pred = traj_sampler.sample(
        cond_traj, cond_traj["positions"][0], num_rollouts=n_forecasts
    )
    pred = {
        k: (v[:-start_frame] if start_frame > 0 else v).detach().cpu()
        for k, v in pred.items()
    }
    return pred


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the
    behavior during failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """

    cfg_wandb = load_run_config_from_wb(
        entity=cfg.eval.wandb_entity,
        project=cfg.eval.wandb_project,
        run_id=cfg.eval.wandb_run_id,
    )

    if cfg.ckpt_path is None:
        cfg.ckpt_path = load_ckpt_path(
            ckpt_dir=cfg_wandb.callbacks.model_checkpoint.dirpath,
            last=cfg.eval.ckpt_last,
        )

    cfg_wandb.pop("data")

    # Remove callbacks from wandb config to use only eval callbacks
    if "callbacks" in cfg_wandb:
        cfg_wandb.pop("callbacks")
    if "forecast_strategy" in cfg_wandb.model:
        cfg_wandb.model.pop("forecast_strategy")

    cfg = OmegaConf.merge(cfg_wandb, cfg)

    log.info(f"Using checkpoint from {cfg.ckpt_path}")
    log.info(f"Checkpoint from run_id {cfg.eval.wandb_run_id}")

    extras(cfg)

    metric_dict, _ = evaluate(cfg)

    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value


if __name__ == "__main__":
    main()
