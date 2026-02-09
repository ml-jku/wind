import os
from typing import Dict, List, Literal, Tuple

import lightning as L
import numpy as np
import torch
import xarray as xr
from einops import rearrange, repeat
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.utils import RankedLogger
from src.utils.era5 import (
    get_day_progress,
    get_seconds_since_epoch,
    get_year_progress,
    load_stats,
)

log = RankedLogger(__name__, rank_zero_only=True)

CHUNKS_TO_USE = 8


class ERA5Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: Literal["train", "val", "test"],
        n_timesteps: int,
        seed: int = 42,
        fields: List[str] = [],
        transform: Dict[int, str] = {},
        hours: Tuple[int, int] = (0, 12),
        stride: int = 1,
        timespan: Tuple[str, str] = ("1959-01-01", "2022-12-31"),
        forecast_all: bool = False,
        forecast_all_stride: int = None,
        n_forecasts: int = None,
        mean: Tensor = None,
        std: Tensor = None,
        add_day_year_progress: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.n_timesteps = n_timesteps
        self.seed = seed
        self.fields = fields
        self.transform = transform
        self._epoch_len = 100  # TODO
        self.hours = hours
        self.stride = stride
        self.timespan = timespan
        self.forecast_all = forecast_all
        self.forecast_all_stride = forecast_all_stride
        self.n_forecasts = n_forecasts
        self.mean = mean
        self.std = std
        self.add_day_year_progress = add_day_year_progress
        self._ds = None  # <- lazy
        log.info(f"Prepared dataset handle for {self.data_dir}")

        # store config you need to rebuild the view after opening
        self._selected_fields = fields
        self._hours = hours
        self._timespan = timespan
        self._n_timesteps = n_timesteps
        self._stride = stride

        # postpone anything that touches ds until after we open it
        self._build_metadata_once = False
        self._epoch_len = 100  # TODO
        self.n_forecasts = n_forecasts
        self.mean = mean
        self.std = std
        self.start_indices = None
        self.len = None
        self.additional_inputs = None
        self._stacked = None

    # ensure ds is opened in the current process (worker-safe)
    def _ensure_open(self):
        if self._ds is not None:
            return
        ds = xr.open_zarr(self.data_dir)  # lazy/dask
        # fields with time
        time_vars = []
        for f in self._selected_fields:
            if f not in ds:
                log.info(f"[WARN] Field '{f}' not found — skipping.")
                continue
            da = ds[f]
            if "time" not in da.dims or da.sizes.get("time", 0) == 0:
                log.info(f"[SKIP] Field '{f}' has no/empty 'time' — skipping.")
                continue
            time_vars.append(f)
        if not time_vars:
            raise ValueError("No valid time-dependent fields found in dataset.")
        ds = ds.sel(time=slice(self._timespan[0], self._timespan[1]))

        if self._hours != (0, 6, 12, 18):
            mask = xr.zeros_like(ds["time"], dtype=bool)
            for h in self._hours:
                mask = mask | (ds["time"].dt.hour == int(h))
            ds = ds.isel(time=mask)

        # cache
        self._ds = ds
        self.time_vars = time_vars

        # derive indices/length now that time is known
        if self.forecast_all:
            if self.forecast_all_stride is not None:
                self.start_indices = np.arange(
                    0,
                    len(ds.time) - self._n_timesteps * self._stride,
                    self.forecast_all_stride * self._stride,
                )
            else:
                self.start_indices = np.arange(
                    0,
                    len(ds.time) - self._n_timesteps * self._stride,
                    self._n_timesteps * self._stride,
                )
            self.len = len(self.start_indices)
        else:
            if self.n_forecasts is not None:
                self.start_indices = np.linspace(
                    0, len(ds.time) - self._n_timesteps, self.n_forecasts, dtype=int
                )
                self.len = len(self.start_indices)
            else:
                self.start_indices = None
                self.len = len(ds.time) - self._stride * self._n_timesteps + 1

        # chunk and stack AFTER opening
        # if self._n_timesteps > 1:
        #     self._ds = self._ds.chunk({"time": self._n_timesteps * self._stride})
        self._prep_stacked_channels()  # builds self._stacked
        self.additional_inputs = self._additional_inputs()  # small & cached
        # Save for later use
        self.latitude = ds["latitude"].data
        self.longitude = ds["longitude"].data
        log.info(f"Opened dataset in pid={os.getpid()} with {self.len} samples.")

    def __len__(self):
        self._ensure_open()
        return self.len

    def __getitem__(self, idx: int):
        self._ensure_open()
        if self.start_indices is not None:
            idx = self.start_indices[idx]
        block = self._stacked.isel(
            time=slice(idx, idx + self._stride * self._n_timesteps, self._stride)
        ).transpose("time", "channel", "latitude", "longitude", missing_dims="ignore")
        try:
            arr = np.asarray(block.data, dtype=np.float32)  # triggers dask compute
        except Exception as e:
            # surface the *real* index + dask key that failed
            raise RuntimeError(f"[VAL-DS] compute failed at base_idx={int(idx)}") from e
        sample_time = np.asarray(block["time"].values)
        x = torch.from_numpy(arr)
        x = self.preprocess(x)
        if self.add_day_year_progress:
            seconds_since_epoch = get_seconds_since_epoch(block.time)
            year_progress, day_progress = self._get_day_year_progress(
                seconds_since_epoch
            )
            additional_inputs = repeat(
                self.additional_inputs.float(), "c h w -> t c h w", t=self.n_timesteps
            )
            additional_inputs = torch.cat(
                [
                    additional_inputs,
                    torch.sin(year_progress),
                    torch.cos(year_progress),
                    torch.sin(day_progress),
                    torch.cos(day_progress),
                ],
                dim=1,
            )
        else:
            additional_inputs = self.additional_inputs.float()

        return {
            "target_fields": x,
            "model_kwargs": {"additional_inputs": additional_inputs},
            "seconds_since_epoch": torch.as_tensor(seconds_since_epoch),
            "time_s": block["time"].values.astype("datetime64[s]").astype(np.int64),
            "idx": idx,
        }

    def _get_day_year_progress(self, seconds_since_epoch: torch.Tensor):
        year_progress = get_year_progress(seconds_since_epoch)
        year_progress = repeat(
            year_progress,
            "t -> t 1 h w",
            h=self.additional_inputs.shape[1],
            w=self.additional_inputs.shape[2],
        )
        year_progress = torch.as_tensor(year_progress)
        day_progress = get_day_progress(seconds_since_epoch, self.longitude)
        day_progress = repeat(
            day_progress,
            "t w -> t 1 h w",
            h=self.additional_inputs.shape[1],
        )
        day_progress = torch.as_tensor(day_progress)
        return year_progress, day_progress

    def _additional_inputs(self):
        land_sea_mask = torch.as_tensor(self._ds["land_sea_mask"].to_numpy())
        land_sea_mask = rearrange(land_sea_mask, "h w -> 1 h w")
        soil_type = torch.as_tensor(self._ds["soil_type"].to_numpy())
        soil_type = rearrange(soil_type, "h w -> 1 h w")
        # geopotential_at_surface needs to be normalized
        geopotential_at_surface = torch.as_tensor(
            self._ds["geopotential_at_surface"].to_numpy()
        )
        geopotential_at_surface = rearrange(geopotential_at_surface, "h w -> 1 h w")
        geopotential_at_surface = (
            geopotential_at_surface - geopotential_at_surface.mean()
        ) / geopotential_at_surface.std()

        lon = self._ds["longitude"]
        lat = self._ds["latitude"]
        Lon2D, Lat2D = xr.broadcast(lon, lat)

        lonr = np.deg2rad(Lon2D)
        latr = np.deg2rad(Lat2D)

        sinlat = torch.as_tensor(np.sin(latr).to_numpy())
        coslat = torch.as_tensor(np.cos(latr).to_numpy())
        coslat_coslon = coslat * torch.as_tensor(np.cos(lonr).to_numpy())
        coslat_sinlon = coslat * torch.as_tensor(np.sin(lonr).to_numpy())

        # Same additional inputs as in ERDM
        additional_inputs = torch.concat(
            [
                land_sea_mask,
                soil_type,
                geopotential_at_surface,
                rearrange(sinlat, "h w -> 1 h w"),
                rearrange(coslat_coslon, "h w -> 1 h w"),
                rearrange(coslat_sinlon, "h w -> 1 h w"),
            ],
            dim=0,
        )
        additional_inputs = rearrange(additional_inputs, "c w h -> c h w")
        return additional_inputs

    def _prep_stacked_channels(self):
        arrays = []
        all_field_names = []
        for f in self.time_vars:
            da = self._ds[f]

            # (optional) strip nuisance dims you don't want to fan out over
            if "expver" in da.dims:
                da = da.isel(expver=0)

            lev = "level" if "level" in da.dims else None
            if lev is not None:
                # rename the level dim to 'channel' and label channels as f:level
                da = da.rename({lev: "channel"})
                # ensure channel labels are strings unique per var+level
                if "channel" in da.coords:
                    labels = [f"{f}:{v}" for v in da["channel"].values]
                    da = da.assign_coords(channel=("channel", labels))
                else:
                    labels = [f"{f}:{i}" for i in range(da.sizes["channel"])]
                    da = da.assign_coords(channel=("channel", labels))

                for level in da["channel"].values:
                    all_field_names.append(str(level))
            else:
                # make a length-1 'channel' dim with the var name
                da = da.expand_dims(channel=[f])
                all_field_names.append(f)

            # put dims in a consistent order
            da = da.transpose(
                "time", "channel", "latitude", "longitude", missing_dims="ignore"
            )

            arrays.append(da)

        # concatenate all variables along the unified 'channel' axis
        stacked = xr.concat(arrays, dim="channel")  # [time, channel, lat, lon]
        # Use your global constant 8 (aligned with Zarr disk chunks)
        # AND force spatial dimensions to be unchunked (-1)
        self._stacked = stacked.chunk(
            {"time": CHUNKS_TO_USE, "latitude": -1, "longitude": -1}
        )

        self.all_field_names = all_field_names
        self._stacked = stacked  # keep as a lazy Dask-backed array

    def preprocess(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32)
        orig_x_ndim = x.ndim
        if orig_x_ndim == 4:
            x = x.unsqueeze(0)
        # Check for all precipitation fields
        if "total_precipitation_6hr" in self.all_field_names:
            idx_precip = self.all_field_names.index("total_precipitation_6hr")
            x[..., idx_precip, :, :] = torch.log10(x[..., idx_precip, :, :] * 1000 + 1)
        if "total_precipitation_12hr" in self.all_field_names:
            idx_precip = self.all_field_names.index("total_precipitation_12hr")
            x[..., idx_precip, :, :] = torch.log10(x[..., idx_precip, :, :] * 1000 + 1)
        if "total_precipitation_24hr" in self.all_field_names:
            idx_precip = self.all_field_names.index("total_precipitation_24hr")
            x[..., idx_precip, :, :] = torch.log10(x[..., idx_precip, :, :] * 1000 + 1)
        x = (x - self.mean) / self.std
        if orig_x_ndim == 4:
            x = x.squeeze(0)
        return x

    def postprocess(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        orig_x_ndim = x.ndim
        if orig_x_ndim == 4:
            x = x.unsqueeze(0)
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x = x * std + mean
        if "total_precipitation_6hr" in self.all_field_names:
            idx_precip = self.all_field_names.index("total_precipitation_6hr")
            x[..., idx_precip, :, :] = (10 ** x[..., idx_precip, :, :] - 1) / 1000
        if "total_precipitation_12hr" in self.all_field_names:
            idx_precip = self.all_field_names.index("total_precipitation_12hr")
            x[..., idx_precip, :, :] = (10 ** x[..., idx_precip, :, :] - 1) / 1000
        if "total_precipitation_24hr" in self.all_field_names:
            idx_precip = self.all_field_names.index("total_precipitation_24hr")
            x[..., idx_precip, :, :] = (10 ** x[..., idx_precip, :, :] - 1) / 1000
        if orig_x_ndim == 4:
            x = x.squeeze(0)
        return x


class ERA5DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir_local: str,
        data_dir_global: str,
        stats_path: str,
        n_timesteps: int,
        fields: List[str] = [],
        transform: Dict[int, str] = {},
        hours_train: Tuple[int, int] = (0, 6, 12, 18),
        hours_val: Tuple[int, int] = (0, 12),
        stride_train: int = 2,
        stride_val: int = 1,
        batch_size: int = 16,
        num_workers: int = 8,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        drop_last: bool = False,
        shuffle: bool = False,
        seed: int = 42,
        n_forecast_timesteps: int = 30,
        timespan_train: Tuple[str, str] = ("1959-01-01", "2020-12-31"),
        timespan_val: Tuple[str, str] = ("2021-01-01", "2021-12-31"),
        n_forecasts: int = 64,
        forecast_all: bool = False,
        forecast_all_stride: int = None,
        add_day_year_progress: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        # Check if data_dir_local exists
        self.data_dir = (
            data_dir_local
            if data_dir_local is not None and os.path.exists(data_dir_local)
            else data_dir_global
        )
        all_stats = load_stats(self.hparams.stats_path, self.hparams.fields)
        self.mean = all_stats["mean"]
        self.std = all_stats["std"]

    def _create_dataloader(self, mode: Literal["train", "val", "test"]) -> DataLoader:
        if mode == "train":
            n_timesteps = self.hparams.n_timesteps
            hours = self.hparams.hours_train
            stride = self.hparams.stride_train
            shuffle = self.hparams.shuffle
            drop_last = self.hparams.drop_last
            n_forecasts = None
            forecast_all = False
            timespan = self.hparams.timespan_train
            forecast_all_stride = self.hparams.forecast_all_stride
        elif mode in ["val", "test"]:
            n_timesteps = self.hparams.n_forecast_timesteps
            hours = self.hparams.hours_val
            stride = self.hparams.stride_val
            shuffle = False
            drop_last = self.hparams.drop_last
            n_forecasts = self.hparams.n_forecasts
            forecast_all = self.hparams.forecast_all
            timespan = self.hparams.timespan_val
            forecast_all_stride = self.hparams.forecast_all_stride

        dataset = ERA5Dataset(
            data_dir=self.data_dir,
            split=mode,
            n_timesteps=n_timesteps,
            seed=self.hparams.seed,
            fields=self.hparams.fields,
            transform=self.hparams.transform,
            hours=hours,
            stride=stride,
            timespan=timespan,
            n_forecasts=n_forecasts,
            forecast_all=forecast_all,
            forecast_all_stride=forecast_all_stride,
            mean=self.mean,
            std=self.std,
            add_day_year_progress=self.hparams.add_day_year_progress,
        )
        if self.hparams.prefetch_factor is None:
            multiprocessing_context = None
        else:
            multiprocessing_context = "spawn"
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            multiprocessing_context=multiprocessing_context,
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.hparams.test_loader)
