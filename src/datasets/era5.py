import os
from typing import Dict, List, Literal, Tuple

import lightning as L
import numpy as np
import torch
import xarray as xr
import zarr
from einops import rearrange
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
        time_chunk_size: int = 8,
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
        self.time_chunk_size = time_chunk_size
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
        self._zarr_var_arrays = None  # list[(zarr.Array, has_level_dim)]
        self._zarr_time_indices = (
            None  # int64 array: dataset idx → zarr time axis position
        )
        self._time_values = None  # datetime64 array of filtered time steps

    # ensure ds is opened in the current process (worker-safe)
    def _ensure_open(self):
        if self._ds is not None:
            return
        ds = xr.open_zarr(
            self.data_dir, chunks={"time": self.time_chunk_size}
        )  # lazy/dask
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

        if set(self._hours) != {0, 6, 12, 18}:
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

        # --- fast path: raw zarr arrays + time index mapping ---
        # Bypass xarray/dask in __getitem__ to cut per-sample graph overhead.
        self._time_values = ds["time"].values  # datetime64[ns], filtered
        # open_consolidated reads .zmetadata once, avoiding GPFS per-array lookups.
        try:
            zstore = zarr.open_consolidated(self.data_dir, mode="r")
        except KeyError:
            zstore = zarr.open(self.data_dir, mode="r")
        # Map filtered datetime64 times → integer positions in zarr time axis.
        zarr_time_hours = zstore["time"][:]  # int64 hours since 1959-01-01
        epoch = np.datetime64("1959-01-01T00:00", "h")
        ds_hours = (self._time_values.astype("datetime64[h]") - epoch).astype(np.int64)
        self._zarr_time_indices = np.searchsorted(zarr_time_hours, ds_hours).astype(
            np.int64
        )
        # Build per-variable zarr array list matching time_vars / all_field_names.
        self._zarr_var_arrays = [
            (zstore[f], zstore[f].ndim == 4)  # (array, has_level_dim)
            for f in self.time_vars
        ]
        # Detect if zarr stores spatial dims as (lon, lat) instead of (lat, lon).
        # WeatherBench 1.5° zarr uses (lon, lat) order; 0.25° uses (lat, lon).
        _sample_arr = zstore[self.time_vars[0]]
        _dims = _sample_arr.attrs.get("_ARRAY_DIMENSIONS", [])
        self._zarr_spatial_transposed = (
            len(_dims) >= 2 and _dims[-2] == "longitude" and _dims[-1] == "latitude"
        )
        if self._zarr_spatial_transposed:
            log.info("Detected (lon, lat) zarr storage order — will transpose spatial dims.")
        log.info(f"Opened dataset in pid={os.getpid()} with {self.len} samples.")

    def __len__(self):
        self._ensure_open()
        return self.len

    def __getitem__(self, idx: int):
        self._ensure_open()
        if self.start_indices is not None:
            idx = self.start_indices[idx]
        # Direct zarr reads — avoids per-sample dask graph compilation.
        t0 = int(self._zarr_time_indices[idx])
        t_end = t0 + self._stride * self._n_timesteps
        channels = []
        for zarr_arr, has_levels in self._zarr_var_arrays:
            chunk = zarr_arr[t0 : t_end : self._stride]
            if self._zarr_spatial_transposed:
                chunk = chunk.swapaxes(-1, -2)  # (..., lon, lat) -> (..., lat, lon)
            if not has_levels:
                chunk = chunk[:, np.newaxis, :, :]  # (t, 1, lat, lon)
            channels.append(chunk)
        arr = np.concatenate(channels, axis=1).astype(np.float32)  # (t, c, lat, lon)
        sample_time = self._time_values[
            idx : idx + self._stride * self._n_timesteps : self._stride
        ]
        x = torch.from_numpy(arr)
        x = self.preprocess(x)
        if self.add_day_year_progress:
            seconds_since_epoch = get_seconds_since_epoch(sample_time)
            year_progress, day_progress = self._get_day_year_progress(
                seconds_since_epoch
            )
            # expand() gives a zero-copy view; torch.cat materialises once below.
            additional_inputs = (
                self.additional_inputs.float()
                .unsqueeze(0)
                .expand(self.n_timesteps, -1, -1, -1)
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
            "time_s": sample_time.astype("datetime64[s]").astype(np.int64),
            "idx": idx,
        }

    def _get_day_year_progress(self, seconds_since_epoch: torch.Tensor):
        H = self.additional_inputs.shape[1]
        W = self.additional_inputs.shape[2]
        year_progress = get_year_progress(seconds_since_epoch)  # (t,)
        # expand() creates a zero-copy view; cat() in the caller materialises once.
        year_progress = (
            torch.as_tensor(year_progress).view(-1, 1, 1, 1).expand(-1, 1, H, W)
        )
        day_progress = get_day_progress(seconds_since_epoch, self.longitude)  # (t, w)
        day_progress = (
            torch.as_tensor(day_progress).view(-1, 1, 1, W).expand(-1, 1, H, W)
        )
        return year_progress, day_progress

    def _load_static_field(self, name: str, h: int, w: int) -> torch.Tensor:
        """Load a 2-D static field from the dataset, falling back to zeros if absent."""
        if name not in self._ds:
            log.info(f"[WARN] Static field '{name}' not found — using zeros.")
            return torch.zeros(1, h, w)
        arr = torch.as_tensor(self._ds[name].to_numpy())
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T  # ensure (lat, lon) = (h, w)
        return rearrange(arr, "w h -> 1 w h")

    def _additional_inputs(self):
        lon = self._ds["longitude"]
        lat = self._ds["latitude"]
        h, w = lat.size, lon.size

        land_sea_mask = self._load_static_field("land_sea_mask", h, w)
        soil_type = self._load_static_field("soil_type", h, w)

        # geopotential_at_surface needs to be normalized
        geopotential_at_surface = self._load_static_field(
            "geopotential_at_surface", h, w
        )
        if geopotential_at_surface.any():
            geopotential_at_surface = (
                geopotential_at_surface - geopotential_at_surface.mean()
            ) / geopotential_at_surface.std()

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
                rearrange(sinlat, "w h -> 1 w h"),
                rearrange(coslat_coslon, "w h -> 1 w h"),
                rearrange(coslat_sinlon, "w h -> 1 w h"),
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
        # Rechunk so each dask chunk spans a fixed number of timesteps but the
        # full spatial field — this gives predictable, large sequential reads.
        self._stacked = stacked.chunk(
            {"time": self.time_chunk_size, "latitude": -1, "longitude": -1}
        )

        self.all_field_names = all_field_names

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
        # In-place normalisation avoids a 1.45 GB intermediate allocation at 0.25°.
        x -= self.mean
        x /= self.std
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
        time_chunk_size: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        # Check if data_dir_local exists
        if data_dir_local is not None:
            if os.path.exists(data_dir_local) or data_dir_local.startswith("gs://"):
                self.data_dir = data_dir_local
            else:
                self.data_dir = data_dir_global
        else:
            self.data_dir = data_dir_global

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
            time_chunk_size=self.hparams.time_chunk_size,
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
