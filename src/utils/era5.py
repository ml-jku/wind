import pickle
from typing import List, Tuple

import numpy as np
import torch
import xarray
from einops import rearrange

ATHMOSPHERIC_CHANNEL_NAMES = [
    "temperature",
    "geopotential",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
]

ATHMOSPHERIC_LEVELS = [
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
]

_SEC_PER_HOUR = 3600
_HOUR_PER_DAY = 24
SEC_PER_DAY = _SEC_PER_HOUR * _HOUR_PER_DAY
_AVG_DAY_PER_YEAR = 365.24219
AVG_SEC_PER_YEAR = SEC_PER_DAY * _AVG_DAY_PER_YEAR


# Code from https://github.com/google-deepmind/graphcast/blob/main/graphcast/losses.py
def get_era5_area_weighting(resolution: float = 1.5):
    # Create latitude tensor from 90 to -90 inclusive
    latitude = torch.arange(90, -90 - resolution, -resolution, dtype=torch.float32)

    # Compute delta_latitude assuming uniform spacing
    delta_latitude = torch.abs(latitude[0] - latitude[1])

    # Check latitude range
    if not torch.isclose(latitude.max(), torch.tensor(90.0)) or not torch.isclose(
        latitude.min(), torch.tensor(-90.0)
    ):
        raise ValueError(
            f"Latitude vector {latitude} does not start/end at ±90 degrees."
        )

    # Compute weights
    weights = torch.cos(torch.deg2rad(latitude)) * torch.sin(
        torch.deg2rad(delta_latitude / 2)
    )

    # Adjust pole weights
    pole_weight = torch.sin(torch.deg2rad(delta_latitude / 4)) ** 2
    weights[0] = pole_weight
    weights[-1] = pole_weight

    # Repeat weights along longitude
    lon_count = int(360 / resolution)
    weights = weights.unsqueeze(1).repeat(1, lon_count)

    weights = weights / weights.mean()

    return weights


def get_era5_channel_weighting(channel_names: List[str]) -> torch.Tensor:
    all_weights = []
    for channel_name in channel_names:
        if channel_name in ATHMOSPHERIC_CHANNEL_NAMES:
            all_weights.append(_atmosphere_weighting())
        else:
            all_weights.append(_surface_weighting(channel_name))
    return torch.cat(all_weights, dim=0)


# Same as in GraphCast
def _atmosphere_weighting() -> torch.Tensor:
    weighting = torch.tensor(ATHMOSPHERIC_LEVELS, dtype=torch.float32)
    weighting = weighting / weighting.sum()
    return weighting


# Same as in GraphCast
def _surface_weighting(channel_name: str) -> torch.Tensor:
    if channel_name == "2m_temperature":
        return torch.tensor([1.0], dtype=torch.float32)
    else:
        return torch.tensor([0.1], dtype=torch.float32)


def load_stats(
    stats_path: str, fields: List[str], diff_var_channel_weights: str = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    all_mean = []
    all_std = []
    all_diff_std = []
    for f in fields:
        # Special treatment for total precipitation
        if f == "total_precipitation_6hr":
            f = "total_precipitation_6hr_log10"
        if f == "total_precipitation_12hr":
            f = "total_precipitation_12hr_log10"
        if f == "total_precipitation_24hr":
            f = "total_precipitation_24hr_log10"
        mean = torch.as_tensor(stats[f]["mean"].values)
        std = torch.as_tensor(stats[f]["std"].values)
        if diff_var_channel_weights is not None:
            try:
                diff_std = torch.as_tensor(stats[f][diff_var_channel_weights].values)
            except KeyError:
                print(f"KeyError: {diff_var_channel_weights} not found for {f}")
            if diff_std.ndim in [0, 2]:
                diff_std = diff_std.unsqueeze(0)
            all_diff_std.append(diff_std)
        if mean.ndim in [0, 2]:
            mean = mean.unsqueeze(0)
        if std.ndim in [0, 2]:
            std = std.unsqueeze(0)

        all_mean.append(mean)
        all_std.append(std)

    # Calculate mean and std
    mean = torch.cat(all_mean, dim=0)
    std = torch.cat(all_std, dim=0)
    if diff_var_channel_weights is not None:
        diff_std = torch.cat(all_diff_std, dim=0)
        if diff_std.ndim == 3:
            diff_std = diff_std.mean(dim=(-1, -2))
        diff_std = rearrange(diff_std, "c -> c 1 1")
    else:
        diff_std = None
    if mean.ndim == 3:
        mean = mean.mean(dim=(-1, -2))
    if std.ndim == 3:
        std = std.mean(dim=(-1, -2))
    mean = rearrange(mean, "c -> c 1 1")
    std = rearrange(std, "c -> c 1 1")

    return {"mean": mean, "std": std, "diff_std": diff_std}


def get_era5_field_names(channel_names: List[str]) -> List[str]:
    all_field_names = []
    for field_name in channel_names:
        if field_name in [
            "temperature",
            "geopotential",
            "specific_humidity",
            "u_component_of_wind",
            "v_component_of_wind",
        ]:
            for level in [
                50,
                100,
                150,
                200,
                250,
                300,
                400,
                500,
                600,
                700,
                850,
                925,
                1000,
            ]:
                all_field_names.append(f"{level}_{field_name}")
        else:
            all_field_names.append(field_name)
    return all_field_names


def get_seconds_since_epoch(datetime_sequence) -> np.ndarray:
    """Computes seconds since epoch from `data` in place if missing."""
    if isinstance(datetime_sequence, xarray.DataArray):
        datetime_sequence = datetime_sequence.data
    return datetime_sequence.astype("datetime64[s]").astype(np.int64)


def get_year_progress(seconds_since_epoch: np.ndarray) -> np.ndarray:
    """Computes year progress for times in seconds.

    Args:
      seconds_since_epoch: Times in seconds since the "epoch" (the point at which
        UNIX time starts).

    Returns:
      Year progress normalized to be in the [0, 1) interval for each time point.
    """

    # Start with the pure integer division, and then float at the very end.
    # We will try to keep as much precision as possible.
    years_since_epoch = (
        seconds_since_epoch / SEC_PER_DAY / np.float64(_AVG_DAY_PER_YEAR)
    )
    # Note depending on how these ops are down, we may end up with a "weak_type"
    # which can cause issues in subtle ways, and hard to track here.
    # In any case, casting to float32 should get rid of the weak type.
    # [0, 1.) Interval.
    return np.mod(years_since_epoch, 1.0).astype(np.float32)


def get_day_progress(
    seconds_since_epoch: np.ndarray,
    longitude: np.ndarray,
) -> np.ndarray:
    """Computes day progress for times in seconds at each longitude.

    Args:
      seconds_since_epoch: 1D array of times in seconds since the 'epoch' (the
        point at which UNIX time starts).
      longitude: 1D array of longitudes at which day progress is computed.

    Returns:
      2D array of day progress values normalized to be in the [0, 1) inverval
        for each time point at each longitude.
    """

    # [0.0, 1.0) Interval.
    day_progress_greenwich = np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY

    # Offset the day progress to the longitude of each point on Earth.
    longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
    day_progress = np.mod(
        day_progress_greenwich[..., np.newaxis] + longitude_offsets, 1.0
    )
    return day_progress.astype(np.float32)
