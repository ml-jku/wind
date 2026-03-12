<div align="center">

# WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling

[![arxiv](https://img.shields.io/badge/arXiv-2602.03924-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.03924)
[![python](https://img.shields.io/badge/-Python_3.11+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.8+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/docs/stable/)
[![lightning](https://img.shields.io/badge/-Lightning_2.6+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

This is the official code repository for the paper
**[WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling](https://arxiv.org/abs/2602.03924)**.

# News
- **March 12, 2026**: Training code for 0.25° resolution added.
- **Feb 9, 2026**: Training code released.
- **Feb 3, 2026**: Paper preprint available on [arXiv](https://arxiv.org/abs/2602.03924).

# Overview

WIND is a single pre-trained foundation model for weather and climate modeling
that replaces specialized baselines across a wide range of tasks without any task-specific fine-tuning.
It learns a task-agnostic prior of the atmosphere via a self-supervised video reconstruction objective using an unconditional video diffusion model.
At inference, diverse domain-specific problems are framed as inverse problems and solved via posterior sampling.

Supported tasks include:
- Probabilistic ensemble forecasting
- Spatial and temporal downscaling
- Sparse reconstruction
- Enforcing conservation laws
- Counterfactual storylines of extreme weather events under global warming scenarios

# Installation

## Requirements

- Python >= 3.11
- PyTorch >= 2.8.0

## Setup

Clone the repository and install dependencies using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ml-jku/wind
cd wind
uv sync
```

# Data

WIND is trained on [ERA5 reanalysis](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) data from the [WeatherBench2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) benchmark, stored in Zarr format and loaded lazily via xarray/dask.

Two spatial resolutions are supported:

| Resolution | Grid | GCS path |
|---|---|---|
| 1.5° (default) | 240 × 121 | `gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr` |
| 0.25° (high-res) | 1440 × 721 | `gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr` |

Common properties:

| Property | Value |
|---|---|
| Time range | 1959–2022, 6-hourly |
| Channels | 70 (see below) |
| Format | Zarr |

**Atmospheric fields** (70 channels):

| Field | Levels | Channels |
|---|---|---|
| Temperature | 13 pressure levels | 13 |
| Geopotential | 13 pressure levels | 13 |
| Specific humidity | 13 pressure levels | 13 |
| u-component of wind | 13 pressure levels | 13 |
| v-component of wind | 13 pressure levels | 13 |
| 2m temperature | surface | 1 |
| Mean sea level pressure | surface | 1 |
| 10m u-component of wind | surface | 1 |
| 10m v-component of wind | surface | 1 |
| Total precipitation (6hr) | surface | 1 |

Static inputs (land-sea mask, soil type, geopotential at surface, and lat/lon encodings) are additionally provided to the model as conditioning.

## Data Setup

The ERA5 data is available on Google Cloud Storage via [WeatherBench2](https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5/). You can either stream it directly from GCS or download it locally.

### Option A: Stream directly from GCS (no download required)

The dataloader uses `xarray.open_zarr` and supports `gs://` paths natively via `gcsfs`. No download needed — data is streamed on the fly:

```bash
# Install gcsfs for GCS access
uv add gcsfs
```

For both resolutions the GCS path is the built-in fallback — no `.env` variable is required unless you want to override it with a local copy.

Public GCS access requires no authentication. For faster throughput, run from a GCP instance in `us-central1`.

### Option B: Download locally

Download the Zarr dataset from [WeatherBench2 on GCS](https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5/) using `gsutil`:

```bash
# 1.5° (~100 GB)
gsutil -m cp -r gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr /path/to/local/

# 0.25° (~7 TB)
gsutil -m cp -r gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr /path/to/local/
```

Then point to the local paths via `.env` (see below).

### Configuring the data path via `.env`

Data paths (and other environment variables) are set in the `.env` file at the project root. Create it in the root folder and edit it:

```bash
# ERA5 data paths — local path or gs:// URI.
# If ERA5_1P5DEG_PATH is unset, falls back to the GCS URI in configs/data/era5_1p5deg.yaml.
# If ERA5_0P25DEG_PATH is unset, falls back to the GCS URI in configs/data/era5_0p25deg.yaml.
ERA5_1P5DEG_PATH=/path/to/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr
ERA5_0P25DEG_PATH=/path/to/1959-2022-6h-1440x721.zarr
```

For each variable, the datamodule uses the value if the path exists locally or starts with `gs://`, otherwise it falls back to the `data_dir_global` defined in the corresponding config.

Normalization statistics are precomputed and included in the repository at `src/datasets/stats/`.

# Training

Training is managed via [Hydra](https://hydra.cc/) and [PyTorch Lightning](https://lightning.ai/).

## Quick Start

```bash
# 1.5° resolution (default, ~240×121 grid)
uv run python src/train.py experiment=era5/train/era5_1p5deg

# 0.25° resolution (high-res, 1440×721 grid — requires more GPU memory)
uv run python src/train.py experiment=era5/train/era5_0p25deg
```

Alternatively, activate the virtual environment first and use `python` directly:

```bash
source .venv/bin/activate
python src/train.py experiment=era5/train/era5_1p5deg
```

### Memory requirements for 0.25°

A single sample at 0.25° is ~1.45 GB (5 timesteps × 70 channels × 1440 × 704 × float32). The default config uses `batch_size=2` and `num_workers=4` — scale down `num_workers` if host RAM is limited (~23 GB in the prefetch queue per GPU at defaults).

## Overriding Configurations

Hydra allows overriding any configuration parameter from the command line:

```bash
# Change batch size and learning rate
uv run python src/train.py experiment=era5/train/era5_1p5deg data.batch_size=16 model.optimizer.lr=1e-4

# Resume from a checkpoint
uv run python src/train.py experiment=era5/train/era5_1p5deg ckpt_path=/path/to/checkpoint.ckpt

# Disable Weights & Biases logging
uv run python src/train.py experiment=era5/train/era5_1p5deg logger=[]

# Run in debug mode
uv run python src/train.py experiment=era5/train/era5_1p5deg debug=default
```

## Logging

Training is logged to [Weights & Biases](https://wandb.ai/) by default. Configure it in `configs/logger/wandb.yaml` or disable it with `logger=[]`.

All environment variables — W&B credentials, data paths, and runtime settings — are configured in the `.env` file at the project root. A minimal setup looks like:

```bash
# W&B
WANDB_ENTITY=<your-wandb-entity>
WANDB_BASE_URL=https://api.wandb.ai
WANDB_IGNORE_GLOBS=*.log

# ERA5 data paths (local path or gs:// URI; omit to stream directly from GCS)
ERA5_1P5DEG_PATH=/path/to/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr
ERA5_0P25DEG_PATH=/path/to/1959-2022-6h-1440x721.zarr
```

Set `WANDB_ENTITY` to your own W&B team or username. 

# Acknowledgments

- [azula](https://github.com/probabilists/azula) - diffusion formalism
- [UViT3D](https://github.com/kwsong0113/diffusion-forcing-transformer) - UViT3D model


# Citation

If you like our work, please consider giving it a star 🌟 and cite us

```bibtex
@article{aich2026wind,
         title={WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling},
         author={Michael Aich and Andreas Fürst and Florian Sestak and Carlos Ruiz-Gonzalez
         and Niklas Boers and Johannes Brandstetter},
         year={2026},
         eprint={2602.03924},
         archivePrefix={arXiv},
         primaryClass={cs.LG},
         url={https://arxiv.org/abs/2602.03924},
}
```
