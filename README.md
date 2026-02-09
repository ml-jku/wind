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
git clone https://github.com/<your-org>/wind.git
cd wind
uv sync
```

To add new dependencies:

```bash
uv add <package-name>
```

# Data

WIND is trained on [ERA5 reanalysis](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) data from the [WeatherBench2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) benchmark, stored in Zarr format and loaded lazily via xarray/dask.

| Property | Value |
|---|---|
| Time range | 1959–2022, 6-hourly |
| Spatial resolution | 240 x 121 equiangular grid with poles |
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

1. Download the ERA5 dataset in Zarr format from [WeatherBench2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5).

2. Configure the data path. You have two options:

   - **Local path** (preferred, checked first): set `data_dir_local` in `configs/local/default.yaml`:

     ```yaml
     data:
       data_dir_local: /path/to/era5.zarr
     ```

   - **Global fallback path**: set `data_dir` in `configs/paths/era5.yaml`:

     ```yaml
     data_dir: /path/to/era5.zarr
     ```

   The datamodule will use the local path if it exists, otherwise fall back to the global path.

3. Normalization statistics are precomputed and included in the repository at `src/datasets/stats/`.

# Training

Training is managed via [Hydra](https://hydra.cc/) and [PyTorch Lightning](https://lightning.ai/).

## Quick Start

```bash
# Train with the default 6-hourly experiment config
python src/train.py experiment=era5/train/6hourly
```

## Overriding Configurations

Hydra allows overriding any configuration parameter from the command line:

```bash
# Change batch size and learning rate
python src/train.py experiment=era5/train/6hourly data.batch_size=16 model.optimizer.lr=1e-4

# Resume from a checkpoint
python src/train.py experiment=era5/train/6hourly ckpt_path=/path/to/checkpoint.ckpt

# Disable Weights & Biases logging
python src/train.py experiment=era5/train/6hourly logger=[]

# Run in debug mode
python src/train.py experiment=era5/train/6hourly debug=default
```

## Logging

Training is logged to [Weights & Biases](https://wandb.ai/) by default. Configure it in `configs/logger/wandb.yaml` or disable it with `logger=[]`.

W&B and other environment variables are configured in the `.env` file at the project root:

```bash
WANDB_ENTITY=<your-wandb-entity>
WANDB_BASE_URL=https://api.wandb.ai
WANDB_IGNORE_GLOBS=*.log
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
