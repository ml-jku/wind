#!/bin/bash
#SBATCH --job-name=train_era5
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=03-00:00:00  # 3 days
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --nodes=1

cd /home/af47162/git/wind

unset SLURM_NTASKS

uv run src/train.py \
    trainer.devices=[0,1,2,3] \
    experiment=era5/train/025_resolution \
    data.batch_size=2
