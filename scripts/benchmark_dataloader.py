"""
DataLoader throughput benchmark for ERA5 0.25° training.

Measures samples/sec and effective GB/sec delivered by the DataLoader,
helping identify whether data loading is a training bottleneck.

Run with:
    PYTHONPATH=. uv run python scripts/benchmark_dataloader.py
    PYTHONPATH=. uv run python scripts/benchmark_dataloader.py --num-workers 4 --batch-size 2
    PYTHONPATH=. uv run python scripts/benchmark_dataloader.py --num-workers 2 4 8 --batch-size 2
"""

import argparse
import statistics
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = "/data/fs201057/af47162/1959-2022-6h-1440x721.zarr"
STATS_PATH = "src/datasets/stats/1959-2022-6h-240x121_equiangular_with_poles_conservative_stats.pkl"
FIELDS = [
    "temperature",
    "geopotential",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation_6hr",
]
N_TIMESTEPS = 5
TIMESPAN_TRAIN = ("1959-01-01", "2020-12-31")
HOURS_TRAIN = [0, 6, 12, 18]
SAMPLE_BYTES = 5 * 70 * 1440 * 704 * 4  # uncompressed float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataloader(
    num_workers: int,
    batch_size: int,
    prefetch_factor: int,
    time_chunk_size: int,
    n_batches_limit: int = None,
):
    from torch.utils.data import DataLoader

    from src.datasets.era5 import ERA5Dataset
    from src.utils.era5 import load_stats

    all_stats = load_stats(STATS_PATH, FIELDS)

    dataset = ERA5Dataset(
        data_dir=DATA_DIR,
        split="train",
        n_timesteps=N_TIMESTEPS,
        fields=FIELDS,
        hours=HOURS_TRAIN,
        stride=1,
        timespan=TIMESPAN_TRAIN,
        mean=all_stats["mean"],
        std=all_stats["std"],
        add_day_year_progress=True,
        time_chunk_size=time_chunk_size,
    )

    if n_batches_limit is not None:
        from torch.utils.data import Subset

        n_samples = n_batches_limit * batch_size
        dataset = Subset(dataset, list(range(min(n_samples, len(dataset)))))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        drop_last=True,
    )
    return loader


def _bench_loader(loader, n_warmup: int, n_measure: int, desc: str) -> dict:
    """
    Iterate the loader, skip n_warmup batches, then measure n_measure batches.
    Returns dict with timing stats.
    """
    print(f"\n  {desc}")
    print(f"  Warming up ({n_warmup} batches)...", flush=True)

    batch_times = []
    t_start_total = time.perf_counter()

    for i, batch in enumerate(loader):
        t_batch = time.perf_counter()

        if i == 0:
            t_first = t_batch - t_start_total
            print(f"  Time to first batch: {t_first:.2f}s")

        if i < n_warmup:
            t_prev = t_batch
            continue

        if i == n_warmup:
            t_prev = t_batch
            print(f"  Measuring ({n_measure} batches)...", flush=True)
            continue

        batch_times.append(t_batch - t_prev)
        t_prev = t_batch

        if len(batch_times) >= n_measure:
            break

    if not batch_times:
        return {}

    batch_size = batch["target_fields"].shape[0]
    sample_gb = SAMPLE_BYTES / 1e9

    times_ms = [t * 1000 for t in batch_times]
    samples_per_sec = [batch_size / t for t in batch_times]
    gb_per_sec = [batch_size * sample_gb / t for t in batch_times]

    results = {
        "batch_size": batch_size,
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "median_samples_per_sec": statistics.median(samples_per_sec),
        "median_gb_per_sec": statistics.median(gb_per_sec),
    }

    print(
        f"  Batch delivery time: median={results['median_ms']:.0f}ms  "
        f"min={results['min_ms']:.0f}ms  max={results['max_ms']:.0f}ms"
    )
    print(
        f"  Throughput: {results['median_samples_per_sec']:.2f} samples/sec  "
        f"({results['median_gb_per_sec']:.2f} GB/sec uncompressed)"
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark ERA5 DataLoader throughput")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument(
        "--num-workers",
        nargs="+",
        type=int,
        default=[2, 4],
        help="Worker counts to test",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--time-chunk-size", type=int, default=1)
    parser.add_argument(
        "--n-warmup", type=int, default=3, help="Batches to skip before measuring"
    )
    parser.add_argument("--n-measure", type=int, default=10, help="Batches to measure")
    args = parser.parse_args()

    print("=" * 70)
    print("ERA5 DataLoader Benchmark")
    print("=" * 70)
    print(f"data_dir:         {args.data_dir}")
    print(f"batch_size:       {args.batch_size}")
    print(f"prefetch_factor:  {args.prefetch_factor}")
    print(f"time_chunk_size:  {args.time_chunk_size}")
    print(f"n_timesteps:      {N_TIMESTEPS}")
    print(f"sample_size:      {SAMPLE_BYTES / 1e9:.2f} GB (uncompressed float32)")
    print(f"n_warmup:         {args.n_warmup}")
    print(f"n_measure:        {args.n_measure}")

    all_results = {}

    for nw in args.num_workers:
        desc = f"num_workers={nw}  batch_size={args.batch_size}  prefetch={args.prefetch_factor}"
        loader = _make_dataloader(
            num_workers=nw,
            batch_size=args.batch_size,
            prefetch_factor=args.prefetch_factor,
            time_chunk_size=args.time_chunk_size,
            n_batches_limit=args.n_warmup + args.n_measure + 2,
        )
        results = _bench_loader(loader, args.n_warmup, args.n_measure, desc)
        all_results[nw] = results
        del loader

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"{'num_workers':>12}  {'median ms/batch':>16}  {'samples/sec':>12}  {'GB/sec (uncomp)':>16}"
    )
    print("-" * 62)
    for nw, r in all_results.items():
        if r:
            print(
                f"{nw:>12}  {r['median_ms']:>16.0f}  "
                f"{r['median_samples_per_sec']:>12.2f}  {r['median_gb_per_sec']:>16.2f}"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
