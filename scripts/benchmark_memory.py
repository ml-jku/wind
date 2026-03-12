"""
Memory & compute benchmark: UViT3DV2 (TransformerBlock) vs UViT3DV3 (FactorizedTransformerBlock).

Key insight: PyTorch's F.scaled_dot_product_attention uses Flash Attention by default,
which is O(N) in memory (not O(N^2)). This means both models have similar memory
footprints at any resolution. The factorized model's advantage is in COMPUTE (FLOPs):

  Full attention:      FLOPs ∝ (T * H * W)^2
  Factorized:          FLOPs ∝ T * (H * W)^2 + H * W * T^2  ≈  T * (H * W)^2  for T << H*W

This gives a T-fold reduction in attention FLOPs (T=5 → 5x fewer FLOPs at level 2+3).

Run with:
    PYTHONPATH=. uv run python scripts/benchmark_memory.py
    PYTHONPATH=. uv run python scripts/benchmark_memory.py --batch-sizes 1 2 4
"""

import argparse
import gc
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from omegaconf import OmegaConf

from src.models.dfot.uvit.uvit3d import UViT3DV2, UViT3DV3

# ---------------------------------------------------------------------------
# Architecture matching configs/model/era5/uvit.yaml
# ---------------------------------------------------------------------------
BASE_CFG = OmegaConf.create(
    {
        "channels": [256, 512, 1024, 2048],
        "emb_channels": 64,
        "patch_size": 2,  # overridden by --patch-size flag
        "block_dropouts": [0.0, 0.0, 0.0, 0.0],
        "num_updown_blocks": [4, 4, 4],
        "num_mid_blocks": 4,
        "num_heads": 4,
        "pos_emb_type": "rope",
        "use_checkpointing": [False, False, False, False],
    }
)

BLOCK_TYPES_FULL = ["ResBlock", "ResBlock", "TransformerBlock", "TransformerBlock"]
BLOCK_TYPES_FACTORIZED = [
    "ResBlock",
    "ResBlock",
    "FactorizedTransformerBlock",
    "FactorizedTransformerBlock",
]


def _valid_resolutions(patch_size: int) -> dict:
    """
    Return benchmark resolutions whose spatial dims are divisible by
    patch_size * 2^(n_downsamples) so ConvPixelUnshuffle never sees odd dims.
    n_downsamples = len(num_updown_blocks) = 3.
    """
    stride = patch_size * 8  # must divide H and W evenly

    def snap(x):
        return (x // stride) * stride

    return {
        f"1.5° ({snap(128)}×{snap(240)})": (70, snap(128), snap(240)),
        f"0.25° ({snap(720)}×{snap(1440)})": (70, snap(720), snap(1440)),
    }


N_TIMESTEPS = 5
ADDITIONAL_INPUTS = 10
WARMUP_ITERS = 2
BENCH_ITERS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    model_name: str
    resolution: str
    batch_size: int
    n_params_M: float
    infer_GiB: Optional[float] = None  # torch.no_grad forward
    train_fwd_GiB: Optional[float] = (
        None  # forward (activations stored for weight grad)
    )
    train_bwd_GiB: Optional[float] = None  # forward + backward
    infer_ms: Optional[float] = None  # median inference latency
    train_ms: Optional[float] = None  # median forward+backward latency
    errors: list = field(default_factory=list)


_DEVICE: torch.device = torch.device("cuda:0")  # set in main()


def _reset_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(_DEVICE)


def _peak_GiB() -> float:
    return torch.cuda.max_memory_allocated(_DEVICE) / (1024**3)


def _count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def _build_model(cls, block_types, x_shape, device, patch_size: int = 2):
    cfg = OmegaConf.merge(
        BASE_CFG,
        OmegaConf.create({"block_types": block_types, "patch_size": patch_size}),
    )
    model = cls(
        cfg=cfg,
        x_shape=torch.Size(x_shape),
        additional_inputs=ADDITIONAL_INPUTS,
        max_tokens=N_TIMESTEPS,
    )
    return model.to(device=device, dtype=torch.bfloat16)


def _make_inputs(batch_size, C, H, W, device, requires_grad=False):
    x = torch.randn(
        batch_size,
        N_TIMESTEPS,
        C,
        H,
        W,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=requires_grad,
    )
    additional = torch.zeros(
        batch_size,
        ADDITIONAL_INPUTS,
        H,
        W,
        device=device,
        dtype=torch.bfloat16,
    )
    t = torch.rand(batch_size, N_TIMESTEPS, device=device, dtype=torch.bfloat16)
    return x, t, additional


def _timed_run(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS) -> Optional[float]:
    """Returns median wall-clock time in ms, or None on OOM."""
    timings = []
    try:
        for i in range(warmup + iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= warmup:
                timings.append((t1 - t0) * 1000)
        timings.sort()
        return timings[len(timings) // 2]
    except torch.cuda.OutOfMemoryError:
        return None


def run_benchmark(
    cls,
    model_name: str,
    block_types: list,
    resolution_name: str,
    x_shape: tuple,
    batch_size: int,
    device: torch.device,
    patch_size: int = 2,
) -> BenchResult:
    C, H, W = x_shape
    result = BenchResult(
        model_name=model_name,
        resolution=resolution_name,
        batch_size=batch_size,
        n_params_M=0.0,
    )
    try:
        model = _build_model(cls, block_types, x_shape, device, patch_size=patch_size)
    except Exception as e:
        result.errors.append(f"build: {e}")
        return result

    result.n_params_M = _count_params(model)

    # ------------------------------------------------------------------
    # 1. Inference memory (torch.no_grad — activations NOT stored)
    # ------------------------------------------------------------------
    _reset_gpu()
    try:
        x, t, additional = _make_inputs(batch_size, C, H, W, device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, t, additional_inputs=additional)
        result.infer_GiB = _peak_GiB()
    except torch.cuda.OutOfMemoryError:
        result.infer_GiB = None
    except Exception as e:
        result.errors.append(f"infer_mem: {e}")
    finally:
        _reset_gpu()

    # ------------------------------------------------------------------
    # 2. Training-forward memory (activations stored for weight gradients)
    # ------------------------------------------------------------------
    _reset_gpu()
    try:
        x, t, additional = _make_inputs(batch_size, C, H, W, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, t, additional_inputs=additional)
        result.train_fwd_GiB = _peak_GiB()
    except torch.cuda.OutOfMemoryError:
        result.train_fwd_GiB = None
    except Exception as e:
        result.errors.append(f"train_fwd_mem: {e}")
    finally:
        _reset_gpu()

    # ------------------------------------------------------------------
    # 3. Training forward+backward memory
    # ------------------------------------------------------------------
    _reset_gpu()
    try:
        x, t, additional = _make_inputs(batch_size, C, H, W, device, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, t, additional_inputs=additional)
            loss = out.mean()
        loss.backward()
        result.train_bwd_GiB = _peak_GiB()
    except torch.cuda.OutOfMemoryError:
        result.train_bwd_GiB = None
    except Exception as e:
        result.errors.append(f"train_bwd_mem: {e}")
    finally:
        _reset_gpu()

    # ------------------------------------------------------------------
    # 4. Inference timing
    # ------------------------------------------------------------------
    if result.infer_GiB is not None:
        _reset_gpu()
        x, t, additional = _make_inputs(batch_size, C, H, W, device)

        def _infer():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(x, t, additional_inputs=additional)

        result.infer_ms = _timed_run(_infer)
        _reset_gpu()

    # ------------------------------------------------------------------
    # 5. Training forward+backward timing
    # ------------------------------------------------------------------
    if result.train_bwd_GiB is not None:
        _reset_gpu()
        x, t, additional = _make_inputs(batch_size, C, H, W, device, requires_grad=True)

        def _train():
            x.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, t, additional_inputs=additional)
                loss = out.mean()
            loss.backward()

        result.train_ms = _timed_run(_train)
        _reset_gpu()

    del model
    _reset_gpu()
    return result


def _fmt_mem(val: Optional[float]) -> str:
    return f"{val:6.1f} GiB" if val is not None else "    OOM   "


def _fmt_time(val: Optional[float]) -> str:
    if val is None:
        return "   OOM  "
    if val >= 1000:
        return f"{val / 1000:5.1f}  s"
    return f"{val:6.0f} ms"


def print_table(results: list[BenchResult]):
    header = (
        f"{'Model':<30} {'Resolution':<20} {'BS':>3} {'Params':>8} "
        f"{'Infer mem':>12} {'Train fwd':>12} {'Fwd+Bwd':>12} "
        f"{'Infer':>9} {'Fwd+Bwd':>9}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(
        f"{'':30} {'':20} {'':3} {'(M)':>8} "
        f"{'(GiB)':>12} {'(GiB)':>12} {'(GiB)':>12} "
        f"{'time':>9} {'time':>9}"
    )
    print(sep)
    for r in results:
        print(
            f"{r.model_name:<30} {r.resolution:<20} {r.batch_size:>3} {r.n_params_M:>7.1f}M "
            f"{_fmt_mem(r.infer_GiB):>12} {_fmt_mem(r.train_fwd_GiB):>12} {_fmt_mem(r.train_bwd_GiB):>12} "
            f"{_fmt_time(r.infer_ms):>9} {_fmt_time(r.train_ms):>9}"
        )
    print(sep)


def print_flops_table(patch_size: int = 2):
    """Theoretical attention FLOPs at 0.25° for the given patch_size."""
    T = N_TIMESTEPS
    H0 = 720 // patch_size  # spatial dims after patch embedding
    W0 = 1440 // patch_size
    print(f"\nTheoretical attention FLOPs at 0.25° — T={T}, patch_size={patch_size}")
    print("  (per head, per block, batch=1; Flash Attn gives O(N) memory for both)")
    print()
    print(
        f"  {'Level':>6}  {'H×W':>12}  {'Full attn FLOPs':>22}  {'Factorized FLOPs':>22}  {'Speedup':>8}"
    )
    print("  " + "-" * 82)
    for i in range(4):
        H = H0 // (2**i)
        W = W0 // (2**i)
        HW = H * W
        N_full = T * HW
        flops_full = N_full**2
        flops_factorized = T * HW**2 + HW * T**2
        speedup = flops_full / flops_factorized
        block = "ResBlock" if i < 2 else "Attn   "
        print(
            f"  {i:>2} ({block})  {H:>4}×{W:<5}  "
            f"{flops_full:>22,.0f}  {flops_factorized:>22,.0f}  {speedup:>7.1f}×"
        )
    print()
    print("  Note: Flash Attention avoids materializing the N×N matrix, so both")
    print("  strategies have O(N·d) memory. The speedup is purely in compute (FLOPs).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1],
        help="Batch sizes to test (default: 1)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=2,
        help="Patch size for the embedding layer (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (default: 0)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device found — exiting.")
        return

    device = torch.device(f"cuda:{args.device}")
    global _DEVICE
    _DEVICE = device
    gpu_name = torch.cuda.get_device_name(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    print(f"\nGPU {args.device}: {gpu_name}  ({total_mem:.1f} GiB)")
    print(f"patch_size={args.patch_size}  batch_sizes={args.batch_sizes}")
    print("Note: Flash Attention is active (F.scaled_dot_product_attention).")
    print("      Memory advantage of factorized attn is COMPUTE (FLOPs), not memory.\n")

    configs = [
        ("UViT3DV2 (FullAttn)", UViT3DV2, BLOCK_TYPES_FULL),
        ("UViT3DV3 (Factorized)", UViT3DV3, BLOCK_TYPES_FACTORIZED),
    ]

    resolutions = _valid_resolutions(args.patch_size)
    results = []
    for res_name, (C, H, W) in resolutions.items():
        print(f"=== {res_name} (C={C}, H={H}, W={W}) ===")
        for bs in args.batch_sizes:
            for model_name, cls, block_types in configs:
                tag = f"  {model_name}  bs={bs}"
                print(f"{tag:<45} ...", end=" ", flush=True)
                r = run_benchmark(
                    cls=cls,
                    model_name=model_name,
                    block_types=block_types,
                    resolution_name=res_name,
                    x_shape=(C, H, W),
                    batch_size=bs,
                    device=device,
                    patch_size=args.patch_size,
                )
                status = f"infer={_fmt_mem(r.infer_GiB).strip()}  train_bwd={_fmt_mem(r.train_bwd_GiB).strip()}"
                print(status)
                if r.errors:
                    for e in r.errors:
                        print(f"    WARN: {e}")
                results.append(r)
        print()

    print("\n" + "=" * 110)
    print("RESULTS SUMMARY")
    print("=" * 110)
    print_table(results)
    print_flops_table(args.patch_size)


if __name__ == "__main__":
    main()
