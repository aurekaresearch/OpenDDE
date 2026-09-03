#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""
Minimal runtime check for cuEquivariance triangle kernels used by OpenDDE.

This script is meant to run directly on an inference server. It checks:

1. CUDA / PyTorch / cuEquivariance imports.
2. The OpenDDE PairformerBlock dispatches into cuEquivariance functions.
3. A CUDA forward pass succeeds with the cuEquivariance path.
4. cuEquivariance output is numerically close to the torch fallback.
5. Optional timing and profiler trace export.

Important:
- Use N >= 128 by default. The repo's own tests note that small sequence lengths
  can fall back away from the optimized cueq path.
- For triangle multiplicative update, c_z must equal c_hidden_mul to stay on the
  cueq path.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether OpenDDE is really using cuEquivariance runtime kernels."
    )
    parser.add_argument(
        "--preset",
        choices=["custom", "opendde_v1"],
        default="custom",
        help="Apply a known OpenDDE model preset before running the check.",
    )
    parser.add_argument("--tokens", type=int, default=128, help="Sequence length.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--dtype",
        choices=["fp32", "bf16", "fp16"],
        default="bf16",
        help="Tensor dtype used for the check.",
    )
    parser.add_argument("--c-z", type=int, default=128, help="Pair channel size.")
    parser.add_argument(
        "--c-s", type=int, default=384, help="Single channel size. Use 0 to disable."
    )
    parser.add_argument(
        "--c-hidden-mul",
        type=int,
        default=128,
        help="Triangle multiplicative hidden size. Must equal c_z for cueq.",
    )
    parser.add_argument(
        "--c-hidden-pair-att",
        type=int,
        default=32,
        help="Triangle attention per-head hidden size.",
    )
    parser.add_argument(
        "--no-heads-pair",
        type=int,
        default=4,
        help="Triangle attention head count.",
    )
    parser.add_argument(
        "--hidden-scale-up",
        action="store_true",
        help="Match OpenDDE hidden_scale_up=True behavior for PairformerBlock.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Warmup iterations before timing.",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=10,
        help="Benchmark iterations for cueq vs torch.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the cueq vs torch latency benchmark.",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip torch profiler trace generation.",
    )
    parser.add_argument(
        "--profile-trace",
        default="cueq_profile_trace.json",
        help="Path to write profiler chrome trace.",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility."
    )
    return parser.parse_args()


def fail(message: str, exit_code: int = 1) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def dtype_from_name(name: str):
    import torch

    mapping = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    return mapping[name]


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "opendde_v1":
        args.c_z = 384
        args.c_s = 384
        args.hidden_scale_up = True
        args.c_hidden_pair_att = 32
    return args


def effective_pairformer_dims(args: argparse.Namespace) -> dict[str, int]:
    c_hidden_mul = args.c_hidden_mul
    no_heads_pair = args.no_heads_pair

    if args.hidden_scale_up:
        c_hidden_mul = args.c_z
        if args.c_z % args.c_hidden_pair_att != 0:
            fail(
                "With --hidden-scale-up, c_z must be divisible by c_hidden_pair_att. "
                f"Got c_z={args.c_z}, c_hidden_pair_att={args.c_hidden_pair_att}."
            )
        no_heads_pair = args.c_z // args.c_hidden_pair_att

    return {
        "c_hidden_mul": c_hidden_mul,
        "no_heads_pair": no_heads_pair,
    }


def autocast_context(dtype):
    import torch

    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def clone_inputs(
    s,
    z,
    pair_mask,
    track_autograd: bool = False,
):
    def clone_tensor(x, needs_autograd: bool):
        if x is None:
            return None
        out = x.detach().clone()
        if needs_autograd and out.is_floating_point():
            out.requires_grad_(True)
        return out

    return (
        clone_tensor(s, track_autograd),
        clone_tensor(z, track_autograd),
        clone_tensor(pair_mask, False),
    )


def build_block(args: argparse.Namespace, dtype, device):
    from opendde.model.modules.pairformer import PairformerBlock

    dims = effective_pairformer_dims(args)

    block = PairformerBlock(
        c_z=args.c_z,
        c_s=args.c_s,
        c_hidden_mul=args.c_hidden_mul,
        c_hidden_pair_att=args.c_hidden_pair_att,
        no_heads_pair=args.no_heads_pair,
        hidden_scale_up=args.hidden_scale_up,
    ).to(device)
    if dtype == "bf16":
        block = block.bfloat16()
    elif dtype == "fp16":
        block = block.half()
    else:
        block = block.float()
    return block


def build_inputs(args: argparse.Namespace, dtype, device):
    import torch

    torch_dtype = dtype_from_name(dtype)
    shape_pair = (args.batch_size, args.tokens, args.tokens, args.c_z)
    shape_mask = (args.batch_size, args.tokens, args.tokens)
    shape_single = (args.batch_size, args.tokens, args.c_s)

    z = torch.randn(*shape_pair, device=device, dtype=torch_dtype)
    pair_mask = torch.ones(*shape_mask, device=device, dtype=torch_dtype)
    s = None
    if args.c_s > 0:
        s = torch.randn(*shape_single, device=device, dtype=torch_dtype)
    return s, z, pair_mask


def print_env(args: argparse.Namespace) -> None:
    import torch

    dims = effective_pairformer_dims(args)

    print("[INFO] Environment")
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "dtype": args.dtype,
                "tokens": args.tokens,
                "batch_size": args.batch_size,
                "c_z": args.c_z,
                "c_s": args.c_s,
                "c_hidden_mul": args.c_hidden_mul,
                "c_hidden_pair_att": args.c_hidden_pair_att,
                "no_heads_pair": args.no_heads_pair,
                "hidden_scale_up": args.hidden_scale_up,
                "effective_c_hidden_mul": dims["c_hidden_mul"],
                "effective_no_heads_pair": dims["no_heads_pair"],
                "triangle_attention_env": os.environ.get("TRIANGLE_ATTENTION"),
                "triangle_multiplicative_env": os.environ.get(
                    "TRIANGLE_MULTIPLICATIVE"
                ),
                "CUEQ_TRITON_TUNING": os.environ.get("CUEQ_TRITON_TUNING"),
                "CUEQ_TRITON_CACHE_DIR": os.environ.get("CUEQ_TRITON_CACHE_DIR"),
            },
            indent=2,
        )
    )


def import_checks() -> dict[str, Any]:
    import cuequivariance_ops_torch  # noqa: F401
    import cuequivariance_torch  # noqa: F401
    import torch
    from cuequivariance_torch.primitives import triangle as cueq_triangle

    if not torch.cuda.is_available():
        fail("CUDA is not available.")

    return {
        "cueq_triangle_module": cueq_triangle,
    }


def install_dispatch_hooks(cueq_triangle_module):
    counts = {
        "triangle_multiplicative_update": 0,
        "triangle_attention": 0,
    }
    records = []

    original_tmu = cueq_triangle_module.triangle_multiplicative_update
    original_tattn = cueq_triangle_module.triangle_attention

    def wrapped_tmu(*args, **kwargs):
        tensor = args[0]
        counts["triangle_multiplicative_update"] += 1
        records.append(
            {
                "fn": "triangle_multiplicative_update",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "is_cuda": bool(tensor.is_cuda),
                "direction": kwargs.get("direction"),
            }
        )
        return original_tmu(*args, **kwargs)

    def wrapped_tattn(*args, **kwargs):
        tensor = args[0]
        counts["triangle_attention"] += 1
        records.append(
            {
                "fn": "triangle_attention",
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "is_cuda": bool(tensor.is_cuda),
            }
        )
        return original_tattn(*args, **kwargs)

    cueq_triangle_module.triangle_multiplicative_update = wrapped_tmu
    cueq_triangle_module.triangle_attention = wrapped_tattn

    def restore():
        cueq_triangle_module.triangle_multiplicative_update = original_tmu
        cueq_triangle_module.triangle_attention = original_tattn

    return counts, records, restore


def compare_tensors(name: str, a, b, dtype_name: str) -> dict[str, float]:
    import torch

    if a is None and b is None:
        return {"max_abs_diff": 0.0, "mean_abs_diff": 0.0}
    if (a is None) != (b is None):
        fail(f"{name} mismatch: one output is None and the other is not.")

    diff = (a.float() - b.float()).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    atol = {
        "fp32": 5e-4,
        "bf16": 5e-2,
        "fp16": 5e-2,
    }[dtype_name]
    rtol = {
        "fp32": 5e-4,
        "bf16": 5e-2,
        "fp16": 5e-2,
    }[dtype_name]

    try:
        torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)
    except AssertionError as exc:
        fail(
            f"{name} cueq vs torch outputs differ too much. "
            f"max_abs_diff={max_abs_diff:.6f}, mean_abs_diff={mean_abs_diff:.6f}. "
            f"Details: {exc}"
        )

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def run_eval_dispatch_and_correctness(
    args: argparse.Namespace, block, s, z, pair_mask, hook_state
) -> dict[str, Any]:
    import torch

    counts, records, _ = hook_state

    block_torch = copy.deepcopy(block).eval()
    block_cueq = copy.deepcopy(block).eval()

    s_torch, z_torch, pair_mask_torch = clone_inputs(s, z, pair_mask, False)
    s_cueq, z_cueq, pair_mask_cueq = clone_inputs(s, z, pair_mask, False)

    with torch.no_grad():
        with autocast_context(dtype_from_name(args.dtype)):
            out_s_torch, out_z_torch = block_torch(
                s_torch,
                z_torch,
                pair_mask_torch,
                triangle_multiplicative="torch",
                triangle_attention="torch",
                inplace_safe=False,
            )
            out_s_cueq, out_z_cueq = block_cueq(
                s_cueq,
                z_cueq,
                pair_mask_cueq,
                triangle_multiplicative="cuequivariance",
                triangle_attention="cuequivariance",
                inplace_safe=False,
            )

    if counts["triangle_multiplicative_update"] < 2:
        fail(
            "cuEquivariance triangle_multiplicative_update was not called as expected "
            f"(count={counts['triangle_multiplicative_update']})."
        )
    if counts["triangle_attention"] < 2:
        fail(
            "cuEquivariance triangle_attention was not called as expected "
            f"(count={counts['triangle_attention']})."
        )

    for item in records:
        if not item["is_cuda"]:
            fail(f"cuEquivariance function received non-CUDA tensor: {item}")

    z_diff = compare_tensors("z", out_z_cueq, out_z_torch, args.dtype)
    s_diff = compare_tensors("s", out_s_cueq, out_s_torch, args.dtype)

    return {
        "dispatch_counts": dict(counts),
        "dispatch_records": list(records),
        "z_diff": z_diff,
        "s_diff": s_diff,
    }


def benchmark_mode(
    args: argparse.Namespace,
    block,
    s,
    z,
    pair_mask,
    triangle_multiplicative: str,
    triangle_attention: str,
) -> float:
    import torch

    bench_block = copy.deepcopy(block).eval()
    s_in, z_in, pair_mask_in = clone_inputs(s, z, pair_mask, False)

    for _ in range(args.warmup_iters):
        with torch.no_grad():
            with autocast_context(dtype_from_name(args.dtype)):
                bench_block(
                    s_in,
                    z_in,
                    pair_mask_in,
                    triangle_multiplicative=triangle_multiplicative,
                    triangle_attention=triangle_attention,
                    inplace_safe=False,
                )
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.bench_iters):
        with torch.no_grad():
            with autocast_context(dtype_from_name(args.dtype)):
                bench_block(
                    s_in,
                    z_in,
                    pair_mask_in,
                    triangle_multiplicative=triangle_multiplicative,
                    triangle_attention=triangle_attention,
                    inplace_safe=False,
                )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / args.bench_iters


def run_profiler(args: argparse.Namespace, block, s, z, pair_mask) -> dict[str, Any]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    profile_block = copy.deepcopy(block).eval()
    s_in, z_in, pair_mask_in = clone_inputs(s, z, pair_mask, track_autograd=False)
    trace_path = Path(args.profile_trace).resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            with autocast_context(dtype_from_name(args.dtype)):
                profile_block(
                    s_in,
                    z_in,
                    pair_mask_in,
                    triangle_multiplicative="cuequivariance",
                    triangle_attention="cuequivariance",
                    inplace_safe=False,
                )
            torch.cuda.synchronize()

    prof.export_chrome_trace(str(trace_path))

    cuda_events = []
    keyword_hits = []
    keywords = ("cue", "triangle", "triton")

    try:
        events = prof.events()
    except Exception:
        events = []

    for event in events:
        device_type = str(getattr(event, "device_type", ""))
        if "CUDA" not in device_type:
            continue
        name = getattr(event, "name", None) or getattr(event, "key", None) or str(event)
        self_cuda = float(getattr(event, "self_cuda_time_total", 0.0) or 0.0)
        cuda_events.append((name, self_cuda))
        if any(token in name.lower() for token in keywords):
            keyword_hits.append((name, self_cuda))

    cuda_events.sort(key=lambda item: item[1], reverse=True)
    keyword_hits.sort(key=lambda item: item[1], reverse=True)

    return {
        "trace_path": str(trace_path),
        "top_cuda_events": cuda_events[:15],
        "keyword_hits": keyword_hits[:15],
    }


def main() -> None:
    args = apply_preset(parse_args())
    dims = effective_pairformer_dims(args)

    if args.tokens <= 100:
        fail(
            "--tokens must be > 100 for a reliable cueq kernel check. "
            "Small lengths can fall back away from the optimized path."
        )
    if dims["c_hidden_mul"] != args.c_z:
        fail(
            "Effective triangle multiplicative hidden size must equal c_z for cueq path. "
            f"Got effective_c_hidden_mul={dims['c_hidden_mul']}, c_z={args.c_z}."
        )
    if args.dtype == "fp16":
        # A100 supports fp16, but bf16 is the more relevant default for this repo.
        print(
            "[WARN] Running in fp16. bf16 is usually the more representative mode on A100."
        )

    import torch

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

    print_env(args)
    state = import_checks()
    device = torch.device("cuda:0")

    block = build_block(args, args.dtype, device)
    s, z, pair_mask = build_inputs(args, args.dtype, device)

    counts, records, restore = install_dispatch_hooks(state["cueq_triangle_module"])
    hook_state = (counts, records, restore)

    try:
        eval_result = run_eval_dispatch_and_correctness(
            args, block, s, z, pair_mask, hook_state
        )
    finally:
        restore()

    benchmark_result = None
    if not args.skip_benchmark:
        cueq_ms = benchmark_mode(
            args,
            block,
            s,
            z,
            pair_mask,
            triangle_multiplicative="cuequivariance",
            triangle_attention="cuequivariance",
        )
        torch_ms = benchmark_mode(
            args,
            block,
            s,
            z,
            pair_mask,
            triangle_multiplicative="torch",
            triangle_attention="torch",
        )
        benchmark_result = {
            "cueq_ms_per_iter": cueq_ms,
            "torch_ms_per_iter": torch_ms,
            "speedup_vs_torch": torch_ms / cueq_ms if cueq_ms > 0 else None,
        }

    profiler_result = None
    if not args.skip_profile:
        profiler_result = run_profiler(args, block, s, z, pair_mask)

    result = {
        "status": "PASS",
        "summary": (
            "cuEquivariance functions were called on CUDA tensors, "
            "forward succeeded, and outputs matched torch fallback."
        ),
        "eval": eval_result,
        "benchmark": benchmark_result,
        "profiler": profiler_result,
    }

    print("[PASS] cuEquivariance runtime check succeeded.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
