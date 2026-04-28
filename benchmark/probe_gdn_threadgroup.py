"""Benchmark MLX gated-delta Metal kernel threadgroup-y choices."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models import gated_delta as gated_delta_mod

from dflash_mlx.runtime import configure_mlx_memory_limits


def _run_kernel(
    *,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    threadgroup_y: int,
) -> tuple[mx.array, mx.array]:
    batch, tokens, key_heads, key_dim = k.shape
    value_heads, value_dim = v.shape[2:]
    return gated_delta_mod._gated_delta_kernel(
        inputs=[q, k, v, g, beta, state, tokens],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", key_dim),
            ("Dv", value_dim),
            ("Hk", key_heads),
            ("Hv", value_heads),
        ],
        grid=(32, value_dim, batch * value_heads),
        threadgroup=(32, int(threadgroup_y), 1),
        output_shapes=[(batch, tokens, value_heads, value_dim), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def _time_variant(args: argparse.Namespace, threadgroup_y: int, tensors: dict[str, mx.array]) -> dict[str, Any]:
    for _ in range(int(args.warmup)):
        mx.eval(*_run_kernel(threadgroup_y=threadgroup_y, **tensors))
    samples: list[float] = []
    last_y = None
    last_state = None
    for _ in range(int(args.repeats)):
        start = time.perf_counter_ns()
        last_y, last_state = _run_kernel(threadgroup_y=threadgroup_y, **tensors)
        mx.eval(last_y, last_state)
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return {
        "best_ms": min(samples),
        "mean_ms": sum(samples) / len(samples),
        "samples_ms": samples,
        "last_y": last_y,
        "last_state": last_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threadgroup-y", default="1,2,4,8,16")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    memory_config = configure_mlx_memory_limits()
    batch = 1
    key_heads = 16
    value_heads = 48
    key_dim = 128
    value_dim = 128
    tokens = int(args.tokens)
    tensors = {
        # Keep synthetic values in a stable range. Raw normal q/k/v can make
        # this recurrent rule overflow and hide correctness behind NaNs.
        "q": (0.02 * mx.random.normal((batch, tokens, key_heads, key_dim))).astype(mx.bfloat16),
        "k": (0.02 * mx.random.normal((batch, tokens, key_heads, key_dim))).astype(mx.bfloat16),
        "v": (0.02 * mx.random.normal((batch, tokens, value_heads, value_dim))).astype(mx.bfloat16),
        "g": (0.9 + 0.09 * mx.random.uniform(shape=(batch, tokens, value_heads))).astype(mx.float32),
        "beta": (0.1 * mx.random.uniform(shape=(batch, tokens, value_heads))).astype(mx.bfloat16),
        "state": mx.zeros((batch, value_heads, value_dim, key_dim), dtype=mx.float32),
    }
    mx.eval(*tensors.values())

    results: dict[str, Any] = {
        "memory_config": memory_config,
        "shape": {
            "batch": batch,
            "tokens": tokens,
            "key_heads": key_heads,
            "value_heads": value_heads,
            "key_dim": key_dim,
            "value_dim": value_dim,
        },
        "variants": {},
    }
    reference_y = None
    reference_state = None
    for raw in args.threadgroup_y.split(","):
        threadgroup_y = int(raw.strip())
        timed = _time_variant(args, threadgroup_y, tensors)
        y = timed.pop("last_y")
        state = timed.pop("last_state")
        if reference_y is None:
            reference_y = y
            reference_state = state
            max_y_diff = 0.0
            max_state_diff = 0.0
        else:
            max_y_diff = float(mx.max(mx.abs(reference_y - y)).item())
            max_state_diff = float(mx.max(mx.abs(reference_state - state)).item())
        timed["max_y_diff_vs_first"] = max_y_diff
        timed["max_state_diff_vs_first"] = max_state_diff
        results["variants"][str(threadgroup_y)] = timed
        mx.clear_cache()

    text = json.dumps(results, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
