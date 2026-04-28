"""Probe whether explicit MLX streams help independent MLP gate/up projections."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from dflash_mlx.runtime import (
    configure_mlx_memory_limits,
    load_target_bundle,
    _target_text_model,
)
from mlx_lm.models.activations import swiglu


def _first_hybrid_mlp(model: Any) -> Any:
    text_model = _target_text_model(model)
    for layer in text_model.layers:
        mlp = getattr(layer, "mlp", None)
        if getattr(mlp, "_dflash_hybrid_mlp", False):
            return mlp
    raise RuntimeError("no hybrid MLP installed")


def _time_ms(fn: Callable[[mx.array], mx.array], x: mx.array, repeats: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        mx.eval(fn(x))
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        y = fn(x)
        mx.eval(y)
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    best = min(samples)
    return {
        "best_ms": best,
        "mean_ms": sum(samples) / len(samples),
        "samples_ms": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--m-values", default="128,256,512,1024,2048")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    memory_config = configure_mlx_memory_limits()
    os.environ["DFLASH_HYBRID_MLP"] = "1"
    os.environ["DFLASH_HYBRID_MLP_THRESHOLD"] = str(int(args.threshold))
    os.environ.setdefault("DFLASH_VERIFY_DECODE_ONLY", "1")
    model, _tokenizer, _meta = load_target_bundle(args.model, lazy=True)
    mlp = _first_hybrid_mlp(model)

    stream_a = mx.new_stream(mx.gpu)
    stream_b = mx.new_stream(mx.gpu)

    def sequential(x: mx.array) -> mx.array:
        gate = mlp.bf16_gate_proj(x)
        up = mlp.bf16_up_proj(x)
        return mlp.bf16_down_proj(swiglu(gate, up))

    def explicit_streams(x: mx.array) -> mx.array:
        # M3 Max / MLX scheduling probe: gate and up projections are
        # independent large-M matmuls. If MLX/Metal can overlap kernels across
        # streams, this should beat the normal single-stream sequence.
        with mx.stream(stream_a):
            gate = mlp.bf16_gate_proj(x)
        with mx.stream(stream_b):
            up = mlp.bf16_up_proj(x)
        return mlp.bf16_down_proj(swiglu(gate, up))

    results: dict[str, Any] = {
        "model": str(Path(args.model).expanduser()),
        "memory_config": memory_config,
        "threshold": int(args.threshold),
        "m_values": {},
    }
    for m_value in [int(v.strip()) for v in args.m_values.split(",") if v.strip()]:
        x = mx.random.normal((1, m_value, int(mlp.bf16_gate_proj.weight.shape[1]))).astype(mx.bfloat16)
        mx.eval(x)
        seq_out = sequential(x)
        stream_out = explicit_streams(x)
        mx.eval(seq_out, stream_out)
        diff = mx.abs(seq_out - stream_out)
        results["m_values"][str(m_value)] = {
            "max_abs_diff": float(mx.max(diff).item()),
            "sequential": _time_ms(sequential, x, int(args.repeats), int(args.warmup)),
            "streams": _time_ms(explicit_streams, x, int(args.repeats), int(args.warmup)),
        }
        mx.clear_cache()

    text = json.dumps(results, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
