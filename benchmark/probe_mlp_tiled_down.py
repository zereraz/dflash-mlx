#!/usr/bin/env python3
"""Probe tiled SwiGLU+down accumulation for one bf16 target MLP.

The deployed hybrid MLP path materializes gate, up, and SwiGLU intermediates of
shape [B, S, intermediate]. On the measured Qwen target that is large enough to
be worth testing whether intermediate tiling can reduce memory traffic on Apple
GPU, even though it launches more matmuls.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
from mlx_lm.models.activations import swiglu

from dflash_mlx.runtime import (
    _dequantize_linear,
    _target_text_model,
    configure_mlx_memory_limits,
    load_target_bundle,
)


def _first_mlp(model: Any, layer_index: int | None) -> tuple[int, Any]:
    text_model = _target_text_model(model)
    if layer_index is not None:
        return int(layer_index), text_model.layers[int(layer_index)].mlp
    for index, layer in enumerate(text_model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate_proj"):
            return index, mlp
    raise ValueError("could not find target MLP")


def _time_ms(fn: Callable[[mx.array], mx.array], x: mx.array, *, repeats: int, warmup: int) -> dict[str, Any]:
    for _ in range(max(0, warmup)):
        mx.eval(fn(x))
    samples: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter_ns()
        y = fn(x)
        mx.eval(y)
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return {
        "best_ms": min(samples),
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
    }


def _linear_input_width(linear: Any) -> int:
    scales = getattr(linear, "scales", None)
    if scales is not None:
        return int(scales.shape[1]) * int(linear.group_size)
    return int(linear.weight.shape[1])


def _tiled_mlp_fn(
    *,
    gate_weight: mx.array,
    up_weight: mx.array,
    down_weight: mx.array,
    tile_size: int,
) -> Callable[[mx.array], mx.array]:
    intermediate = int(gate_weight.shape[0])
    tile = max(1, int(tile_size))

    def _call(x: mx.array) -> mx.array:
        out = None
        for start in range(0, intermediate, tile):
            end = min(intermediate, start + tile)
            gate = mx.matmul(x, gate_weight[start:end].T)
            up = mx.matmul(x, up_weight[start:end].T)
            partial = mx.matmul(swiglu(gate, up), down_weight[:, start:end].T)
            out = partial if out is None else out + partial
        if out is None:
            raise RuntimeError("empty MLP tile range")
        return out

    return _call


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--m-values", default="512,1024,2048")
    parser.add_argument("--tile-sizes", default="2048,4352,8704")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.environ["DFLASH_HYBRID_MLP"] = "0"
    os.environ["DFLASH_DEQUANTIZE_MLP"] = "0"
    os.environ["DFLASH_VERIFY_LINEAR"] = "0"

    memory_config = configure_mlx_memory_limits()
    model, _tokenizer, target_meta = load_target_bundle(args.model, lazy=True)
    layer_index, mlp = _first_mlp(model, args.layer)

    bf16_gate, gate_nbytes = _dequantize_linear(mlp.gate_proj)
    bf16_up, up_nbytes = _dequantize_linear(mlp.up_proj)
    bf16_down, down_nbytes = _dequantize_linear(mlp.down_proj)
    mx.eval(bf16_gate.weight, bf16_up.weight, bf16_down.weight)

    def baseline(x: mx.array) -> mx.array:
        return bf16_down(swiglu(bf16_gate(x), bf16_up(x)))

    hidden_size = _linear_input_width(mlp.gate_proj)
    m_values = [int(value) for value in str(args.m_values).split(",") if value.strip()]
    tile_sizes = [
        int(value) for value in str(args.tile_sizes).split(",") if value.strip()
    ]
    rows: dict[str, Any] = {}
    for m_value in m_values:
        x = mx.random.normal((1, int(m_value), hidden_size)).astype(mx.bfloat16)
        mx.eval(x)
        expected = baseline(x)
        mx.eval(expected)
        row: dict[str, Any] = {
            "baseline": _time_ms(
                baseline,
                x,
                repeats=int(args.repeats),
                warmup=int(args.warmup),
            ),
            "tiles": {},
        }
        for tile_size in tile_sizes:
            tiled = _tiled_mlp_fn(
                gate_weight=bf16_gate.weight,
                up_weight=bf16_up.weight,
                down_weight=bf16_down.weight,
                tile_size=int(tile_size),
            )
            actual = tiled(x)
            mx.eval(actual)
            diff = mx.abs(expected - actual)
            timing = _time_ms(
                tiled,
                x,
                repeats=int(args.repeats),
                warmup=int(args.warmup),
            )
            row["tiles"][str(tile_size)] = {
                **timing,
                "max_abs_diff": float(mx.max(diff).item()),
                "baseline_best_speedup": float(row["baseline"]["best_ms"])
                / float(timing["best_ms"]),
            }
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        rows[str(m_value)] = row

    result = {
        "model": str(Path(args.model).expanduser()),
        "layer": int(layer_index),
        "m_values": m_values,
        "tile_sizes": tile_sizes,
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "memory_config": memory_config,
        "target_meta": {
            key: value for key, value in target_meta.items() if key != "config"
        },
        "bf16_weight_nbytes": int(gate_nbytes + up_nbytes + down_nbytes),
        "rows": rows,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
