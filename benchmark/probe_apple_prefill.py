"""Probe Apple/MLX prefill bottlenecks on the real target MLP.

This is intentionally narrow: it benchmarks one actual target MLP over
decode-like and prefill-like token counts. The goal is to verify whether
MLX on Apple Silicon behaves differently for tiny matvec-ish work versus
large prefill matrix-matrix work, and whether q4 dequant overhead is visible.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from dflash_mlx.runtime import (
    _dequantize_linear,
    _target_text_model,
    configure_mlx_memory_limits,
    load_target_bundle,
)


def _time_eval(fn, *, repeats: int, warmup: int) -> dict[str, Any]:
    for _ in range(max(0, warmup)):
        y = fn()
        mx.eval(y)

    times_ms: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter_ns()
        y = fn()
        mx.eval(y)
        times_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    return {
        "best_ms": min(times_ms),
        "median_ms": statistics.median(times_ms),
        "times_ms": times_ms,
    }


def _module_kind(module: Any) -> str:
    if isinstance(module, nn.QuantizedLinear):
        return (
            f"QuantizedLinear(bits={int(module.bits)}, "
            f"group_size={int(module.group_size)})"
        )
    return type(module).__name__


def _linear_input_output_dims(module: Any) -> tuple[int, int]:
    if isinstance(module, nn.QuantizedLinear):
        return (
            int(module.scales.shape[1]) * int(module.group_size),
            int(module.weight.shape[0]),
        )
    return int(module.weight.shape[1]), int(module.weight.shape[0])


def _find_mlp_layer(model: Any, layer_index: int | None) -> tuple[int, Any]:
    text_model = _target_text_model(model)
    if layer_index is not None:
        layer = text_model.layers[int(layer_index)]
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError(f"layer {layer_index} has no mlp")
        return int(layer_index), mlp

    for index, layer in enumerate(text_model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate_proj"):
            return index, mlp
    raise ValueError("could not find an unpacked MLP layer")


def _dequantize_one_mlp(mlp: Any, *, dtype: Any | None = None) -> dict[str, Any]:
    info = {"linears": 0, "weight_nbytes": 0, "attrs": {}}
    eval_weights: list[mx.array] = []
    for attr_name in ("gate_proj", "up_proj", "down_proj"):
        module = getattr(mlp, attr_name, None)
        linear, nbytes = _dequantize_linear(module)
        if dtype is not None and hasattr(linear, "weight"):
            linear.weight = linear.weight.astype(dtype)
        if linear is module:
            info["attrs"][attr_name] = _module_kind(module)
            continue
        setattr(mlp, attr_name, linear)
        eval_weights.append(linear.weight)
        info["linears"] += 1
        info["weight_nbytes"] += nbytes
        info["attrs"][attr_name] = _module_kind(linear)
    if eval_weights:
        mx.eval(*eval_weights)
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    return info


def _cast_one_mlp(mlp: Any, *, dtype: Any) -> dict[str, Any]:
    info = {"dtype": str(dtype), "linears": 0, "weight_nbytes": 0, "attrs": {}}
    eval_weights: list[mx.array] = []
    for attr_name in ("gate_proj", "up_proj", "down_proj"):
        module = getattr(mlp, attr_name, None)
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        module.weight = weight.astype(dtype)
        eval_weights.append(module.weight)
        info["linears"] += 1
        info["weight_nbytes"] += int(module.weight.nbytes)
        info["attrs"][attr_name] = _module_kind(module)
    if eval_weights:
        mx.eval(*eval_weights)
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    return info


def _bench_mlp(
    mlp: Any,
    *,
    hidden_size: int,
    m_values: list[int],
    repeats: int,
    warmup: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in m_values:
        x = mx.random.normal((1, int(m), int(hidden_size))).astype(mx.bfloat16)
        mx.eval(x)

        timing = _time_eval(lambda: mlp(x), repeats=repeats, warmup=warmup)
        best_ms = float(timing["best_ms"])
        median_ms = float(timing["median_ms"])
        rows.append(
            {
                "tokens": int(m),
                "best_ms": best_ms,
                "median_ms": median_ms,
                "best_us_per_token": (best_ms * 1000.0) / float(m),
                "median_us_per_token": (median_ms * 1000.0) / float(m),
                "times_ms": timing["times_ms"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument(
        "--m-values",
        default="1,4,16,64,256,512,1024,2048",
        help="Comma-separated token counts to benchmark through one MLP.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--compiled",
        action="store_true",
        help="Also benchmark mx.compile(lambda x: mlp(x)) for q4 and bf16 MLP paths.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Keep this probe about stock q4 MLX matmul versus explicit bf16 MLP.
    os.environ["DFLASH_PACK_MLP_GATE_UP"] = "0"
    os.environ["DFLASH_DEQUANTIZE_MLP"] = "0"
    os.environ["DFLASH_VERIFY_LINEAR"] = "0"

    memory_config = configure_mlx_memory_limits()
    model, _tokenizer, meta = load_target_bundle(args.model, lazy=True)
    layer_index, mlp = _find_mlp_layer(model, args.layer)

    gate_proj = getattr(mlp, "gate_proj", None)
    up_proj = getattr(mlp, "up_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    hidden_size, intermediate_size = _linear_input_output_dims(gate_proj)

    m_values = [
        int(value.strip())
        for value in str(args.m_values).split(",")
        if value.strip()
    ]

    q4_attrs = {
        "gate_proj": _module_kind(gate_proj),
        "up_proj": _module_kind(up_proj),
        "down_proj": _module_kind(down_proj),
    }
    q4_rows = _bench_mlp(
        mlp,
        hidden_size=hidden_size,
        m_values=m_values,
        repeats=int(args.repeats),
        warmup=int(args.warmup),
    )
    q4_compiled_rows = None
    if args.compiled and hasattr(mx, "compile"):
        q4_compiled = mx.compile(lambda x: mlp(x))
        q4_compiled_rows = _bench_mlp(
            q4_compiled,
            hidden_size=hidden_size,
            m_values=m_values,
            repeats=int(args.repeats),
            warmup=int(args.warmup),
        )

    dequant_info = _dequantize_one_mlp(mlp)
    bf16_rows = _bench_mlp(
        mlp,
        hidden_size=hidden_size,
        m_values=m_values,
        repeats=int(args.repeats),
        warmup=int(args.warmup),
    )
    bf16_compiled_rows = None
    if args.compiled and hasattr(mx, "compile"):
        bf16_compiled = mx.compile(lambda x: mlp(x))
        bf16_compiled_rows = _bench_mlp(
            bf16_compiled,
            hidden_size=hidden_size,
            m_values=m_values,
            repeats=int(args.repeats),
            warmup=int(args.warmup),
        )
    fp16_info = _cast_one_mlp(mlp, dtype=mx.float16)
    fp16_rows = _bench_mlp(
        mlp,
        hidden_size=hidden_size,
        m_values=m_values,
        repeats=int(args.repeats),
        warmup=int(args.warmup),
    )

    by_tokens = {row["tokens"]: row for row in q4_rows}
    fp16_by_tokens = {row["tokens"]: row for row in fp16_rows}
    comparison = []
    for row in bf16_rows:
        q4 = by_tokens[row["tokens"]]
        fp16 = fp16_by_tokens[row["tokens"]]
        comparison.append(
            {
                "tokens": row["tokens"],
                "q4_best_ms": q4["best_ms"],
                "bf16_best_ms": row["best_ms"],
                "fp16_best_ms": fp16["best_ms"],
                "bf16_vs_q4_speedup": q4["best_ms"] / row["best_ms"],
                "fp16_vs_q4_speedup": q4["best_ms"] / fp16["best_ms"],
                "fp16_vs_bf16_speedup": row["best_ms"] / fp16["best_ms"],
                "q4_best_us_per_token": q4["best_us_per_token"],
                "bf16_best_us_per_token": row["best_us_per_token"],
                "fp16_best_us_per_token": fp16["best_us_per_token"],
            }
        )

    result = {
        "model": str(Path(args.model).expanduser()),
        "target_meta": meta,
        "memory_config": memory_config,
        "layer_index": layer_index,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "m_values": m_values,
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "q4_attrs": q4_attrs,
        "dequant_info": dequant_info,
        "fp16_info": fp16_info,
        "q4_rows": q4_rows,
        "q4_compiled_rows": q4_compiled_rows,
        "bf16_rows": bf16_rows,
        "bf16_compiled_rows": bf16_compiled_rows,
        "fp16_rows": fp16_rows,
        "comparison": comparison,
    }

    text = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
