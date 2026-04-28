"""Benchmark mx.compile around one real Qwen3.5 GDN block.

This is a probe, not production code. The hot server path already uses MLX's
compiled subkernels and Metal recurrent kernel; this checks whether compiling
the larger Python-level GDN block gives MLX any extra scheduling/fusion room on
the M3 Max prefill shape.
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

from dflash_mlx.runtime import (
    _target_text_model,
    configure_mlx_memory_limits,
    load_target_bundle,
)


def _time_eval(fn: Callable[[mx.array], mx.array], x: mx.array, *, repeats: int, warmup: int) -> dict[str, Any]:
    for _ in range(max(0, warmup)):
        mx.eval(fn(x))

    times_ms: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter_ns()
        y = fn(x)
        mx.eval(y)
        times_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return {
        "best_ms": min(times_ms),
        "median_ms": statistics.median(times_ms),
        "times_ms": times_ms,
    }


def _first_gdn_layer(model: Any, layer_index: int | None) -> tuple[int, Any]:
    text_model = _target_text_model(model)
    if layer_index is not None:
        layer = text_model.layers[int(layer_index)]
        attn = getattr(layer, "linear_attn", None)
        if attn is None:
            raise ValueError(f"layer {layer_index} is not a GDN layer")
        return int(layer_index), attn
    for index, layer in enumerate(text_model.layers):
        attn = getattr(layer, "linear_attn", None)
        if attn is not None:
            return index, attn
    raise ValueError("could not find a GDN layer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--tokens", default="512,2048")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--hybrid-gdn-linear", action="store_true")
    parser.add_argument("--hybrid-gdn-linear-threshold", type=int, default=256)
    parser.add_argument("--hybrid-gdn-linear-attrs", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.environ["DFLASH_VERIFY_LINEAR"] = "0"
    if args.hybrid_gdn_linear:
        os.environ["DFLASH_HYBRID_GDN_LINEAR"] = "1"
        os.environ["DFLASH_HYBRID_GDN_LINEAR_THRESHOLD"] = str(
            int(args.hybrid_gdn_linear_threshold)
        )
        if args.hybrid_gdn_linear_attrs:
            os.environ["DFLASH_HYBRID_GDN_LINEAR_ATTRS"] = str(
                args.hybrid_gdn_linear_attrs
            )

    memory_config = configure_mlx_memory_limits()
    model, _tokenizer, target_meta = load_target_bundle(args.model, lazy=True)
    layer_index, attn = _first_gdn_layer(model, args.layer)
    hidden_size = int(getattr(attn, "hidden_size"))
    token_values = [int(value) for value in str(args.tokens).split(",") if value.strip()]

    def uncompiled(x: mx.array) -> mx.array:
        return attn(x, mask=None, cache=None)

    # Shape-specific compile is intentional here. GDN splits depend on static
    # model dimensions, and shapeless compile cannot infer all split outputs.
    # Real prefill also uses a small set of fixed chunk shapes such as 2048.
    compiled = mx.compile(lambda x: attn(x, mask=None, cache=None))

    rows: list[dict[str, Any]] = []
    for tokens in token_values:
        x = mx.random.normal((1, int(tokens), hidden_size)).astype(mx.bfloat16)
        mx.eval(x)
        y_uncompiled = uncompiled(x)
        y_compiled = compiled(x)
        mx.eval(y_uncompiled, y_compiled)
        diff = mx.max(mx.abs(y_uncompiled.astype(mx.float32) - y_compiled.astype(mx.float32)))
        mx.eval(diff)
        uncompiled_timing = _time_eval(
            uncompiled,
            x,
            repeats=int(args.repeats),
            warmup=int(args.warmup),
        )
        compiled_timing = _time_eval(
            compiled,
            x,
            repeats=int(args.repeats),
            warmup=int(args.warmup),
        )
        rows.append(
            {
                "tokens": int(tokens),
                "uncompiled": uncompiled_timing,
                "compiled": compiled_timing,
                "compiled_speedup_best": (
                    float(uncompiled_timing["best_ms"])
                    / float(compiled_timing["best_ms"])
                ),
                "compiled_speedup_median": (
                    float(uncompiled_timing["median_ms"])
                    / float(compiled_timing["median_ms"])
                ),
                "max_abs_diff": float(diff.item()),
            }
        )

    result = {
        "model": str(Path(args.model).expanduser()),
        "layer": int(layer_index),
        "tokens": token_values,
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "hybrid_gdn_linear": bool(args.hybrid_gdn_linear),
        "hybrid_gdn_linear_attrs": str(args.hybrid_gdn_linear_attrs or ""),
        "memory_config": memory_config,
        "target_meta": target_meta,
        "rows": rows,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
