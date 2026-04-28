"""Synchronized layer-component profile for Qwen3 Next prefill.

This intentionally changes scheduling by synchronizing after each component.
Use it for attribution, not absolute throughput.
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

from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from dflash_mlx.runtime import (
    _target_text_model,
    configure_mlx_memory_limits,
    load_target_bundle,
    make_target_cache,
)


PROMPT = (
    "You are profiling a long-context local language model. "
    "Explain the bottlenecks precisely and keep the answer technical.\n"
)


def _target_prompt_tokens(tokenizer: Any, target_tokens: int) -> list[int]:
    pieces: list[str] = []
    tokens: list[int] = []
    while len(tokens) < target_tokens:
        pieces.append(PROMPT)
        tokens = list(tokenizer.encode("".join(pieces)))
    return tokens[:target_tokens]


def _timed_eval(value: mx.array) -> tuple[mx.array, float]:
    start = time.perf_counter_ns()
    mx.eval(value)
    return value, (time.perf_counter_ns() - start) / 1_000.0


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"total_us": 0.0, "median_us": 0.0, "count": 0.0}
    return {
        "total_us": float(sum(values)),
        "median_us": float(statistics.median(values)),
        "count": float(len(values)),
    }


def _kind_component_summary(layer_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    component_keys = (
        "input_norm_us",
        "attention_us",
        "attention_residual_us",
        "post_norm_us",
        "mlp_us",
        "mlp_residual_us",
    )
    for kind in ("linear", "full"):
        kind_rows = [row for row in layer_rows if row.get("kind") == kind]
        for key in component_keys:
            values = [float(row[key]) for row in kind_rows if key in row]
            summaries[f"{kind}_{key[:-3]}"] = _summary(values)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--hybrid-mlp", action="store_true")
    parser.add_argument("--hybrid-mlp-threshold", type=int, default=256)
    parser.add_argument("--hybrid-gdn-proj", action="store_true")
    parser.add_argument("--hybrid-gdn-proj-threshold", type=int, default=256)
    parser.add_argument("--hybrid-gdn-linear", action="store_true")
    parser.add_argument("--hybrid-gdn-linear-threshold", type=int, default=256)
    parser.add_argument("--hybrid-gdn-linear-attrs", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.hybrid_mlp:
        os.environ["DFLASH_HYBRID_MLP"] = "1"
        os.environ["DFLASH_HYBRID_MLP_THRESHOLD"] = str(int(args.hybrid_mlp_threshold))
    if args.hybrid_gdn_proj:
        os.environ["DFLASH_HYBRID_GDN_PROJ"] = "1"
        os.environ["DFLASH_HYBRID_GDN_PROJ_THRESHOLD"] = str(
            int(args.hybrid_gdn_proj_threshold)
        )
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
    model, tokenizer, target_meta = load_target_bundle(args.model, lazy=True)
    inner = _target_text_model(model)

    tokens = _target_prompt_tokens(tokenizer, int(args.prompt_tokens))
    input_ids = mx.array(tokens, dtype=mx.uint32)[None]
    cache = make_target_cache(model, enable_speculative_linear_cache=False)

    component_us: dict[str, list[float]] = {
        "embed": [],
        "input_norm": [],
        "attention": [],
        "attention_residual": [],
        "post_norm": [],
        "mlp": [],
        "mlp_residual": [],
    }
    layer_rows: list[dict[str, Any]] = []

    h, elapsed_us = _timed_eval(inner.embed_tokens(input_ids))
    component_us["embed"].append(elapsed_us)

    fa_mask = create_attention_mask(h, cache[inner.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])

    for layer_index, (layer, layer_cache) in enumerate(
        zip(inner.layers, cache, strict=True)
    ):
        mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
        layer_row: dict[str, Any] = {
            "layer": layer_index,
            "kind": "linear" if getattr(layer, "is_linear", False) else "full",
        }

        normed, elapsed_us = _timed_eval(layer.input_layernorm(h))
        component_us["input_norm"].append(elapsed_us)
        layer_row["input_norm_us"] = elapsed_us

        if getattr(layer, "is_linear", False):
            attn_value = layer.linear_attn(normed, mask, layer_cache)
        else:
            attn_value = layer.self_attn(normed, mask, layer_cache)
        attn_value, elapsed_us = _timed_eval(attn_value)
        component_us["attention"].append(elapsed_us)
        layer_row["attention_us"] = elapsed_us

        h, elapsed_us = _timed_eval(h + attn_value)
        component_us["attention_residual"].append(elapsed_us)
        layer_row["attention_residual_us"] = elapsed_us

        normed, elapsed_us = _timed_eval(layer.post_attention_layernorm(h))
        component_us["post_norm"].append(elapsed_us)
        layer_row["post_norm_us"] = elapsed_us

        mlp_value, elapsed_us = _timed_eval(layer.mlp(normed))
        component_us["mlp"].append(elapsed_us)
        layer_row["mlp_us"] = elapsed_us

        h, elapsed_us = _timed_eval(h + mlp_value)
        component_us["mlp_residual"].append(elapsed_us)
        layer_row["mlp_residual_us"] = elapsed_us
        layer_rows.append(layer_row)

    result = {
        "model": str(Path(args.model).expanduser()),
        "prompt_tokens": int(args.prompt_tokens),
        "hybrid_mlp": bool(args.hybrid_mlp),
        "hybrid_mlp_threshold": int(args.hybrid_mlp_threshold),
        "hybrid_gdn_proj": bool(args.hybrid_gdn_proj),
        "hybrid_gdn_proj_threshold": int(args.hybrid_gdn_proj_threshold),
        "hybrid_gdn_linear": bool(args.hybrid_gdn_linear),
        "hybrid_gdn_linear_threshold": int(args.hybrid_gdn_linear_threshold),
        "hybrid_gdn_linear_attrs": str(args.hybrid_gdn_linear_attrs or ""),
        "memory_config": memory_config,
        "target_meta": target_meta,
        "component_summary_us": {
            name: _summary(values) for name, values in component_us.items()
        },
        "kind_component_summary_us": _kind_component_summary(layer_rows),
        "layer_rows": layer_rows,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
