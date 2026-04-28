#!/usr/bin/env python3
"""Compare default fp32 GDN state with an experimental lower-precision state.

The GDN kernel accumulates each chunk in float, so a same-chunk logit check can
miss state dtype drift. This benchmark forces chunk boundaries, then compares
the next-token logits after the cached prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from dflash_mlx.runtime import (
    configure_mlx_memory_limits,
    load_target_bundle,
    make_target_cache,
    target_forward_with_hidden_states,
    target_prefill_without_logits,
)


PROMPT = (
    "You are checking recurrent-state precision in a hybrid Qwen prefill path. "
    "Keep the text deterministic and long enough to cross cache chunks.\n"
)


def _target_prompt_tokens(tokenizer: Any, target_tokens: int) -> list[int]:
    pieces: list[str] = []
    tokens: list[int] = []
    while len(tokens) < target_tokens:
        pieces.append(PROMPT)
        tokens = list(tokenizer.encode("".join(pieces)))
    return tokens[:target_tokens]


def _set_common_hybrid_env(args: argparse.Namespace) -> None:
    os.environ["DFLASH_HYBRID_MLP"] = "1"
    os.environ["DFLASH_HYBRID_MLP_THRESHOLD"] = str(int(args.threshold))
    if args.hybrid_gdn_linear:
        os.environ["DFLASH_HYBRID_GDN_LINEAR"] = "1"
        os.environ["DFLASH_HYBRID_GDN_LINEAR_THRESHOLD"] = str(int(args.threshold))
        os.environ["DFLASH_HYBRID_GDN_LINEAR_ATTRS"] = str(args.hybrid_gdn_linear_attrs)
    else:
        os.environ.pop("DFLASH_HYBRID_GDN_LINEAR", None)
        os.environ.pop("DFLASH_HYBRID_GDN_LINEAR_ATTRS", None)


def _next_logits_after_chunked_prefix(
    model: Any,
    *,
    prefix_tokens: list[int],
    next_token: int,
    prefill_step_size: int,
) -> tuple[mx.array, float]:
    cache = make_target_cache(model, enable_speculative_linear_cache=False)
    start_ns = time.perf_counter_ns()
    for start in range(0, len(prefix_tokens), int(prefill_step_size)):
        chunk = prefix_tokens[start : start + int(prefill_step_size)]
        chunk_ids = mx.array(chunk, dtype=mx.uint32)[None]
        h = target_prefill_without_logits(
            model,
            input_ids=chunk_ids,
            cache=cache,
            skip_final_layer_mlp=False,
            skip_final_layer_attention=False,
        )
        mx.eval(h)
    prefill_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0

    next_ids = mx.array([int(next_token)], dtype=mx.uint32)[None]
    logits, _ = target_forward_with_hidden_states(
        model,
        input_ids=next_ids,
        cache=cache,
        capture_layer_ids=set(),
        last_logits_only=True,
    )
    if logits is None:
        raise RuntimeError("target forward unexpectedly returned no logits")
    mx.eval(logits)
    return logits[:, -1, :], prefill_ms


def _load_and_run(
    args: argparse.Namespace,
    prompt_tokens: list[int],
    *,
    state_dtype: str | None,
) -> tuple[mx.array, float, dict[str, Any]]:
    _set_common_hybrid_env(args)
    if state_dtype:
        os.environ["DFLASH_GDN_STATE_DTYPE"] = state_dtype
    else:
        os.environ.pop("DFLASH_GDN_STATE_DTYPE", None)
    model, _tokenizer, meta = load_target_bundle(args.model, lazy=True)
    return (
        *_next_logits_after_chunked_prefix(
            model,
            prefix_tokens=prompt_tokens[:-1],
            next_token=prompt_tokens[-1],
            prefill_step_size=int(args.prefill_step_size),
        ),
        meta,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--threshold", type=int, default=256)
    parser.add_argument("--state-dtype", choices=("bf16", "bfloat16", "fp16", "float16"), default="bf16")
    parser.add_argument("--hybrid-gdn-linear", action="store_true")
    parser.add_argument(
        "--hybrid-gdn-linear-attrs",
        default="in_proj_qkv,in_proj_z,out_proj",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    memory_config = configure_mlx_memory_limits()
    os.environ.pop("DFLASH_GDN_STATE_DTYPE", None)
    model, tokenizer, _ = load_target_bundle(args.model, lazy=True)
    prompt_tokens = _target_prompt_tokens(tokenizer, int(args.prompt_tokens) + 1)
    del model
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()

    control_logits, control_ms, control_meta = _load_and_run(
        args,
        prompt_tokens,
        state_dtype=None,
    )
    experiment_logits, experiment_ms, experiment_meta = _load_and_run(
        args,
        prompt_tokens,
        state_dtype=str(args.state_dtype),
    )

    diff = mx.abs(control_logits - experiment_logits)
    mx.eval(diff)
    control_top = mx.argsort(control_logits, axis=-1)[:, -10:]
    experiment_top = mx.argsort(experiment_logits, axis=-1)[:, -10:]
    control_top_list = [int(x) for x in control_top.reshape(-1).tolist()]
    experiment_top_list = [int(x) for x in experiment_top.reshape(-1).tolist()]
    result = {
        "model": str(Path(args.model).expanduser()),
        "prompt_tokens": len(prompt_tokens) - 1,
        "prefill_step_size": int(args.prefill_step_size),
        "state_dtype": str(args.state_dtype),
        "hybrid_gdn_linear": bool(args.hybrid_gdn_linear),
        "hybrid_gdn_linear_attrs": str(args.hybrid_gdn_linear_attrs),
        "memory_config": memory_config,
        "control_prefill_ms": control_ms,
        "experiment_prefill_ms": experiment_ms,
        "speedup": control_ms / experiment_ms if experiment_ms > 0 else 0.0,
        "max_abs_logit_diff": float(mx.max(diff).item()),
        "mean_abs_logit_diff": float(mx.mean(diff).item()),
        "control_argmax": int(mx.argmax(control_logits, axis=-1).item()),
        "experiment_argmax": int(mx.argmax(experiment_logits, axis=-1).item()),
        "top10_overlap": len(set(control_top_list) & set(experiment_top_list)),
        "control_meta": {k: v for k, v in control_meta.items() if k != "config"},
        "experiment_meta": {k: v for k, v in experiment_meta.items() if k != "config"},
    }
    text = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
