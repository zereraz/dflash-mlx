#!/usr/bin/env python3
"""Compare KV-cache quantization speed and greedy-output drift.

This is intentionally a benchmark/eval harness, not runtime behavior. It loads
the target and draft model once, then runs the same prompt through several cache
variants so we can decide whether q4/q2 KV-cache prefill wins are usable.
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
    configure_full_attention_split,
    configure_mlx_memory_limits,
    generate_dflash_once,
    load_draft_bundle,
    load_target_bundle,
)


DEFAULT_PROMPT = (
    "You are reviewing a long coding conversation. Explain the likely bug, "
    "give the smallest safe patch, and keep the reasoning concrete.\n"
)


def target_prompt_tokens(tokenizer: Any, target_tokens: int, base_prompt: str) -> list[int]:
    if target_tokens <= 0:
        return list(tokenizer.encode(base_prompt))
    pieces: list[str] = []
    tokens: list[int] = []
    while len(tokens) < target_tokens:
        pieces.append(base_prompt)
        tokens = list(tokenizer.encode("".join(pieces)))
    return tokens[:target_tokens]


def decode_text(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return str(tokenizer.decode([int(token_id) for token_id in token_ids]))
    except Exception:
        return ""


def generation_tps(result: dict[str, Any]) -> float:
    elapsed_us = float(result.get("elapsed_us", 0.0))
    prefill_us = float(dict(result.get("phase_timings_us", {})).get("prefill", 0.0))
    decode_us = max(0.0, elapsed_us - prefill_us)
    tokens = int(result.get("generation_tokens", 0) or 0)
    return tokens / (decode_us / 1_000_000.0) if decode_us > 0.0 else 0.0


def common_prefix_len(lhs: list[int], rhs: list[int]) -> int:
    total = min(len(lhs), len(rhs))
    for index in range(total):
        if int(lhs[index]) != int(rhs[index]):
            return index
    return total


def variant_config(name: str) -> tuple[bool, int | None, bool]:
    lowered = name.strip().lower()
    if lowered in {"default", "split"}:
        return False, None, True
    if lowered in {"fp", "fp_no_split", "nosplit"}:
        return False, None, False
    if lowered.startswith("q"):
        return True, int(lowered[1:]), False
    raise ValueError(f"unknown variant {name!r}; expected default, fp_no_split, q8, q4, or q2")


def compact_result(
    *,
    name: str,
    tokenizer: Any,
    result: dict[str, Any],
    default_tokens: list[int] | None,
    fp_no_split_tokens: list[int] | None,
) -> dict[str, Any]:
    token_ids = [int(token_id) for token_id in result.get("generated_token_ids", [])]
    row = {
        "name": name,
        "prefill_ms": float(dict(result.get("phase_timings_us", {})).get("prefill", 0.0))
        / 1_000.0,
        "elapsed_ms": float(result.get("elapsed_us", 0.0)) / 1_000.0,
        "decode_tps": generation_tps(result),
        "acceptance_ratio": float(result.get("acceptance_ratio", 0.0) or 0.0),
        "draft_acceptance_ratio": float(result.get("draft_acceptance_ratio", 0.0) or 0.0),
        "tokens_per_cycle": float(result.get("tokens_per_cycle", 0.0) or 0.0),
        "cycles": int(result.get("cycles_completed", 0) or 0),
        "generated_token_ids": token_ids,
        "generated_text": decode_text(tokenizer, token_ids),
        "quantize_kv_cache": bool(result.get("quantize_kv_cache", False)),
        "kv_cache_bits": int(result.get("kv_cache_bits", 0) or 0),
        "kv_cache_group_size": int(result.get("kv_cache_group_size", 0) or 0),
    }
    if default_tokens is not None:
        prefix = common_prefix_len(default_tokens, token_ids)
        row["common_prefix_vs_default"] = prefix
        row["token_match_ratio_vs_default"] = prefix / max(1, min(len(default_tokens), len(token_ids)))
    if fp_no_split_tokens is not None:
        prefix = common_prefix_len(fp_no_split_tokens, token_ids)
        row["common_prefix_vs_fp_no_split"] = prefix
        row["token_match_ratio_vs_fp_no_split"] = prefix / max(
            1, min(len(fp_no_split_tokens), len(token_ids))
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--prompt-tokens", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--hybrid-mlp", action="store_true")
    parser.add_argument("--hybrid-mlp-threshold", type=int, default=256)
    parser.add_argument("--hybrid-gdn-linear", action="store_true")
    parser.add_argument("--hybrid-gdn-linear-threshold", type=int, default=256)
    parser.add_argument("--hybrid-gdn-linear-attrs", default=None)
    parser.add_argument("--kv-cache-group-size", type=int, default=64, choices=(32, 64, 128))
    parser.add_argument(
        "--variants",
        default="default,fp_no_split,q8,q4,q2",
        help="Comma-separated variants: default, fp_no_split, q8, q4, q2.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    configure_mlx_memory_limits()
    if args.hybrid_mlp:
        os.environ["DFLASH_HYBRID_MLP"] = "1"
        os.environ["DFLASH_HYBRID_MLP_THRESHOLD"] = str(int(args.hybrid_mlp_threshold))
    if args.hybrid_gdn_linear:
        os.environ["DFLASH_HYBRID_GDN_LINEAR"] = "1"
        os.environ["DFLASH_HYBRID_GDN_LINEAR_THRESHOLD"] = str(
            int(args.hybrid_gdn_linear_threshold)
        )
        if args.hybrid_gdn_linear_attrs:
            os.environ["DFLASH_HYBRID_GDN_LINEAR_ATTRS"] = str(
                args.hybrid_gdn_linear_attrs
            )
    prompt_text = args.prompt
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).expanduser().read_text(encoding="utf-8")

    target_model, tokenizer, _ = load_target_bundle(
        args.model,
        lazy=True,
        split_full_attention_sdpa=True,
    )
    draft_model, _ = load_draft_bundle(args.draft, lazy=True)
    prompt_tokens = target_prompt_tokens(tokenizer, int(args.prompt_tokens), prompt_text)

    rows: list[dict[str, Any]] = []
    default_tokens: list[int] | None = None
    fp_no_split_tokens: list[int] | None = None
    variants = [name.strip() for name in args.variants.split(",") if name.strip()]

    started = time.perf_counter()
    for name in variants:
        quantize_kv, bits, split_sdpa = variant_config(name)
        configure_full_attention_split(target_model, enabled=split_sdpa)
        result = generate_dflash_once(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            prompt=prompt_text,
            max_new_tokens=int(args.max_tokens),
            use_chat_template=False,
            block_tokens=int(args.block_tokens),
            prompt_tokens_override=prompt_tokens,
            quantize_kv_cache=quantize_kv,
            kv_cache_bits=int(bits or 8),
            kv_cache_group_size=int(args.kv_cache_group_size),
            prefill_step_size=int(args.prefill_step_size),
        )
        row = compact_result(
            name=name,
            tokenizer=tokenizer,
            result=result,
            default_tokens=default_tokens,
            fp_no_split_tokens=fp_no_split_tokens,
        )
        rows.append(row)
        token_ids = list(row["generated_token_ids"])
        if name.strip().lower() in {"default", "split"} and default_tokens is None:
            default_tokens = token_ids
        if name.strip().lower() in {"fp", "fp_no_split", "nosplit"} and fp_no_split_tokens is None:
            fp_no_split_tokens = token_ids
        print(
            f"{name:>12s} prefill={row['prefill_ms']:.1f}ms "
            f"decode={row['decode_tps']:.1f} tok/s "
            f"accept={row['acceptance_ratio'] * 100:.1f}% "
            f"prefix_default={row.get('common_prefix_vs_default', '-')}",
            flush=True,
        )
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    report = {
        "model": args.model,
        "draft": args.draft,
        "prompt_tokens": len(prompt_tokens),
        "max_tokens": int(args.max_tokens),
        "prefill_step_size": int(args.prefill_step_size),
        "block_tokens": int(args.block_tokens),
        "hybrid_mlp": bool(args.hybrid_mlp),
        "hybrid_mlp_threshold": int(args.hybrid_mlp_threshold),
        "hybrid_gdn_linear": bool(args.hybrid_gdn_linear),
        "hybrid_gdn_linear_threshold": int(args.hybrid_gdn_linear_threshold),
        "hybrid_gdn_linear_attrs": str(args.hybrid_gdn_linear_attrs or ""),
        "kv_cache_group_size": int(args.kv_cache_group_size),
        "variants": variants,
        "elapsed_s": time.perf_counter() - started,
        "rows": rows,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
