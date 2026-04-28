#!/usr/bin/env python3
"""Profile DFlash runtime variants without reloading models between runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx

from dflash_mlx.runtime import (
    configure_mlx_memory_limits,
    generate_dflash_once,
    load_draft_bundle,
    load_target_bundle,
    stream_dflash_generate,
)


DEFAULT_PROMPT = (
    "You are solving a systems performance problem. Explain each step, track "
    "important numbers, and keep the final answer concise.\n"
)


SCENARIOS: dict[str, dict[str, Any]] = {
    "auto_b16": {"env": {"DFLASH_VERIFY_VARIANT": "auto"}, "block_tokens": 16},
    "adaptive_b16": {
        "env": {
            "DFLASH_VERIFY_VARIANT": "auto",
            "DFLASH_ADAPTIVE_FALLBACK": "1",
        },
        "block_tokens": 16,
    },
    "prefill_clear_b16": {
        "env": {
            "DFLASH_VERIFY_VARIANT": "auto",
            "DFLASH_CLEAR_CACHE_AFTER_PREFILL": "1",
        },
        "block_tokens": 16,
    },
    "no_prefill_clear_b16": {
        "env": {
            "DFLASH_VERIFY_VARIANT": "auto",
            "DFLASH_CLEAR_CACHE_AFTER_PREFILL": "0",
        },
        "block_tokens": 16,
    },
    "fused_draft_kv_b16": {
        "env": {
            "DFLASH_VERIFY_VARIANT": "auto",
            "DFLASH_FUSED_DRAFT_CONTEXT_KV": "1",
        },
        "block_tokens": 16,
    },
    "draft_kv_loop_b16": {
        "env": {
            "DFLASH_VERIFY_VARIANT": "auto",
            "DFLASH_FUSED_DRAFT_CONTEXT_KV": "0",
        },
        "block_tokens": 16,
    },
    "no_defer_b16": {
        "env": {
            "DFLASH_VERIFY_VARIANT": "auto",
            "DFLASH_PREFILL_DEFER_DRAFT_CONTEXT": "0",
        },
        "block_tokens": 16,
    },
    "full_logits_b16": {
        "env": {"DFLASH_VERIFY_VARIANT": "auto", "DFLASH_LM_HEAD_ARGMAX": "0"},
        "block_tokens": 16,
    },
    "prefill_full_logits_b16": {
        "env": {
            "DFLASH_VERIFY_VARIANT": "auto",
            "DFLASH_PREFILL_LAST_LOGITS_ONLY": "0",
        },
        "block_tokens": 16,
    },
    "auto_kp4_b16": {
        "env": {"DFLASH_VERIFY_VARIANT": "auto", "DFLASH_VERIFY_AUTO_KPARTS": "4"},
        "block_tokens": 16,
    },
    "auto_kp16_b16": {
        "env": {"DFLASH_VERIFY_VARIANT": "auto", "DFLASH_VERIFY_AUTO_KPARTS": "16"},
        "block_tokens": 16,
    },
    "mma_b16": {"env": {"DFLASH_VERIFY_VARIANT": "mma2big"}, "block_tokens": 16},
    "pipe4_b16": {
        "env": {"DFLASH_VERIFY_VARIANT": "mma2big_pipe", "DFLASH_VERIFY_QMM_KPARTS": "4"},
        "block_tokens": 16,
    },
    "pipe8_b16": {
        "env": {"DFLASH_VERIFY_VARIANT": "mma2big_pipe", "DFLASH_VERIFY_QMM_KPARTS": "8"},
        "block_tokens": 16,
    },
    "auto_b8": {"env": {"DFLASH_VERIFY_VARIANT": "auto"}, "block_tokens": 8},
    "auto_b4": {"env": {"DFLASH_VERIFY_VARIANT": "auto"}, "block_tokens": 4},
    "auto_b12": {"env": {"DFLASH_VERIFY_VARIANT": "auto"}, "block_tokens": 12},
    "argmax_b16": {
        "env": {"DFLASH_VERIFY_VARIANT": "auto", "DFLASH_LM_HEAD_ARGMAX": "1"},
        "block_tokens": 16,
    },
    "verify8_b16": {
        "env": {"DFLASH_VERIFY_VARIANT": "auto", "DFLASH_VERIFY_LEN": "8"},
        "block_tokens": 16,
    },
    "verify12_b16": {
        "env": {"DFLASH_VERIFY_VARIANT": "auto", "DFLASH_VERIFY_LEN": "12"},
        "block_tokens": 16,
    },
    # SDPA block-count sweeps are intentionally exposed because the default
    # applegpu_g15s/M3 Max heuristic may not be optimal on every Apple GPU.
    "sdpa64_b16": {
        "env": {"DFLASH_VERIFY_VARIANT": "auto", "DFLASH_SDPA_2PASS_BLOCKS": "64"},
        "block_tokens": 16,
    },
    "sdpa128_b16": {
        "env": {"DFLASH_VERIFY_VARIANT": "auto", "DFLASH_SDPA_2PASS_BLOCKS": "128"},
        "block_tokens": 16,
    },
    "sdpa256_b16": {
        "env": {"DFLASH_VERIFY_VARIANT": "auto", "DFLASH_SDPA_2PASS_BLOCKS": "256"},
        "block_tokens": 16,
    },
    "fastpath_b16": {
        "env": {"DFLASH_VERIFY_VARIANT": "auto", "DFLASH_PREFILL_CACHE_FASTPATH": "1"},
        "block_tokens": 16,
    },
    "fastpath_skip_b16": {
        "env": {
            "DFLASH_VERIFY_VARIANT": "auto",
            "DFLASH_PREFILL_CACHE_FASTPATH": "1",
            "DFLASH_PREFILL_SKIP_CAPTURE": "1",
        },
        "block_tokens": 16,
    },
    "defer_ctx_b16": {
        "env": {
            "DFLASH_VERIFY_VARIANT": "auto",
            "DFLASH_PREFILL_DEFER_DRAFT_CONTEXT": "1",
        },
        "block_tokens": 16,
    },
}


@contextmanager
def patched_env(values: dict[str, str]) -> Iterator[None]:
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _target_prompt_tokens(tokenizer: Any, target_tokens: int, base_prompt: str) -> list[int]:
    if target_tokens <= 0:
        return list(tokenizer.encode(base_prompt))

    pieces: list[str] = []
    tokens: list[int] = []
    while len(tokens) < target_tokens:
        pieces.append(base_prompt)
        tokens = list(tokenizer.encode("".join(pieces)))
    return tokens[:target_tokens]


def _phase_ms(result: dict[str, Any], key: str) -> float:
    return float(dict(result.get("phase_timings_us", {})).get(key, 0.0)) / 1_000.0


def _decode_tps(result: dict[str, Any]) -> float:
    elapsed_us = float(result.get("elapsed_us", 0.0))
    prefill_us = float(dict(result.get("phase_timings_us", {})).get("prefill", 0.0))
    generation_us = max(0.0, elapsed_us - prefill_us)
    tokens = int(result.get("generation_tokens", 0))
    return tokens / (generation_us / 1_000_000.0) if generation_us > 0.0 else 0.0


def _profile_totals_ms(result: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value) / 1_000.0
        for key, value in dict(result.get("cycle_profile_totals_us", {})).items()
    }


def _mlx_memory_snapshot() -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for key, attr in (
        ("active_gb", "get_active_memory"),
        ("cache_gb", "get_cache_memory"),
        ("peak_gb", "get_peak_memory"),
    ):
        fn = getattr(mx, attr, None)
        if not callable(fn):
            continue
        try:
            snapshot[key] = float(fn()) / 1e9
        except Exception:
            pass
    return snapshot


def _filtered_env() -> dict[str, str]:
    prefixes = ("DFLASH_", "MLX_", "MTL_")
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(prefixes)
    }


def _device_metadata() -> dict[str, Any]:
    if not mx.metal.is_available():
        return {"metal": False}
    try:
        info = dict(mx.device_info())
    except Exception as exc:
        return {"metal": True, "device_info_error": str(exc)}
    return {
        "metal": True,
        **{
            str(key): value
            for key, value in info.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
    }


def _compact_result(
    name: str,
    run_index: int,
    result: dict[str, Any],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "run": int(run_index),
        "block_tokens": int(result.get("block_tokens", scenario.get("block_tokens", 0)) or 0),
        "prefill_step_size": int(result.get("prefill_step_size", 0) or 0),
        "quantize_kv_cache": bool(result.get("quantize_kv_cache", False)),
        "kv_cache_bits": int(result.get("kv_cache_bits", 0) or 0),
        "kv_cache_group_size": int(result.get("kv_cache_group_size", 0) or 0),
        "prompt_tokens": int(result.get("prompt_token_count", 0) or 0),
        "generation_tokens": int(result.get("generation_tokens", 0) or 0),
        "cache_only_prefill": bool(result.get("cache_only_prefill", False)),
        "returned_prompt_cache": "prompt_cache" in result,
        "elapsed_ms": float(result.get("elapsed_us", 0.0)) / 1_000.0,
        "prefill_ms": _phase_ms(result, "prefill"),
        "draft_ms": _phase_ms(result, "draft"),
        "draft_prefill_ms": _phase_ms(result, "draft_prefill"),
        "verify_ms": _phase_ms(result, "verify"),
        "verify_linear_install_ms": _phase_ms(result, "verify_linear_install"),
        "verify_linear_swapped": int(result.get("verify_linear_swapped", 0) or 0),
        "replay_ms": _phase_ms(result, "replay"),
        "commit_ms": _phase_ms(result, "commit"),
        "decode_tps": _decode_tps(result),
        "acceptance_ratio": float(result.get("acceptance_ratio", 0.0) or 0.0),
        "draft_acceptance_ratio": float(
            result.get("draft_acceptance_ratio", 0.0) or 0.0
        ),
        "draft_tokens_attempted": int(result.get("draft_tokens_attempted", 0) or 0),
        "acceptance_position_rates": list(result.get("acceptance_position_rates", [])),
        "tokens_per_cycle": float(result.get("tokens_per_cycle", 0.0) or 0.0),
        "cycles": int(result.get("cycles_completed", 0) or 0),
        "adaptive_fallback_ar": bool(result.get("adaptive_fallback_ar", False)),
        "adaptive_fallback_count": int(result.get("adaptive_fallback_count", 0) or 0),
        "prompt_cache_checkpoint_tokens": int(
            result.get("prompt_cache_checkpoint_tokens", 0) or 0
        ),
        "prompt_cache_checkpoints": int(
            result.get("prompt_cache_checkpoints", 0) or 0
        ),
        "prompt_cache_checkpoint_ms": float(
            result.get("prompt_cache_checkpoint_us", 0.0) or 0.0
        )
        / 1_000.0,
        "prompt_cache_target_only_checkpoints": bool(
            result.get("prompt_cache_target_only_checkpoints", False)
        ),
        "profile_totals_ms": _profile_totals_ms(result),
        "prefill_chunk_profile_us": list(result.get("prefill_chunk_profile_us", [])),
    }


def _print_row(row: dict[str, Any]) -> None:
    print(
        f"{row['name']:>12s} run={row['run']} "
        f"prefill={row['prefill_ms']:.1f}ms "
        f"decode={row['decode_tps']:.2f} tok/s "
        f"accept={row['acceptance_ratio'] * 100:.1f}% "
        f"draft_accept={row['draft_acceptance_ratio'] * 100:.1f}% "
        f"cache={'yes' if row['returned_prompt_cache'] else 'no'} "
        f"ckpt={row['prompt_cache_checkpoints']} "
        f"fallback={row['adaptive_fallback_count']} "
        f"draft={row['draft_ms']:.1f}ms verify={row['verify_ms']:.1f}ms "
        f"cycles={row['cycles']}",
        flush=True,
    )


def _consume_stream_summary(iterator: Iterator[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] | None = None
    for event in iterator:
        if event.get("event") == "summary":
            summary = event
    if summary is None:
        raise RuntimeError("stream_dflash_generate did not produce a summary event")
    return summary


def _normalize_capture_path(path: str | Path) -> Path:
    capture_path = Path(path).expanduser()
    if capture_path.suffix != ".gputrace":
        capture_path = Path(str(capture_path) + ".gputrace")
    return capture_path


def _unique_capture_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not find a free capture path for {path}")


def _capture_path_for_run(
    base_path: str,
    *,
    scenario_name: str,
    run_index: int,
    scenario_count: int,
    repeat_count: int,
) -> Path:
    path = _normalize_capture_path(base_path)
    if scenario_count > 1 or repeat_count > 1:
        path = path.with_name(f"{path.stem}-{scenario_name}-run{run_index}{path.suffix}")
    return _unique_capture_path(path)


def _run_with_metal_capture(path: Path | None, fn):
    if path is None:
        return fn()

    path.parent.mkdir(parents=True, exist_ok=True)
    if os.environ.get("MTL_CAPTURE_ENABLED", "").strip() != "1":
        sys.stderr.write(
            "warning: MLX Metal capture usually requires launching with "
            "MTL_CAPTURE_ENABLED=1\n"
        )
        sys.stderr.flush()

    sys.stderr.write(f"capturing MLX Metal trace to {path}\n")
    sys.stderr.flush()
    started = False
    try:
        mx.metal.start_capture(str(path))
        started = True
        return fn()
    finally:
        if started:
            mx.metal.stop_capture()
            sys.stderr.write(f"wrote MLX Metal trace to {path}\n")
            sys.stderr.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Read benchmark prompt text from a file before truncating to --prompt-tokens.",
    )
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument(
        "--prefill-step-sizes",
        default=None,
        help="Comma-separated prefill step sizes to sweep using the same loaded models.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--scenarios",
        default="auto_b16,mma_b16,pipe4_b16,pipe8_b16,auto_b8,fastpath_b16",
        help=f"Comma-separated scenario names. Available: {', '.join(SCENARIOS)}",
    )
    parser.add_argument("--profile", action="store_true", help="Collect DFLASH_PROFILE cycle timings.")
    parser.add_argument(
        "--profile-prefill",
        action="store_true",
        help=(
            "Collect per-prefill-chunk path timings and MLX memory snapshots "
            "via DFLASH_PROFILE_PREFILL."
        ),
    )
    parser.add_argument(
        "--split-sdpa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable split full-attention SDPA hooks when loading the target model.",
    )
    parser.add_argument(
        "--prefill-split-sdpa-threshold",
        type=int,
        default=0,
        help=(
            "When >0, split full-attention SDPA for prefill chunks whose "
            "query length is at least this many tokens. This is an Apple "
            "unified-memory scheduling experiment for large prefill chunks."
        ),
    )
    parser.add_argument(
        "--prefill-split-sdpa-chunk",
        type=int,
        default=2048,
        help="Query tile size for --prefill-split-sdpa-threshold.",
    )
    parser.add_argument(
        "--quantize-kv-cache",
        action="store_true",
        help="Use quantized target full-attention KV cache during DFlash runs.",
    )
    parser.add_argument(
        "--mlx-cache-fraction",
        type=float,
        default=None,
        help=(
            "Set DFLASH_MLX_CACHE_FRACTION before configuring MLX memory. "
            "Useful for Apple unified-memory/cache-pressure sweeps."
        ),
    )
    parser.add_argument(
        "--hybrid-mlp",
        action="store_true",
        help=(
            "Set DFLASH_HYBRID_MLP=1 before loading the target. This keeps q4 "
            "for decode/small verify and uses bf16 MLP weights for large prefill chunks."
        ),
    )
    parser.add_argument(
        "--hybrid-mlp-threshold",
        type=int,
        default=256,
        help="Effective linear row count where --hybrid-mlp switches from q4 to bf16.",
    )
    parser.add_argument(
        "--hybrid-gdn-proj",
        action="store_true",
        help=(
            "Set DFLASH_HYBRID_GDN_PROJ=1 before loading the target. This "
            "uses packed GDN qkv+z and b+a projections for large prefill chunks."
        ),
    )
    parser.add_argument(
        "--hybrid-gdn-proj-threshold",
        type=int,
        default=256,
        help="Effective linear row count where --hybrid-gdn-proj switches to packed GDN inputs.",
    )
    parser.add_argument(
        "--hybrid-gdn-linear",
        action="store_true",
        help=(
            "Set DFLASH_HYBRID_GDN_LINEAR=1 before loading the target. This "
            "uses bf16 GDN projection linears for large prefill chunks."
        ),
    )
    parser.add_argument(
        "--hybrid-gdn-linear-threshold",
        type=int,
        default=256,
        help="Effective linear row count where --hybrid-gdn-linear switches to bf16 GDN linears.",
    )
    parser.add_argument(
        "--hybrid-gdn-linear-attrs",
        default=None,
        help=(
            "Comma-separated GDN projections for --hybrid-gdn-linear. "
            "Known: in_proj_qkv,in_proj_z,out_proj."
        ),
    )
    parser.add_argument(
        "--kv-cache-bits",
        type=int,
        default=8,
        choices=(2, 4, 8),
        help="Bits for --quantize-kv-cache.",
    )
    parser.add_argument(
        "--kv-cache-group-size",
        type=int,
        default=64,
        choices=(32, 64, 128),
        help="Quantization group size for --quantize-kv-cache.",
    )
    parser.add_argument(
        "--return-prompt-cache",
        action="store_true",
        help=(
            "Use the streaming path and export the DFlash prompt cache. "
            "This matches the server's --dflash-prompt-cache cache-build cost."
        ),
    )
    parser.add_argument(
        "--middle-no-logits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Control DFLASH_PREFILL_MIDDLE_NO_LOGITS. When enabled, chunks "
            "that do not need draft context features use target_prefill_without_logits."
        ),
    )
    parser.add_argument(
        "--slice-capture",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Control DFLASH_PREFILL_SLICE_CAPTURE. When enabled, retained "
            "draft-context hidden states are captured as the smallest token slice."
        ),
    )
    parser.add_argument(
        "--skip-final-mlp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Control DFLASH_PREFILL_SKIP_FINAL_MLP. When enabled, cache-only "
            "prefill chunks skip the final layer MLP because no later cache "
            "or logits consume that hidden output."
        ),
    )
    parser.add_argument(
        "--skip-final-attention",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Control DFLASH_PREFILL_SKIP_FINAL_ATTENTION. Cache-only prefill "
            "chunks update the final full-attention layer K/V cache without "
            "computing final-layer Q/SDPA/O/MLP."
        ),
    )
    parser.add_argument(
        "--prompt-cache-checkpoint-tokens",
        type=int,
        default=0,
        help=(
            "When used with --return-prompt-cache, emit reusable DFlash prompt "
            "cache checkpoints about every N tokens."
        ),
    )
    parser.add_argument(
        "--target-only-checkpoints",
        action="store_true",
        help=(
            "With --return-prompt-cache checkpoints, save exact target-cache "
            "checkpoints but omit persisted draft-context K/V. This keeps dense "
            "prefix reuse cheap; decode after a mid-checkpoint may draft worse "
            "until fresh suffix context is added."
        ),
    )
    parser.add_argument(
        "--reuse-suffix-tokens",
        type=int,
        default=0,
        help=(
            "After a --return-prompt-cache run, immediately reuse that cache "
            "with this many additional prompt tokens."
        ),
    )
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop launching new scenario runs after this wall-clock budget. 0 disables the guard.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--metal-capture",
        default=None,
        help=(
            "Write an MLX Metal .gputrace around the real generate call. "
            "Launch with MTL_CAPTURE_ENABLED=1."
        ),
    )
    args = parser.parse_args()

    if args.mlx_cache_fraction is not None:
        os.environ["DFLASH_MLX_CACHE_FRACTION"] = str(float(args.mlx_cache_fraction))
    mlx_memory_config = configure_mlx_memory_limits()

    if args.profile:
        os.environ["DFLASH_PROFILE"] = "1"
    if args.profile_prefill:
        os.environ["DFLASH_PROFILE_PREFILL"] = "1"
    if args.middle_no_logits is not None:
        os.environ["DFLASH_PREFILL_MIDDLE_NO_LOGITS"] = (
            "1" if args.middle_no_logits else "0"
        )
    if args.slice_capture is not None:
        os.environ["DFLASH_PREFILL_SLICE_CAPTURE"] = "1" if args.slice_capture else "0"
    if args.skip_final_mlp is not None:
        os.environ["DFLASH_PREFILL_SKIP_FINAL_MLP"] = (
            "1" if args.skip_final_mlp else "0"
        )
    if args.skip_final_attention is not None:
        os.environ["DFLASH_PREFILL_SKIP_FINAL_ATTENTION"] = (
            "1" if args.skip_final_attention else "0"
        )
    if int(args.prefill_split_sdpa_threshold) > 0:
        os.environ["DFLASH_PREFILL_SPLIT_SDPA_THRESHOLD"] = str(
            int(args.prefill_split_sdpa_threshold)
        )
        os.environ["DFLASH_PREFILL_SPLIT_SDPA_CHUNK"] = str(
            max(1, int(args.prefill_split_sdpa_chunk))
        )
    if args.hybrid_mlp:
        if args.hybrid_mlp_threshold <= 0:
            raise SystemExit("--hybrid-mlp-threshold must be > 0")
        os.environ["DFLASH_HYBRID_MLP"] = "1"
        os.environ["DFLASH_HYBRID_MLP_THRESHOLD"] = str(args.hybrid_mlp_threshold)
    if args.hybrid_gdn_proj:
        if args.hybrid_gdn_proj_threshold <= 0:
            raise SystemExit("--hybrid-gdn-proj-threshold must be > 0")
        os.environ["DFLASH_HYBRID_GDN_PROJ"] = "1"
        os.environ["DFLASH_HYBRID_GDN_PROJ_THRESHOLD"] = str(
            args.hybrid_gdn_proj_threshold
        )
    if args.hybrid_gdn_linear:
        if args.hybrid_gdn_linear_threshold <= 0:
            raise SystemExit("--hybrid-gdn-linear-threshold must be > 0")
        os.environ["DFLASH_HYBRID_GDN_LINEAR"] = "1"
        os.environ["DFLASH_HYBRID_GDN_LINEAR_THRESHOLD"] = str(
            args.hybrid_gdn_linear_threshold
        )
        if args.hybrid_gdn_linear_attrs:
            os.environ["DFLASH_HYBRID_GDN_LINEAR_ATTRS"] = str(
                args.hybrid_gdn_linear_attrs
            )
    prompt_text = args.prompt
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).expanduser().read_text(encoding="utf-8")

    target_model, tokenizer, target_meta = load_target_bundle(
        args.model,
        lazy=True,
        split_full_attention_sdpa=bool(args.split_sdpa),
        quantize_kv_cache=bool(args.quantize_kv_cache),
        kv_cache_bits=int(args.kv_cache_bits),
        kv_cache_group_size=int(args.kv_cache_group_size),
    )
    draft_model, _ = load_draft_bundle(args.draft, lazy=True)
    prompt_tokens = _target_prompt_tokens(tokenizer, args.prompt_tokens, prompt_text)
    reuse_suffix_tokens: list[int] = []
    if int(args.reuse_suffix_tokens) > 0:
        reuse_prompt_tokens = _target_prompt_tokens(
            tokenizer,
            args.prompt_tokens + int(args.reuse_suffix_tokens),
            prompt_text,
        )
        reuse_suffix_tokens = reuse_prompt_tokens[len(prompt_tokens) :]
    scenario_names = [name.strip() for name in args.scenarios.split(",") if name.strip()]
    if args.prefill_step_sizes:
        prefill_step_sizes = [
            int(value.strip())
            for value in args.prefill_step_sizes.split(",")
            if value.strip()
        ]
    else:
        prefill_step_sizes = [int(args.prefill_step_size)]
    repeat_count = max(1, int(args.repeat))
    max_seconds = max(0.0, float(args.max_seconds))
    benchmark_start = time.perf_counter()

    rows: list[dict[str, Any]] = []
    stopped_by_budget = False
    for run_index in range(1, repeat_count + 1):
        for prefill_step_size in prefill_step_sizes:
            for name in scenario_names:
                row_name = (
                    f"{name}_step{prefill_step_size}"
                    if len(prefill_step_sizes) > 1
                    else name
                )
                elapsed_seconds = time.perf_counter() - benchmark_start
                if max_seconds > 0.0 and elapsed_seconds >= max_seconds:
                    sys.stderr.write(
                        f"stopping profile_variants after {elapsed_seconds:.1f}s "
                        f"(--max-seconds {max_seconds:.1f})\n"
                    )
                    sys.stderr.flush()
                    stopped_by_budget = True
                    break
                if name not in SCENARIOS:
                    raise ValueError(f"unknown scenario '{name}'")
                scenario = SCENARIOS[name]
                env = dict(scenario.get("env") or {})
                with patched_env(env):
                    row_start = time.perf_counter()
                    row_memory_before = _mlx_memory_snapshot()
                    capture_path = (
                        _capture_path_for_run(
                            args.metal_capture,
                            scenario_name=row_name,
                            run_index=run_index,
                            scenario_count=len(scenario_names) * len(prefill_step_sizes),
                            repeat_count=repeat_count,
                        )
                        if args.metal_capture
                        else None
                    )
                    run_kwargs = {
                        "target_model": target_model,
                        "tokenizer": tokenizer,
                        "draft_model": draft_model,
                        "prompt": prompt_text,
                        "max_new_tokens": int(args.max_tokens),
                        "use_chat_template": False,
                        "block_tokens": int(scenario.get("block_tokens", 16)),
                        "prompt_tokens_override": prompt_tokens,
                        "quantize_kv_cache": bool(args.quantize_kv_cache),
                        "kv_cache_bits": int(args.kv_cache_bits),
                        "kv_cache_group_size": int(args.kv_cache_group_size),
                        "stop_token_ids": [],
                        "prefill_step_size": int(prefill_step_size),
                    }
                    if args.return_prompt_cache:
                        result = _run_with_metal_capture(
                            capture_path,
                            lambda: _consume_stream_summary(
                                stream_dflash_generate(
                                    **run_kwargs,
                                    return_prompt_cache=True,
                                    prompt_cache_checkpoint_tokens=int(
                                        args.prompt_cache_checkpoint_tokens
                                    ),
                                    prompt_cache_target_only_checkpoints=bool(
                                        args.target_only_checkpoints
                                    ),
                                )
                            ),
                        )
                    else:
                        result = _run_with_metal_capture(
                            capture_path,
                            lambda: generate_dflash_once(**run_kwargs),
                        )
                    row = _compact_result(row_name, run_index, result, scenario)
                    row["run_wall_ms"] = (time.perf_counter() - row_start) * 1_000.0
                    row["mlx_memory_before"] = row_memory_before
                    row["mlx_memory_after"] = _mlx_memory_snapshot()
                    row["env"] = dict(env)
                    if capture_path is not None:
                        row["metal_capture"] = str(capture_path)
                    rows.append(row)
                    _print_row(row)
                    if (
                        args.return_prompt_cache
                        and reuse_suffix_tokens
                        and result.get("prompt_cache") is not None
                    ):
                        reuse_kwargs = {
                            **run_kwargs,
                            "prompt_tokens_override": reuse_suffix_tokens,
                            "prompt_cache": result["prompt_cache"],
                            "prompt_cache_count": len(prompt_tokens),
                        }
                        reuse_result = _run_with_metal_capture(
                            None,
                            lambda: _consume_stream_summary(
                                stream_dflash_generate(
                                    **reuse_kwargs,
                                    return_prompt_cache=True,
                                    prompt_cache_checkpoint_tokens=int(
                                        args.prompt_cache_checkpoint_tokens
                                    ),
                                    prompt_cache_target_only_checkpoints=bool(
                                        args.target_only_checkpoints
                                    ),
                                )
                            ),
                        )
                        reuse_row = _compact_result(
                            f"{row_name}_reuse{len(reuse_suffix_tokens)}",
                            run_index,
                            reuse_result,
                            scenario,
                        )
                        rows.append(reuse_row)
                        _print_row(reuse_row)
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
                if args.cooldown > 0:
                    time.sleep(float(args.cooldown))
            if stopped_by_budget:
                break
        if stopped_by_budget:
            break

    report = {
        "metadata": {
            "created_at_s": time.time(),
            "device": _device_metadata(),
            "mlx_memory_config": mlx_memory_config,
            "target_meta": target_meta,
            "env": _filtered_env(),
        },
        "model": args.model,
        "draft": args.draft,
        "prompt_tokens": len(prompt_tokens),
        "max_tokens": int(args.max_tokens),
        "prefill_step_size": int(args.prefill_step_size),
        "prefill_step_sizes": list(prefill_step_sizes),
        "quantize_kv_cache": bool(args.quantize_kv_cache),
        "mlx_cache_fraction": args.mlx_cache_fraction,
        "hybrid_mlp": bool(args.hybrid_mlp),
        "hybrid_mlp_threshold": int(args.hybrid_mlp_threshold),
        "hybrid_gdn_proj": bool(args.hybrid_gdn_proj),
        "hybrid_gdn_proj_threshold": int(args.hybrid_gdn_proj_threshold),
        "hybrid_gdn_linear": bool(args.hybrid_gdn_linear),
        "hybrid_gdn_linear_threshold": int(args.hybrid_gdn_linear_threshold),
        "hybrid_gdn_linear_attrs": args.hybrid_gdn_linear_attrs,
        "kv_cache_bits": int(args.kv_cache_bits),
        "kv_cache_group_size": int(args.kv_cache_group_size),
        "return_prompt_cache": bool(args.return_prompt_cache),
        "prompt_cache_checkpoint_tokens": int(args.prompt_cache_checkpoint_tokens),
        "target_only_checkpoints": bool(args.target_only_checkpoints),
        "reuse_suffix_tokens": int(args.reuse_suffix_tokens),
        "profile": bool(args.profile),
        "profile_prefill": bool(args.profile_prefill),
        "metal_capture": args.metal_capture,
        "max_seconds": max_seconds,
        "stopped_by_budget": bool(stopped_by_budget),
        "rows": rows,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
