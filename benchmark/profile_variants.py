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

from dflash_mlx.runtime import generate_dflash_once, load_draft_bundle, load_target_bundle


DEFAULT_PROMPT = (
    "You are solving a systems performance problem. Explain each step, track "
    "important numbers, and keep the final answer concise.\n"
)


SCENARIOS: dict[str, dict[str, Any]] = {
    "auto_b16": {"env": {"DFLASH_VERIFY_VARIANT": "auto"}, "block_tokens": 16},
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
        "prompt_tokens": int(result.get("prompt_token_count", 0) or 0),
        "generation_tokens": int(result.get("generation_tokens", 0) or 0),
        "elapsed_ms": float(result.get("elapsed_us", 0.0)) / 1_000.0,
        "prefill_ms": _phase_ms(result, "prefill"),
        "draft_ms": _phase_ms(result, "draft"),
        "draft_prefill_ms": _phase_ms(result, "draft_prefill"),
        "verify_ms": _phase_ms(result, "verify"),
        "replay_ms": _phase_ms(result, "replay"),
        "commit_ms": _phase_ms(result, "commit"),
        "decode_tps": _decode_tps(result),
        "acceptance_ratio": float(result.get("acceptance_ratio", 0.0) or 0.0),
        "tokens_per_cycle": float(result.get("tokens_per_cycle", 0.0) or 0.0),
        "cycles": int(result.get("cycles_completed", 0) or 0),
        "profile_totals_ms": _profile_totals_ms(result),
    }


def _print_row(row: dict[str, Any]) -> None:
    print(
        f"{row['name']:>12s} run={row['run']} "
        f"prefill={row['prefill_ms']:.1f}ms "
        f"decode={row['decode_tps']:.2f} tok/s "
        f"accept={row['acceptance_ratio'] * 100:.1f}% "
        f"draft={row['draft_ms']:.1f}ms verify={row['verify_ms']:.1f}ms "
        f"cycles={row['cycles']}",
        flush=True,
    )


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
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--scenarios",
        default="auto_b16,mma_b16,pipe4_b16,pipe8_b16,auto_b8,fastpath_b16",
        help=f"Comma-separated scenario names. Available: {', '.join(SCENARIOS)}",
    )
    parser.add_argument("--profile", action="store_true", help="Collect DFLASH_PROFILE cycle timings.")
    parser.add_argument(
        "--split-sdpa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable split full-attention SDPA hooks when loading the target model.",
    )
    parser.add_argument(
        "--quantize-kv-cache",
        action="store_true",
        help="Use quantized target full-attention KV cache during DFlash runs.",
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

    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)
        mx.set_cache_limit(wired_limit // 4)

    if args.profile:
        os.environ["DFLASH_PROFILE"] = "1"

    target_model, tokenizer, _ = load_target_bundle(
        args.model,
        lazy=True,
        split_full_attention_sdpa=bool(args.split_sdpa),
        quantize_kv_cache=bool(args.quantize_kv_cache),
    )
    draft_model, _ = load_draft_bundle(args.draft, lazy=True)
    prompt_tokens = _target_prompt_tokens(tokenizer, args.prompt_tokens, args.prompt)
    scenario_names = [name.strip() for name in args.scenarios.split(",") if name.strip()]
    repeat_count = max(1, int(args.repeat))
    max_seconds = max(0.0, float(args.max_seconds))
    benchmark_start = time.perf_counter()

    rows: list[dict[str, Any]] = []
    stopped_by_budget = False
    for run_index in range(1, repeat_count + 1):
        for name in scenario_names:
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
                capture_path = (
                    _capture_path_for_run(
                        args.metal_capture,
                        scenario_name=name,
                        run_index=run_index,
                        scenario_count=len(scenario_names),
                        repeat_count=repeat_count,
                    )
                    if args.metal_capture
                    else None
                )
                result = _run_with_metal_capture(
                    capture_path,
                    lambda: generate_dflash_once(
                        target_model=target_model,
                        tokenizer=tokenizer,
                        draft_model=draft_model,
                        prompt=args.prompt,
                        max_new_tokens=int(args.max_tokens),
                        use_chat_template=False,
                        block_tokens=int(scenario.get("block_tokens", 16)),
                        prompt_tokens_override=prompt_tokens,
                        quantize_kv_cache=bool(args.quantize_kv_cache),
                        stop_token_ids=[],
                        prefill_step_size=int(args.prefill_step_size),
                    ),
                )
                row = _compact_result(name, run_index, result, scenario)
                if capture_path is not None:
                    row["metal_capture"] = str(capture_path)
                rows.append(row)
                _print_row(row)
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            if args.cooldown > 0:
                time.sleep(float(args.cooldown))
        if stopped_by_budget:
            break

    report = {
        "model": args.model,
        "draft": args.draft,
        "prompt_tokens": len(prompt_tokens),
        "max_tokens": int(args.max_tokens),
        "prefill_step_size": int(args.prefill_step_size),
        "quantize_kv_cache": bool(args.quantize_kv_cache),
        "profile": bool(args.profile),
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
