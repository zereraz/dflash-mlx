from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from dflash_mlx.runtime import load_draft_bundle, load_target_bundle, stream_dflash_generate


DEFAULT_PROMPT = "Write a Python quicksort with tests."


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _hardware() -> dict[str, Any]:
    out: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        out["chip"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        out["chip"] = "unknown"
    try:
        out["memory_gb"] = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        ) / (1024**3)
    except Exception:
        out["memory_gb"] = None
    return out


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _stop_token_ids(tokenizer: Any) -> list[int]:
    out = list(getattr(tokenizer, "eos_token_ids", None) or [])
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None and int(eos) not in out:
        out.append(int(eos))
    return out


def _prompt_tokens(tokenizer: Any, prompt: str, n_tokens: int) -> list[int]:
    base = list(tokenizer.encode(prompt))
    if not base:
        raise ValueError("prompt encoded to zero tokens")
    reps = (int(n_tokens) + len(base) - 1) // len(base)
    return (base * reps)[: int(n_tokens)]


def _generation_tps(summary: dict[str, Any]) -> float:
    elapsed_us = float(summary.get("elapsed_us", 0.0))
    phase = dict(summary.get("phase_timings_us", {}) or {})
    prefill_us = float(phase.get("prefill", 0.0))
    gen_tokens = int(summary.get("generation_tokens", 0))
    gen_us = max(0.0, elapsed_us - prefill_us)
    return gen_tokens / (gen_us / 1e6) if gen_us > 0 else 0.0


def _avg_cycle_field(cycles: list[dict[str, Any]], field: str) -> float | None:
    values = [float(c[field]) for c in cycles if field in c]
    if not values:
        return None
    return sum(values) / len(values)


def _run_one(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    block_tokens: int | None,
    window: int,
) -> dict[str, Any]:
    old_profile = os.environ.get("DFLASH_PROFILE")
    os.environ["DFLASH_TARGET_FA_WINDOW"] = str(int(window))
    os.environ["DFLASH_PROFILE"] = "1"
    cycles: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    t0 = time.perf_counter_ns()
    try:
        stream = stream_dflash_generate(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            prompt="",
            max_new_tokens=int(max_tokens),
            use_chat_template=False,
            block_tokens=block_tokens,
            stop_token_ids=[],
            suppress_token_ids=_stop_token_ids(tokenizer),
            prompt_tokens_override=prompt_tokens,
            quantize_kv_cache=False,
        )
        try:
            for event in stream:
                if event.get("event") == "cycle_complete":
                    cycles.append(dict(event))
                elif event.get("event") == "summary":
                    summary = dict(event)
        finally:
            stream.close()
    finally:
        if old_profile is None:
            os.environ.pop("DFLASH_PROFILE", None)
        else:
            os.environ["DFLASH_PROFILE"] = old_profile
    wall_us = (time.perf_counter_ns() - t0) / 1_000.0
    if summary is None:
        raise RuntimeError("DFlash stream did not yield a summary")
    phase = dict(summary.get("phase_timings_us", {}) or {})
    verify_total_us = float(phase.get("verify", 0.0))
    cycles_completed = int(summary.get("cycles_completed", 0))
    out = {
        "window": int(window),
        "prompt_tokens": len(prompt_tokens),
        "generated_tokens": int(summary.get("generation_tokens", 0)),
        "wall_us": wall_us,
        "tok_s": _generation_tps(summary),
        "verify_us_total": verify_total_us,
        "verify_us_per_cycle": (
            verify_total_us / cycles_completed if cycles_completed > 0 else None
        ),
        "cycle_total_us_per_cycle": _avg_cycle_field(cycles, "cycle_total_us"),
        "acceptance_ratio": float(summary.get("acceptance_ratio", 0.0)),
        "tokens_per_cycle": float(summary.get("tokens_per_cycle", 0.0)),
        "cycles_completed": cycles_completed,
        "peak_memory_gb": summary.get("peak_memory_gb"),
        "target_fa_window_reported": int(summary.get("target_fa_window", -1)),
        "phase_timings_us": phase,
        "cycle_profile_us": cycles,
    }
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    return out


def _default_output_path() -> Path:
    ts = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    return Path("/tmp") / f"dflash_target_fa_window_probe_{ts}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen3.6-27B-4bit")
    parser.add_argument("--draft", default="z-lab/Qwen3.6-27B-DFlash")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-tokens", default="2048,16384")
    parser.add_argument("--windows", default="0,2048,4096,8192")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--block-tokens", type=int, default=22)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows = _parse_int_list(args.windows)
    prompt_lengths = _parse_int_list(args.prompt_tokens)

    report: dict[str, Any] = {
        "metadata": {
            "command": " ".join(sys.argv),
            "git_hash": _git_hash(),
            "python": platform.python_version(),
            "mlx_version": getattr(mx, "__version__", _version("mlx")),
            "mlx_lm_version": _version("mlx-lm"),
            "hardware": _hardware(),
            "model": args.model,
            "draft": args.draft,
            "max_tokens": int(args.max_tokens),
            "block_tokens": int(args.block_tokens) if args.block_tokens else None,
            "prompt_regime": "raw tokenizer encode, repeated/truncated to prompt_tokens",
        },
        "runs": [],
    }

    target_model, tokenizer, target_meta = load_target_bundle(args.model, lazy=True)
    draft_model, draft_meta = load_draft_bundle(args.draft, lazy=True)
    report["metadata"]["resolved_model"] = target_meta.get("resolved_model_ref")
    report["metadata"]["resolved_draft"] = draft_meta.get("resolved_model_ref")

    try:
        for prompt_len in prompt_lengths:
            tokens = _prompt_tokens(tokenizer, args.prompt, prompt_len)
            for window in windows:
                run = {
                    "prompt_tokens": prompt_len,
                    "window": window,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                try:
                    run.update(
                        _run_one(
                            target_model=target_model,
                            tokenizer=tokenizer,
                            draft_model=draft_model,
                            prompt_tokens=tokens,
                            max_tokens=args.max_tokens,
                            block_tokens=args.block_tokens,
                            window=window,
                        )
                    )
                    run["error"] = None
                except Exception as exc:
                    run["error"] = repr(exc)
                report["runs"].append(run)
                output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps(run, indent=2))
                if args.cooldown > 0:
                    time.sleep(float(args.cooldown))
    finally:
        del target_model
        del tokenizer
        del draft_model
        gc.collect()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
