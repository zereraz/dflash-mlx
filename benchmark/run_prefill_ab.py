"""Run alternating prefill A/B trials in separate processes.

Hybrid MLP is chosen at target-model load time, so a fair check must launch
separate processes. This runner alternates variants and writes one JSON file
per trial plus a compact aggregate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _load_prefill_ms(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows") or []
    if not rows:
        raise ValueError(f"{path} has no rows")
    return float(rows[0]["prefill_ms"])


def _variant_args(name: str, *, mlp_threshold: int, gdn_threshold: int) -> list[str]:
    mlp_threshold_s = str(int(mlp_threshold))
    gdn_threshold_s = str(int(gdn_threshold))
    if name == "q4":
        return []
    if name == "hybrid":
        return ["--hybrid-mlp", "--hybrid-mlp-threshold", mlp_threshold_s]
    if name == "hybrid_middle":
        return [
            "--hybrid-mlp",
            "--hybrid-mlp-threshold",
            mlp_threshold_s,
            "--middle-no-logits",
        ]
    if name == "hybrid_no_middle":
        return [
            "--hybrid-mlp",
            "--hybrid-mlp-threshold",
            mlp_threshold_s,
            "--no-middle-no-logits",
        ]
    if name == "hybrid_no_cacheopt":
        return [
            "--hybrid-mlp",
            "--hybrid-mlp-threshold",
            mlp_threshold_s,
            "--no-slice-capture",
            "--no-skip-final-mlp",
        ]
    if name == "hybrid_gdn":
        return [
            "--hybrid-mlp",
            "--hybrid-mlp-threshold",
            mlp_threshold_s,
            "--hybrid-gdn-proj",
            "--hybrid-gdn-proj-threshold",
            gdn_threshold_s,
        ]
    if name == "hybrid_gdn_linear":
        return [
            "--hybrid-mlp",
            "--hybrid-mlp-threshold",
            mlp_threshold_s,
            "--hybrid-gdn-linear",
            "--hybrid-gdn-linear-threshold",
            gdn_threshold_s,
        ]
    if name == "hybrid_gdn_qkv":
        return [
            "--hybrid-mlp",
            "--hybrid-mlp-threshold",
            mlp_threshold_s,
            "--hybrid-gdn-linear",
            "--hybrid-gdn-linear-threshold",
            gdn_threshold_s,
            "--hybrid-gdn-linear-attrs",
            "in_proj_qkv",
        ]
    if name == "hybrid_gdn_z":
        return [
            "--hybrid-mlp",
            "--hybrid-mlp-threshold",
            mlp_threshold_s,
            "--hybrid-gdn-linear",
            "--hybrid-gdn-linear-threshold",
            gdn_threshold_s,
            "--hybrid-gdn-linear-attrs",
            "in_proj_z",
        ]
    if name == "hybrid_gdn_out":
        return [
            "--hybrid-mlp",
            "--hybrid-mlp-threshold",
            mlp_threshold_s,
            "--hybrid-gdn-linear",
            "--hybrid-gdn-linear-threshold",
            gdn_threshold_s,
            "--hybrid-gdn-linear-attrs",
            "out_proj",
        ]
    if name == "hybrid_gdn_out_skip_final_attn":
        return [
            "--hybrid-mlp",
            "--hybrid-mlp-threshold",
            mlp_threshold_s,
            "--hybrid-gdn-linear",
            "--hybrid-gdn-linear-threshold",
            gdn_threshold_s,
            "--hybrid-gdn-linear-attrs",
            "out_proj",
            "--skip-final-attention",
        ]
    if name == "hybrid_gdn_out_no_skip_final_attn":
        return [
            "--hybrid-mlp",
            "--hybrid-mlp-threshold",
            mlp_threshold_s,
            "--hybrid-gdn-linear",
            "--hybrid-gdn-linear-threshold",
            gdn_threshold_s,
            "--hybrid-gdn-linear-attrs",
            "out_proj",
            "--no-skip-final-attention",
        ]
    if name == "hybrid_gdn_z_out":
        return [
            "--hybrid-mlp",
            "--hybrid-mlp-threshold",
            mlp_threshold_s,
            "--hybrid-gdn-linear",
            "--hybrid-gdn-linear-threshold",
            gdn_threshold_s,
            "--hybrid-gdn-linear-attrs",
            "in_proj_z,out_proj",
        ]
    if name == "hybrid_gdn_qkv_z_out":
        return [
            "--hybrid-mlp",
            "--hybrid-mlp-threshold",
            mlp_threshold_s,
            "--hybrid-gdn-linear",
            "--hybrid-gdn-linear-threshold",
            gdn_threshold_s,
            "--hybrid-gdn-linear-attrs",
            "in_proj_qkv,in_proj_z,out_proj",
        ]
    if name == "hybrid_gdn_qkv_z_out_no_split_sdpa":
        return [
            "--hybrid-mlp",
            "--hybrid-mlp-threshold",
            mlp_threshold_s,
            "--hybrid-gdn-linear",
            "--hybrid-gdn-linear-threshold",
            gdn_threshold_s,
            "--hybrid-gdn-linear-attrs",
            "in_proj_qkv,in_proj_z,out_proj",
            "--no-split-sdpa",
        ]
    if name == "hybrid_gdn_qkv_out":
        return [
            "--hybrid-mlp",
            "--hybrid-mlp-threshold",
            mlp_threshold_s,
            "--hybrid-gdn-linear",
            "--hybrid-gdn-linear-threshold",
            gdn_threshold_s,
            "--hybrid-gdn-linear-attrs",
            "in_proj_qkv,out_proj",
        ]
    raise ValueError(f"unknown variant: {name}")


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "count": float(len(values)),
        "best_ms": min(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Forward a real prompt text file to benchmark/profile_variants.py.",
    )
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help=(
            "Forwarded to profile_variants.py. Use 0 to benchmark cache-only "
            "prefill/checkpoint builds."
        ),
    )
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument(
        "--prompt-cache-checkpoint-tokens",
        type=int,
        default=0,
        help=(
            "Forwarded to profile_variants.py with --return-prompt-cache. "
            "Use 2048 to match the long-chat prompt-cache checkpoint path."
        ),
    )
    parser.add_argument(
        "--target-only-checkpoints",
        action="store_true",
        help=(
            "Forward --target-only-checkpoints to profile_variants.py. This "
            "benchmarks dense target-cache checkpoints without repeatedly "
            "persisting draft-context K/V at every checkpoint."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Deprecated shortcut: set both --mlp-threshold and --gdn-threshold.",
    )
    parser.add_argument("--mlp-threshold", type=int, default=256)
    parser.add_argument("--gdn-threshold", type=int, default=256)
    parser.add_argument("--scenarios", default="fastpath_b16")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument(
        "--order",
        default="q4,hybrid",
        help=(
            "Comma-separated variants for each pair. Known: "
            "q4,hybrid,hybrid_middle,hybrid_no_middle,hybrid_no_cacheopt,"
            "hybrid_gdn,hybrid_gdn_linear,hybrid_gdn_qkv,hybrid_gdn_z,"
            "hybrid_gdn_out,hybrid_gdn_out_skip_final_attn,"
            "hybrid_gdn_out_no_skip_final_attn,hybrid_gdn_z_out,"
            "hybrid_gdn_qkv_z_out,hybrid_gdn_qkv_z_out_no_split_sdpa,"
            "hybrid_gdn_qkv_out."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark/results/prefill-ab",
    )
    parser.add_argument("--aggregate", default=None)
    args = parser.parse_args()

    variants = [value.strip() for value in args.order.split(",") if value.strip()]
    if not variants:
        raise SystemExit("--order must contain at least one variant")
    if args.threshold is not None:
        mlp_threshold = int(args.threshold)
        gdn_threshold = int(args.threshold)
    else:
        mlp_threshold = int(args.mlp_threshold)
        gdn_threshold = int(args.gdn_threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = (
        Path(args.aggregate)
        if args.aggregate
        else output_dir / f"aggregate-{int(time.time())}.json"
    )

    trials: list[dict[str, Any]] = []
    for pair_index in range(1, max(1, int(args.pairs)) + 1):
        for variant in variants:
            trial_name = (
                f"{variant}-p{int(args.prompt_tokens)}-"
                f"s{int(args.prefill_step_size)}-pair{pair_index}"
            )
            output_path = output_dir / f"{trial_name}.json"
            cmd = [
                sys.executable,
                "benchmark/profile_variants.py",
                "--model",
                str(args.model),
                "--draft",
                str(args.draft),
                "--prompt-tokens",
                str(int(args.prompt_tokens)),
                "--max-tokens",
                str(int(args.max_tokens)),
                "--scenarios",
                str(args.scenarios),
                "--return-prompt-cache",
                "--prefill-step-size",
                str(int(args.prefill_step_size)),
                "--output",
                str(output_path),
                *_variant_args(
                    variant,
                    mlp_threshold=mlp_threshold,
                    gdn_threshold=gdn_threshold,
                ),
            ]
            if int(args.prompt_cache_checkpoint_tokens) > 0:
                cmd.extend(
                    [
                        "--prompt-cache-checkpoint-tokens",
                        str(int(args.prompt_cache_checkpoint_tokens)),
                    ]
                )
            if args.target_only_checkpoints:
                cmd.append("--target-only-checkpoints")
            if args.prompt_file:
                cmd.extend(["--prompt-file", str(Path(args.prompt_file).expanduser())])
            start = time.perf_counter()
            completed = subprocess.run(cmd, check=False, text=True)
            wall_s = time.perf_counter() - start
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
            prefill_ms = _load_prefill_ms(output_path)
            trial = {
                "variant": variant,
                "pair": pair_index,
                "prefill_ms": prefill_ms,
                "wall_s": wall_s,
                "output": str(output_path),
            }
            trials.append(trial)
            print(
                f"{trial_name}: prefill={prefill_ms:.1f}ms wall={wall_s:.1f}s",
                flush=True,
            )
            if float(args.cooldown) > 0.0:
                time.sleep(float(args.cooldown))

    by_variant: dict[str, list[float]] = {}
    for trial in trials:
        by_variant.setdefault(str(trial["variant"]), []).append(float(trial["prefill_ms"]))
    summaries = {variant: _summary(values) for variant, values in by_variant.items()}
    result = {
        "model": str(args.model),
        "draft": str(args.draft),
        "prompt_file": str(Path(args.prompt_file).expanduser()) if args.prompt_file else None,
        "prompt_tokens": int(args.prompt_tokens),
        "max_tokens": int(args.max_tokens),
        "prefill_step_size": int(args.prefill_step_size),
        "prompt_cache_checkpoint_tokens": int(args.prompt_cache_checkpoint_tokens),
        "target_only_checkpoints": bool(args.target_only_checkpoints),
        "scenarios": str(args.scenarios),
        "threshold": int(args.threshold) if args.threshold is not None else None,
        "mlp_threshold": int(mlp_threshold),
        "gdn_threshold": int(gdn_threshold),
        "pairs": int(args.pairs),
        "order": variants,
        "trials": trials,
        "summary": summaries,
    }
    if "q4" in summaries and "hybrid" in summaries:
        q4 = summaries["q4"]["median_ms"]
        hybrid = summaries["hybrid"]["median_ms"]
        result["hybrid_vs_q4_median_speedup"] = q4 / hybrid
        result["hybrid_vs_q4_median_delta_ms"] = q4 - hybrid
    if "q4" in summaries and "hybrid_gdn_out" in summaries:
        q4 = summaries["q4"]["median_ms"]
        hybrid_gdn_out = summaries["hybrid_gdn_out"]["median_ms"]
        result["hybrid_gdn_out_vs_q4_median_speedup"] = q4 / hybrid_gdn_out
        result["hybrid_gdn_out_vs_q4_median_delta_ms"] = q4 - hybrid_gdn_out
    if "q4" in summaries and "hybrid_gdn_z_out" in summaries:
        q4 = summaries["q4"]["median_ms"]
        hybrid_gdn_z_out = summaries["hybrid_gdn_z_out"]["median_ms"]
        result["hybrid_gdn_z_out_vs_q4_median_speedup"] = q4 / hybrid_gdn_z_out
        result["hybrid_gdn_z_out_vs_q4_median_delta_ms"] = q4 - hybrid_gdn_z_out
    if "q4" in summaries and "hybrid_gdn_qkv_z_out" in summaries:
        q4 = summaries["q4"]["median_ms"]
        hybrid_gdn_qkv_z_out = summaries["hybrid_gdn_qkv_z_out"]["median_ms"]
        result["hybrid_gdn_qkv_z_out_vs_q4_median_speedup"] = (
            q4 / hybrid_gdn_qkv_z_out
        )
        result["hybrid_gdn_qkv_z_out_vs_q4_median_delta_ms"] = (
            q4 - hybrid_gdn_qkv_z_out
        )
    if "hybrid" in summaries and "hybrid_gdn" in summaries:
        hybrid = summaries["hybrid"]["median_ms"]
        hybrid_gdn = summaries["hybrid_gdn"]["median_ms"]
        result["hybrid_gdn_vs_hybrid_median_speedup"] = hybrid / hybrid_gdn
        result["hybrid_gdn_vs_hybrid_median_delta_ms"] = hybrid - hybrid_gdn
    if "hybrid_no_cacheopt" in summaries and "hybrid" in summaries:
        hybrid_no_cacheopt = summaries["hybrid_no_cacheopt"]["median_ms"]
        hybrid = summaries["hybrid"]["median_ms"]
        result["hybrid_vs_hybrid_no_cacheopt_median_speedup"] = (
            hybrid_no_cacheopt / hybrid
        )
        result["hybrid_vs_hybrid_no_cacheopt_median_delta_ms"] = (
            hybrid_no_cacheopt - hybrid
        )
    if "hybrid" in summaries and "hybrid_middle" in summaries:
        hybrid = summaries["hybrid"]["median_ms"]
        hybrid_middle = summaries["hybrid_middle"]["median_ms"]
        result["hybrid_middle_vs_hybrid_median_speedup"] = hybrid / hybrid_middle
        result["hybrid_middle_vs_hybrid_median_delta_ms"] = hybrid - hybrid_middle
    if "hybrid" in summaries and "hybrid_gdn_linear" in summaries:
        hybrid = summaries["hybrid"]["median_ms"]
        hybrid_gdn_linear = summaries["hybrid_gdn_linear"]["median_ms"]
        result["hybrid_gdn_linear_vs_hybrid_median_speedup"] = (
            hybrid / hybrid_gdn_linear
        )
        result["hybrid_gdn_linear_vs_hybrid_median_delta_ms"] = (
            hybrid - hybrid_gdn_linear
        )
    for selective_name in (
        "hybrid_gdn_qkv",
        "hybrid_gdn_z",
        "hybrid_gdn_out",
        "hybrid_gdn_z_out",
        "hybrid_gdn_qkv_z_out",
        "hybrid_gdn_qkv_z_out_no_split_sdpa",
        "hybrid_gdn_qkv_out",
    ):
        if "hybrid" in summaries and selective_name in summaries:
            hybrid = summaries["hybrid"]["median_ms"]
            selective = summaries[selective_name]["median_ms"]
            result[f"{selective_name}_vs_hybrid_median_speedup"] = hybrid / selective
            result[f"{selective_name}_vs_hybrid_median_delta_ms"] = hybrid - selective
    if "hybrid_gdn_out_no_skip_final_attn" in summaries and "hybrid_gdn_out" in summaries:
        no_skip_final_attn = summaries["hybrid_gdn_out_no_skip_final_attn"]["median_ms"]
        skip_final_attn = summaries["hybrid_gdn_out"]["median_ms"]
        result["skip_final_attn_vs_no_skip_median_speedup"] = (
            no_skip_final_attn / skip_final_attn
        )
        result["skip_final_attn_vs_no_skip_median_delta_ms"] = (
            no_skip_final_attn - skip_final_attn
        )
    if "hybrid_gdn_out" in summaries and "hybrid_gdn_out_skip_final_attn" in summaries:
        hybrid_gdn_out = summaries["hybrid_gdn_out"]["median_ms"]
        skip_final_attn = summaries["hybrid_gdn_out_skip_final_attn"]["median_ms"]
        result["skip_final_attn_vs_hybrid_gdn_out_median_speedup"] = (
            hybrid_gdn_out / skip_final_attn
        )
        result["skip_final_attn_vs_hybrid_gdn_out_median_delta_ms"] = (
            hybrid_gdn_out - skip_final_attn
        )
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    if "hybrid_vs_q4_median_speedup" in result:
        print(
            "hybrid_vs_q4_median_speedup="
            f"{result['hybrid_vs_q4_median_speedup']:.3f}",
            flush=True,
        )
    if "hybrid_gdn_out_vs_q4_median_speedup" in result:
        print(
            "hybrid_gdn_out_vs_q4_median_speedup="
            f"{result['hybrid_gdn_out_vs_q4_median_speedup']:.3f}",
            flush=True,
        )
    if "hybrid_gdn_z_out_vs_q4_median_speedup" in result:
        print(
            "hybrid_gdn_z_out_vs_q4_median_speedup="
            f"{result['hybrid_gdn_z_out_vs_q4_median_speedup']:.3f}",
            flush=True,
        )
    if "hybrid_gdn_qkv_z_out_vs_q4_median_speedup" in result:
        print(
            "hybrid_gdn_qkv_z_out_vs_q4_median_speedup="
            f"{result['hybrid_gdn_qkv_z_out_vs_q4_median_speedup']:.3f}",
            flush=True,
        )
    if "hybrid_gdn_vs_hybrid_median_speedup" in result:
        print(
            "hybrid_gdn_vs_hybrid_median_speedup="
            f"{result['hybrid_gdn_vs_hybrid_median_speedup']:.3f}",
            flush=True,
        )
    if "hybrid_vs_hybrid_no_cacheopt_median_speedup" in result:
        print(
            "hybrid_vs_hybrid_no_cacheopt_median_speedup="
            f"{result['hybrid_vs_hybrid_no_cacheopt_median_speedup']:.3f}",
            flush=True,
        )
    if "hybrid_middle_vs_hybrid_median_speedup" in result:
        print(
            "hybrid_middle_vs_hybrid_median_speedup="
            f"{result['hybrid_middle_vs_hybrid_median_speedup']:.3f}",
            flush=True,
        )
    if "hybrid_gdn_linear_vs_hybrid_median_speedup" in result:
        print(
            "hybrid_gdn_linear_vs_hybrid_median_speedup="
            f"{result['hybrid_gdn_linear_vs_hybrid_median_speedup']:.3f}",
            flush=True,
        )
    for selective_name in (
        "hybrid_gdn_qkv",
        "hybrid_gdn_z",
        "hybrid_gdn_out",
        "hybrid_gdn_out_no_skip_final_attn",
        "hybrid_gdn_z_out",
        "hybrid_gdn_qkv_z_out",
        "hybrid_gdn_qkv_z_out_no_split_sdpa",
        "hybrid_gdn_qkv_out",
    ):
        key = f"{selective_name}_vs_hybrid_median_speedup"
        if key in result:
            print(f"{key}={result[key]:.3f}", flush=True)
    if "skip_final_attn_vs_hybrid_gdn_out_median_speedup" in result:
        print(
            "skip_final_attn_vs_hybrid_gdn_out_median_speedup="
            f"{result['skip_final_attn_vs_hybrid_gdn_out_median_speedup']:.3f}",
            flush=True,
        )
    if "skip_final_attn_vs_no_skip_median_speedup" in result:
        print(
            "skip_final_attn_vs_no_skip_median_speedup="
            f"{result['skip_final_attn_vs_no_skip_median_speedup']:.3f}",
            flush=True,
        )
    print(f"aggregate={aggregate_path}", flush=True)


if __name__ == "__main__":
    main()
