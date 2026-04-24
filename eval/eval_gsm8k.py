#!/usr/bin/env python3
"""Minimal GSM8K eval for dflash-mlx quality gating.

Compares baseline (stock mlx_lm) vs DFlash generation accuracy.
Runs a small subset by default (30 problems) to keep it fast.

Usage:
    cd ~/Code/Zereraz/research-oss/dflash-mlx
    source .venv/bin/activate
    python eval/eval_gsm8k.py \
        --model ~/models/Huihui-Qwen3.6-27B-abliterated-4.5bit-msq \
        --draft ~/models/Qwen3.6-27B-DFlash \
        --num-samples 30
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx


def extract_boxed_answer(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()
    numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


def extract_gsm8k_answer(answer_text: str) -> str | None:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return extract_boxed_answer(answer_text)


def normalize_number(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):
        return s


def load_gsm8k(num_samples: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for i, row in enumerate(ds):
        if i >= num_samples:
            break
        items.append({
            "question": row["question"],
            "gold_answer": extract_gsm8k_answer(row["answer"]),
        })
    return items


def generate_baseline(model, tokenizer, prompt: str, max_tokens: int) -> dict:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    tokens = []
    first_token_time = None
    final_response = None
    start = time.perf_counter()
    for r in stream_generate(model, tokenizer, prompt, max_tokens, sampler=sampler):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        tokens.append(r.token)
        final_response = r
    elapsed = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else elapsed
    decode_tps = float(getattr(final_response, "generation_tps", 0.0) or 0.0)
    if decode_tps <= 0.0:
        decode_time = elapsed - ttft
        decode_tokens = max(0, len(tokens) - 1)
        decode_tps = decode_tokens / decode_time if decode_time > 0 and decode_tokens else 0
    return {
        "text": tokenizer.decode(tokens),
        "tokens": tokens,
        "elapsed": elapsed,
        "ttft": ttft,
        "decode_tps": decode_tps,
        "prefill_tps": 0,
        "acceptance": 0,
    }


def _run_dflash(target_model, tokenizer, draft_model, prompt: str, max_tokens: int, quantize_kv: bool = False) -> dict:
    from dflash_mlx.runtime import stream_dflash_generate
    from dflash_mlx.generate import get_stop_token_ids

    stop_ids = get_stop_token_ids(tokenizer)
    tokens = []
    summary = None
    first_token_time = None
    start = time.perf_counter()
    for event in stream_dflash_generate(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompt=prompt,
        max_new_tokens=max_tokens,
        use_chat_template=False,
        stop_token_ids=stop_ids,
        quantize_kv_cache=quantize_kv,
    ):
        if event.get("event") == "token":
            if first_token_time is None:
                first_token_time = time.perf_counter()
            tokens.append(int(event["token_id"]))
        elif event.get("event") == "summary":
            summary = event
    elapsed = time.perf_counter() - start
    ttft = (first_token_time - start) if first_token_time else elapsed

    phase_timings = dict(summary.get("phase_timings_us", {})) if summary else {}
    prefill_us = float(summary.get("prefill_us", phase_timings.get("prefill", 0))) if summary else 0
    prefill_s = prefill_us / 1e6
    prompt_tokens = int(summary.get("prompt_token_count", 0)) if summary else 0
    prefill_tps = prompt_tokens / prefill_s if prefill_s > 0 else 0
    measured_elapsed = float(summary.get("elapsed_us", elapsed * 1e6)) / 1e6 if summary else elapsed
    decode_time = measured_elapsed - prefill_s
    decode_tps = len(tokens) / decode_time if decode_time > 0 and tokens else 0
    acceptance = float(summary.get("acceptance_ratio", 0)) if summary else 0

    return {
        "text": tokenizer.decode(tokens),
        "tokens": tokens,
        "elapsed": elapsed,
        "ttft": ttft,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
        "acceptance": acceptance,
    }


def generate_dflash(target_model, tokenizer, draft_model, prompt: str, max_tokens: int) -> dict:
    return _run_dflash(target_model, tokenizer, draft_model, prompt, max_tokens, quantize_kv=False)


def generate_dflash_quantized(target_model, tokenizer, draft_model, prompt: str, max_tokens: int) -> dict:
    return _run_dflash(target_model, tokenizer, draft_model, prompt, max_tokens, quantize_kv=True)


def format_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def main():
    parser = argparse.ArgumentParser(description="GSM8K eval for dflash-mlx")
    parser.add_argument("--model", type=str, required=True, help="Target model path")
    parser.add_argument("--draft", type=str, default=None, help="Draft model path")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--modes", type=str, default="baseline,dflash,dflash_qkv",
                        help="Comma-separated: baseline,dflash,dflash_qkv")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]

    print(f"Loading GSM8K ({args.num_samples} samples)...")
    dataset = load_gsm8k(args.num_samples)

    has_baseline = "baseline" in modes
    has_dflash = any(m.startswith("dflash") for m in modes)

    baseline_model = None
    baseline_tokenizer = None
    if has_baseline:
        print(f"Loading baseline model (stock mlx_lm): {args.model}")
        from mlx_lm import load as mlx_load
        baseline_model, baseline_tokenizer = mlx_load(args.model)

    target_model = None
    tokenizer = None
    draft_model = None
    if has_dflash:
        print(f"Loading DFlash target model: {args.model}")
        from dflash_mlx.runtime import load_target_bundle, load_draft_bundle
        target_model, tokenizer, _ = load_target_bundle(args.model, lazy=True)
        assert args.draft, "--draft required for dflash modes"
        print(f"Loading draft model: {args.draft}")
        draft_model, _ = load_draft_bundle(args.draft, lazy=True)

    if tokenizer is None:
        tokenizer = baseline_tokenizer

    results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Running: {mode}")
        print(f"{'='*60}")

        correct = 0
        total = 0
        total_tokens = 0
        total_time = 0.0
        per_sample = []

        sum_decode_tps = 0.0
        sum_prefill_tps = 0.0
        sum_ttft = 0.0
        sum_acceptance = 0.0

        for i, item in enumerate(dataset):
            use_tokenizer = baseline_tokenizer if mode == "baseline" else tokenizer
            prompt = format_prompt(use_tokenizer, item["question"])

            if mode == "baseline":
                r = generate_baseline(baseline_model, baseline_tokenizer, prompt, args.max_tokens)
            elif mode == "dflash":
                r = generate_dflash(target_model, tokenizer, draft_model, prompt, args.max_tokens)
            elif mode == "dflash_qkv":
                r = generate_dflash_quantized(target_model, tokenizer, draft_model, prompt, args.max_tokens)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            text, tokens, elapsed = r["text"], r["tokens"], r["elapsed"]
            predicted = normalize_number(extract_boxed_answer(text))
            gold = normalize_number(item["gold_answer"])
            is_correct = predicted == gold

            if is_correct:
                correct += 1
            total += 1
            total_tokens += len(tokens)
            total_time += elapsed
            sum_decode_tps += r["decode_tps"]
            sum_prefill_tps += r["prefill_tps"]
            sum_ttft += r["ttft"]
            sum_acceptance += r["acceptance"]

            per_sample.append({
                "index": i,
                "correct": is_correct,
                "predicted": predicted,
                "gold": gold,
                "num_tokens": len(tokens),
                "elapsed": round(elapsed, 2),
                "decode_tps": round(r["decode_tps"], 1),
                "ttft": round(r["ttft"], 3),
                "acceptance": round(r["acceptance"], 3),
            })

            status = "PASS" if is_correct else "FAIL"
            print(f"  [{i+1:3d}/{args.num_samples}] {status} | pred={predicted} gold={gold} | {len(tokens)} tok | decode={r['decode_tps']:.1f} t/s | ttft={r['ttft']:.2f}s")

        accuracy = correct / total if total > 0 else 0
        avg_tps = total_tokens / total_time if total_time > 0 else 0

        avg_decode_tps = sum_decode_tps / total if total > 0 else 0
        avg_prefill_tps = sum_prefill_tps / total if total > 0 else 0
        avg_ttft = sum_ttft / total if total > 0 else 0
        avg_acceptance = sum_acceptance / total if total > 0 else 0

        results[mode] = {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total,
            "e2e_tps": round(avg_tps, 2),
            "decode_tps": round(avg_decode_tps, 2),
            "prefill_tps": round(avg_prefill_tps, 2),
            "avg_ttft": round(avg_ttft, 3),
            "acceptance": round(avg_acceptance, 4),
            "total_tokens": total_tokens,
            "total_time": round(total_time, 2),
            "per_sample": per_sample,
        }

        print(f"\n  {mode}: {correct}/{total} = {accuracy*100:.1f}% | decode={avg_decode_tps:.1f} t/s | ttft={avg_ttft:.2f}s | accept={avg_acceptance*100:.1f}%")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Mode':15s} {'Acc':>6s} {'Decode':>10s} {'TTFT':>8s} {'Accept':>8s} {'E2E':>8s}")
    print(f"  {'-'*55}")
    for mode, r in results.items():
        acc_str = f"{r['correct']}/{r['total']}"
        print(f"  {mode:15s} {acc_str:>6s} {r['decode_tps']:>8.1f}/s {r['avg_ttft']:>7.2f}s {r['acceptance']*100:>6.1f}% {r['e2e_tps']:>6.1f}/s")

    if len(results) > 1:
        modes_list = list(results.keys())
        base = results[modes_list[0]]
        for m in modes_list[1:]:
            other = results[m]
            match = sum(
                1 for a, b in zip(base["per_sample"], other["per_sample"])
                if a["predicted"] == b["predicted"]
            )
            print(f"  {modes_list[0]} vs {m}: {match}/{base['total']} predicted-answer agreement")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "model": args.model,
            "draft": args.draft,
            "num_samples": args.num_samples,
            "max_tokens": args.max_tokens,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(out_path, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        print(f"\n  Saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
