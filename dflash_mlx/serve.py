# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import json
import sys
import os
import logging
import threading
import time
import warnings
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any, Optional

warnings.filterwarnings("ignore", message="mlx_lm.server is not recommended")

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
try:
    from huggingface_hub.utils import disable_progress_bars
except Exception:
    disable_progress_bars = None

if disable_progress_bars is not None:
    try:
        disable_progress_bars()
    except Exception:
        pass

import mlx.core as mx
import mlx_lm.server as mlx_server

from dflash_mlx.generate import (
    get_stop_token_ids,
    load_runtime_components,
    resolve_optional_draft_ref,
)
from dflash_mlx.prompt_disk_cache import DiskBackedPromptCache
from dflash_mlx.runtime import stream_dflash_generate


_STATEFUL_SERVER_API = "state" in getattr(mlx_server.Response, "__annotations__", {})
_METRICS_LOG_ENV = "DFLASH_METRICS_LOG"
_METRICS_TOKEN_INTERVAL_ENV = "DFLASH_METRICS_TOKEN_INTERVAL"
_PROMPT_CACHE_DIR_ENV = "DFLASH_PROMPT_CACHE_DIR"
_METRICS_LOG_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def _dflash_metrics_log_path(cli_args: Any) -> Optional[str]:
    path = getattr(cli_args, "dflash_metrics_log", None)
    if path is None or str(path).strip() == "":
        path = os.environ.get(_METRICS_LOG_ENV, "").strip()
    if path is None or str(path).strip() == "":
        return None
    return os.path.expanduser(str(path))


def _dflash_metrics_token_interval() -> int:
    raw_value = os.environ.get(_METRICS_TOKEN_INTERVAL_ENV, "").strip()
    if not raw_value:
        return 128
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 128


def _dflash_prompt_cache_dir(cli_args: Any) -> Optional[str]:
    path = getattr(cli_args, "dflash_prompt_cache_dir", None)
    if path is None or str(path).strip() == "":
        path = os.environ.get(_PROMPT_CACHE_DIR_ENV, "").strip()
    if path is None or str(path).strip() == "":
        return None
    return os.path.expanduser(str(path))


def _kv_cache_bits(cli_args: Any) -> int:
    return int(getattr(cli_args, "kv_cache_bits", 8) or 8)


def _kv_cache_group_size(cli_args: Any) -> int:
    return int(getattr(cli_args, "kv_cache_group_size", 64) or 64)


def _dflash_model_key(
    model_ref: str,
    resolved_draft_ref: Optional[str],
    cli_args: Any,
) -> tuple[Any, Any, Any]:
    kv_cache_config = (
        "target_kv",
        bool(getattr(cli_args, "quantize_kv_cache", False)),
        _kv_cache_bits(cli_args),
        _kv_cache_group_size(cli_args),
    )
    return (model_ref, kv_cache_config, resolved_draft_ref)


def _phase_timings_ms(
    phase_timings_us: dict[str, Any],
    *,
    stable_cache_build_us: float = 0.0,
) -> dict[str, float]:
    timings = {
        str(key): float(value or 0.0) / 1_000.0
        for key, value in phase_timings_us.items()
    }
    if stable_cache_build_us:
        timings["prefill"] = timings.get("prefill", 0.0) + (
            stable_cache_build_us / 1_000.0
        )
        timings["stable_cache_build"] = stable_cache_build_us / 1_000.0
    return timings


def _build_dflash_metrics_record(
    *,
    request_id: str,
    summary_event: dict[str, Any],
    prompt_len: int,
    finish_reason: Optional[str],
    prompt_cache_count: int = 0,
    stable_cache_build_us: float = 0.0,
    using_stable_prompt_cache: bool = False,
    timestamp_s: Optional[float] = None,
) -> dict[str, Any]:
    phase_timings_us = dict(summary_event.get("phase_timings_us") or {})
    elapsed_us = float(summary_event.get("elapsed_us", 0.0) or 0.0)
    prefill_us = float(phase_timings_us.get("prefill", 0.0) or 0.0)
    elapsed_us += stable_cache_build_us
    prefill_us += stable_cache_build_us
    decode_us = max(0.0, elapsed_us - prefill_us)
    generation_tokens = int(summary_event.get("generation_tokens", 0) or 0)
    cycles_completed = int(summary_event.get("cycles_completed", 0) or 0)
    draft_tokens_attempted = int(
        summary_event.get("draft_tokens_attempted", 0) or 0
    )
    prompt_token_count = int(
        summary_event.get("prompt_token_count", prompt_len) or prompt_len
    )
    cached_prompt_tokens = max(0, int(prompt_cache_count))
    timestamp_s = time.time() if timestamp_s is None else timestamp_s

    return {
        "schema": "dflash_session_metrics_v1",
        "event": "summary",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(timestamp_s)),
        "timestamp_s": float(timestamp_s),
        "request_id": request_id,
        "finish_reason": finish_reason,
        "prompt_tokens": int(prompt_len),
        "runtime_prompt_tokens": prompt_token_count,
        "cached_prompt_tokens": cached_prompt_tokens,
        "uncached_prompt_tokens": max(0, prompt_token_count - cached_prompt_tokens),
        "using_stable_prompt_cache": bool(using_stable_prompt_cache),
        "stable_cache_build_ms": stable_cache_build_us / 1_000.0,
        "generation_tokens": generation_tokens,
        "elapsed_ms": elapsed_us / 1_000.0,
        "prefill_ms": prefill_us / 1_000.0,
        "decode_ms": decode_us / 1_000.0,
        "decode_tps": (
            generation_tokens / (decode_us / 1_000_000.0) if decode_us > 0.0 else 0.0
        ),
        "accepted_from_draft": int(summary_event.get("accepted_from_draft", 0) or 0),
        "acceptance_ratio": float(summary_event.get("acceptance_ratio", 0.0) or 0.0),
        "draft_tokens_attempted": draft_tokens_attempted,
        "draft_acceptance_ratio": float(
            summary_event.get("draft_acceptance_ratio", 0.0) or 0.0
        ),
        "cycles_completed": cycles_completed,
        "tokens_per_cycle": float(summary_event.get("tokens_per_cycle", 0.0) or 0.0),
        "block_tokens": summary_event.get("block_tokens"),
        "verify_len_cap": summary_event.get("verify_len_cap"),
        "verify_chunk_tokens": summary_event.get("verify_chunk_tokens"),
        "prefill_step_size": summary_event.get("prefill_step_size"),
        "quantize_kv_cache": bool(summary_event.get("quantize_kv_cache", False)),
        "kv_cache_bits": summary_event.get("kv_cache_bits"),
        "kv_cache_group_size": summary_event.get("kv_cache_group_size"),
        "speculative_linear_cache": bool(
            summary_event.get("speculative_linear_cache", False)
        ),
        "prefill_cache_fastpath": bool(
            summary_event.get("prefill_cache_fastpath", False)
        ),
        "prefill_defer_draft_context": bool(
            summary_event.get("prefill_defer_draft_context", False)
        ),
        "prefill_skip_capture": bool(summary_event.get("prefill_skip_capture", False)),
        "prefill_context_tokens": int(
            summary_event.get("prefill_context_tokens", 0) or 0
        ),
        "cache_only_prefill": bool(summary_event.get("cache_only_prefill", False)),
        "dflash_generation_tokens": int(
            summary_event.get("dflash_generation_tokens", generation_tokens) or 0
        ),
        "fallback_ar_generation_tokens": int(
            summary_event.get("fallback_ar_generation_tokens", 0) or 0
        ),
        "fallback_ar": bool(summary_event.get("fallback_ar", False)),
        "fallback_reason": summary_event.get("fallback_reason"),
        "adaptive_fallback_ar": bool(
            summary_event.get("adaptive_fallback_ar", False)
        ),
        "adaptive_fallback_cycle": summary_event.get("adaptive_fallback_cycle"),
        "adaptive_fallback_reason": summary_event.get("adaptive_fallback_reason"),
        "adaptive_fallback_recent_tokens_per_cycle": summary_event.get(
            "adaptive_fallback_recent_tokens_per_cycle"
        ),
        "adaptive_fallback_probe_cycles": summary_event.get(
            "adaptive_fallback_probe_cycles"
        ),
        "adaptive_fallback_window": summary_event.get("adaptive_fallback_window"),
        "adaptive_fallback_min_tokens_per_cycle": summary_event.get(
            "adaptive_fallback_min_tokens_per_cycle"
        ),
        "adaptive_fallback_cooldown_tokens": summary_event.get(
            "adaptive_fallback_cooldown_tokens"
        ),
        "adaptive_fallback_reprobe_block_tokens": summary_event.get(
            "adaptive_fallback_reprobe_block_tokens"
        ),
        "adaptive_fallback_count": int(
            summary_event.get("adaptive_fallback_count", 0) or 0
        ),
        "adaptive_reprobe_count": int(
            summary_event.get("adaptive_reprobe_count", 0) or 0
        ),
        "adaptive_current_block_tokens": summary_event.get(
            "adaptive_current_block_tokens"
        ),
        "phase_timings_ms": _phase_timings_ms(
            phase_timings_us,
            stable_cache_build_us=stable_cache_build_us,
        ),
        "acceptance_position_attempts": list(
            summary_event.get("acceptance_position_attempts", []) or []
        ),
        "acceptance_position_accepts": list(
            summary_event.get("acceptance_position_accepts", []) or []
        ),
        "acceptance_position_rates": list(
            summary_event.get("acceptance_position_rates", []) or []
        ),
        "acceptance_first_20_avg": float(
            summary_event.get("acceptance_first_20_avg", 0.0) or 0.0
        ),
        "acceptance_last_20_avg": float(
            summary_event.get("acceptance_last_20_avg", 0.0) or 0.0
        ),
        "peak_memory_gb": summary_event.get("peak_memory_gb"),
    }


def _append_dflash_metrics_event(cli_args: Any, record: dict[str, Any]) -> None:
    path = _dflash_metrics_log_path(cli_args)
    if path is None:
        return
    record = dict(record)
    record.setdefault("schema", "dflash_session_metrics_v1")
    record.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    record.setdefault("timestamp_s", time.time())
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with _METRICS_LOG_LOCK:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
    except Exception as exc:
        logger.warning("failed to write DFlash metrics log %s: %s", path, exc)


def _fetch_dflash_prompt_cache(
    prompt_cache_store: Any,
    model_key: Any,
    prompt: list[int],
    *,
    allow_exact: bool = False,
):
    prompt_cache, prompt_rest = prompt_cache_store.fetch_nearest_cache(
        model_key,
        prompt,
    )
    prompt_cache_count = len(prompt) - len(prompt_rest)
    if prompt_cache is not None and len(prompt_rest) == 0 and not allow_exact:
        # DFlash needs at least one uncached token to recover the first-token
        # logits from a reused KV state.
        prompt_cache = None
        prompt_rest = prompt
        prompt_cache_count = 0
    return prompt_cache, prompt_rest, prompt_cache_count


def _select_dflash_stable_prompt_prefix(
    prompt: list[int],
    segments: list[list[int]],
    segment_types: list[str],
) -> tuple[list[int], list[int]]:
    if len(prompt) <= 1:
        return [], prompt

    stable_len: Optional[int] = None
    if (
        segments
        and segment_types
        and len(segments) == len(segment_types)
        and segment_types[-1] == "assistant"
        and len(segments[-1]) > 0
    ):
        stable_len = len(prompt) - len(segments[-1])

    if stable_len is None or stable_len <= 0 or stable_len >= len(prompt):
        stable_len = len(prompt) - 1

    return prompt[:stable_len], prompt[stable_len:]


def _dflash_server_prompt_cache_enabled() -> bool:
    raw = os.environ.get("DFLASH_SERVER_PROMPT_CACHE", "").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _use_dflash_prompt_cache(cli_args: Any) -> bool:
    return bool(
        getattr(cli_args, "dflash_prompt_cache", False)
        or _dflash_prompt_cache_dir(cli_args) is not None
    ) or _dflash_server_prompt_cache_enabled()


def _stabilize_dflash_prompt_cache_chat_template_args(cli_args: Any) -> None:
    if not _use_dflash_prompt_cache(cli_args):
        return

    chat_template_args = getattr(cli_args, "chat_template_args", None)
    if not isinstance(chat_template_args, dict):
        return

    if (
        chat_template_args.get("enable_thinking") is False
        and "preserve_thinking" not in chat_template_args
    ):
        cli_args.chat_template_args = {
            **chat_template_args,
            "preserve_thinking": True,
        }


def _auto_dflash_prefill_step_size(prompt_len: int) -> int:
    _ = prompt_len
    # Hardware note: on this 40-core M3 Max / applegpu_g15s machine, real
    # Pi Mono 10k-token cache-export sweeps put 1024 and 2048-token chunks
    # within noise, so keep the simpler MLX-friendly default.
    return 2048


def _resolve_dflash_prefill_step_size(cli_args: Any, prompt_len: int) -> int:
    raw_value = getattr(cli_args, "prefill_step_size", 0)
    try:
        requested = int(raw_value or 0)
    except (TypeError, ValueError):
        requested = 0
    if requested > 0:
        return requested
    return _auto_dflash_prefill_step_size(prompt_len)


def _read_project_version() -> str:
    try:
        return package_version("dflash-mlx")
    except PackageNotFoundError:
        return "unknown"


def _state_machine_is_terminal(state: Any) -> bool:
    try:
        return state is not None and state[0] is None
    except (TypeError, IndexError):
        return False


class DFlashModelProvider(mlx_server.ModelProvider):
    def load(self, model_path, adapter_path=None, draft_model_path=None):
        requested_model = self._model_map.get(model_path, model_path)
        if self.cli_args.model is not None:
            model_ref = self.cli_args.model
        elif requested_model == "default_model":
            raise ValueError(
                "A model path has to be given as a CLI argument or in the HTTP request"
            )
        else:
            model_ref = requested_model

        if draft_model_path == "default_model":
            draft_ref = self.cli_args.draft_model
        elif draft_model_path is not None:
            draft_ref = draft_model_path
        else:
            draft_ref = None
        resolved_draft_ref = resolve_optional_draft_ref(model_ref, draft_ref)
        model_key = _dflash_model_key(model_ref, resolved_draft_ref, self.cli_args)

        if self.model_key == model_key:
            return self.model, self.tokenizer

        self.model = None
        self.tokenizer = None
        self.model_key = None
        self.draft_model = None

        model, tokenizer, draft_model, resolved_draft_ref = load_runtime_components(
            model_ref=model_ref,
            draft_ref=draft_ref,
            quantize_kv_cache=getattr(self.cli_args, "quantize_kv_cache", False),
            kv_cache_bits=_kv_cache_bits(self.cli_args),
            kv_cache_group_size=_kv_cache_group_size(self.cli_args),
        )
        model_key = _dflash_model_key(model_ref, resolved_draft_ref, self.cli_args)

        if self.cli_args.chat_template:
            tokenizer.chat_template = self.cli_args.chat_template
        if self.cli_args.use_default_chat_template and tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template

        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.model_key = model_key
        self.is_batchable = False
        return self.model, self.tokenizer


class DFlashResponseGenerator(mlx_server.ResponseGenerator):
    @staticmethod
    def _build_generation_context(tokenizer, prompt, stop_words=None, sequences=None):
        if _STATEFUL_SERVER_API:
            return mlx_server.GenerationContext(
                has_thinking=tokenizer.has_thinking,
                has_tool_calling=tokenizer.has_tool_calling,
                tool_parser=tokenizer.tool_parser,
                sequences=sequences or {},
                prompt=prompt,
            )
        return mlx_server.GenerationContext(
            has_tool_calling=tokenizer.has_tool_calling,
            tool_call_start=tokenizer.tool_call_start,
            tool_call_end=tokenizer.tool_call_end,
            tool_parser=tokenizer.tool_parser,
            has_thinking=tokenizer.has_thinking,
            think_start_id=tokenizer.think_start_id,
            think_end=tokenizer.think_end,
            think_end_id=tokenizer.think_end_id,
            eos_token_ids=tokenizer.eos_token_ids,
            stop_token_sequences=[
                tokenizer.encode(stop_word, add_special_tokens=False)
                for stop_word in (stop_words or [])
            ],
            prompt=prompt,
        )

    @staticmethod
    def _make_response(
        *,
        text: str,
        token: int,
        state: Optional[str],
        match: Optional[tuple[int, ...]],
        finish_reason: Optional[str],
    ):
        if _STATEFUL_SERVER_API:
            return mlx_server.Response(
                text,
                token,
                state or "normal",
                match,
                0.0,
                finish_reason,
                (),
            )
        return mlx_server.Response(
            text,
            token,
            0.0,
            finish_reason,
            (),
        )

    def _serve_single(self, request):
        request_tuple = request
        rqueue, request, args = request_tuple
        cli_args = self.model_provider.cli_args

        if args.max_tokens <= 256:
            request_id = f"{time.time_ns():x}"
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] fast-path AR | max_tokens={args.max_tokens}\n"
            )
            sys.stderr.flush()
            _append_dflash_metrics_event(
                cli_args,
                {
                    "event": "fast_path_ar",
                    "request_id": request_id,
                    "max_tokens": int(args.max_tokens),
                    "reason": "max_tokens <= 256",
                },
            )
            saved_draft_model = self.model_provider.draft_model
            try:
                self.model_provider.draft_model = None
                return super()._serve_single((rqueue, request, args))
            finally:
                self.model_provider.draft_model = saved_draft_model

        try:
            model = self.model_provider.model
            tokenizer = self.model_provider.tokenizer
            draft_model = self.model_provider.draft_model
            tokenized = self._tokenize(tokenizer, request, args)
            if isinstance(tokenized, tuple):
                prompt, segments, segment_types, initial_state = tokenized
            else:
                prompt = tokenized
                segments = [prompt]
                segment_types = ["assistant"]
                initial_state = "normal"

            sm = None
            sm_state = None
            sequences = {}
            if _STATEFUL_SERVER_API and hasattr(self, "_make_state_machine"):
                sm, sequences = self._make_state_machine(
                    self.model_provider.model_key,
                    tokenizer,
                    args.stop_words,
                    initial_state=initial_state,
                )
                sm_state = sm.make_state()

            ctx = self._build_generation_context(
                tokenizer,
                prompt,
                stop_words=args.stop_words,
                sequences=sequences,
            )
            rqueue.put(ctx)

            if args.seed is not None:
                mx.random.seed(args.seed)

            stop_token_ids = get_stop_token_ids(tokenizer)
            detokenizer = tokenizer.detokenizer
            if hasattr(detokenizer, "reset"):
                detokenizer.reset()
            eos_token_ids = set(int(token_id) for token_id in tokenizer.eos_token_ids)
            pending_token: Optional[int] = None
            pending_text = ""
            pending_state: Optional[str] = "normal"
            pending_match: Optional[tuple[int, ...]] = None
            pending_finish_reason: Optional[str] = None
            first_token_flushed = False
            finish_reason: Optional[str] = None
            summary_event: Optional[dict[str, Any]] = None
            request_id = f"{time.time_ns():x}"
            metrics_log_enabled = _dflash_metrics_log_path(cli_args) is not None
            metrics_token_interval = (
                _dflash_metrics_token_interval() if metrics_log_enabled else 0
            )
            request_start_ns = time.perf_counter_ns()
            prefill_elapsed_s = 0.0
            live_tok_s = 0.0
            live_token_count = 0
            live_acceptance_pct = 0.0
            live_prompt_len = len(prompt)
            printed_prefill_progress = False
            stable_cache_build_us = 0.0
            use_dflash_prompt_cache = _use_dflash_prompt_cache(
                self.model_provider.cli_args
            )
            prefill_step_size = _resolve_dflash_prefill_step_size(cli_args, len(prompt))
            using_stable_prompt_cache = False
            if use_dflash_prompt_cache:
                stable_prompt, active_prompt_tail = _select_dflash_stable_prompt_prefix(
                    prompt,
                    segments,
                    segment_types,
                )
                prompt_cache = None
                prompt_rest = prompt
                prompt_cache_count = 0

                if stable_prompt and active_prompt_tail and draft_model is not None:
                    stable_cache, stable_rest, stable_cache_count = (
                        _fetch_dflash_prompt_cache(
                            self.prompt_cache,
                            self.model_provider.model_key,
                            stable_prompt,
                            allow_exact=True,
                        )
                    )
                    if stable_rest:
                        if stable_cache_count > 0:
                            sys.stderr.write(
                                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] stable prompt cache: "
                                f"{stable_cache_count}/{len(stable_prompt)} cached; "
                                f"prefill stable suffix {len(stable_rest)} tokens\n"
                            )
                            sys.stderr.flush()
                        stable_summary: Optional[dict[str, Any]] = None
                        for stable_event in stream_dflash_generate(
                            target_model=model,
                            tokenizer=tokenizer,
                            draft_model=draft_model,
                            prompt="",
                            max_new_tokens=0,
                            use_chat_template=False,
                            stop_token_ids=stop_token_ids,
                            prompt_tokens_override=stable_rest,
                            quantize_kv_cache=getattr(
                                self.model_provider.cli_args,
                                "quantize_kv_cache",
                                False,
                            ),
                            kv_cache_bits=_kv_cache_bits(self.model_provider.cli_args),
                            kv_cache_group_size=_kv_cache_group_size(
                                self.model_provider.cli_args
                            ),
                            prefill_step_size=_resolve_dflash_prefill_step_size(
                                cli_args,
                                len(stable_prompt),
                            ),
                            block_tokens=getattr(
                                self.model_provider.cli_args,
                                "block_tokens",
                                None,
                            ),
                            prompt_cache=stable_cache,
                            prompt_cache_count=stable_cache_count,
                            return_prompt_cache=True,
                        ):
                            if stable_event.get("event") in (
                                "prefill",
                                "prefill_progress",
                            ):
                                processed = int(
                                    stable_event.get(
                                        "tokens_processed",
                                        stable_event.get(
                                            "prompt_token_count",
                                            len(stable_prompt),
                                        ),
                                    )
                                )
                                total = int(
                                    stable_event.get(
                                        "tokens_total",
                                        stable_event.get(
                                            "prompt_token_count",
                                            len(stable_prompt),
                                        ),
                                    )
                                )
                                elapsed_s = (
                                    time.perf_counter_ns() - request_start_ns
                                ) / 1e9
                                sys.stderr.write(
                                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] stable prefill: "
                                    f"{processed}/{total} tokens | {elapsed_s:.1f}s\n"
                                )
                                sys.stderr.flush()
                                _append_dflash_metrics_event(
                                    cli_args,
                                    {
                                        "event": "stable_prefill_progress",
                                        "request_id": request_id,
                                        "prompt_tokens": len(prompt),
                                        "tokens_processed": processed,
                                        "tokens_total": total,
                                        "elapsed_ms": elapsed_s * 1_000.0,
                                    },
                                )
                            if stable_event.get("event") == "summary":
                                stable_summary = stable_event
                                stable_cache_build_us = float(
                                    stable_event.get("elapsed_us", 0.0) or 0.0
                                )
                        stable_cache = (
                            stable_summary.get("prompt_cache")
                            if stable_summary is not None
                            else None
                        )
                        if stable_cache is not None:
                            self.prompt_cache.insert_cache(
                                self.model_provider.model_key,
                                stable_prompt,
                                stable_cache,
                                cache_type="user",
                            )
                            stable_cache, stable_rest, stable_cache_count = (
                                _fetch_dflash_prompt_cache(
                                    self.prompt_cache,
                                    self.model_provider.model_key,
                                    stable_prompt,
                                    allow_exact=True,
                                )
                            )

                    if stable_cache is not None and not stable_rest:
                        prompt_cache = stable_cache
                        prompt_rest = active_prompt_tail
                        prompt_cache_count = len(stable_prompt)
                        using_stable_prompt_cache = True

                if not using_stable_prompt_cache:
                    prompt_cache, prompt_rest, prompt_cache_count = (
                        _fetch_dflash_prompt_cache(
                            self.prompt_cache,
                            self.model_provider.model_key,
                            prompt,
                        )
                    )

                if prompt_cache_count > 0:
                    sys.stderr.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prompt cache: "
                        f"{prompt_cache_count}/{len(prompt)} cached; "
                        f"prefill suffix {len(prompt_rest)} tokens\n"
                    )
                    sys.stderr.flush()
            else:
                prompt_cache = None
                prompt_rest = prompt
                prompt_cache_count = 0
            ctx.prompt_cache_count = prompt_cache_count
            _append_dflash_metrics_event(
                cli_args,
                {
                    "event": "request_start",
                    "request_id": request_id,
                    "prompt_tokens": len(prompt),
                    "cached_prompt_tokens": prompt_cache_count,
                    "uncached_prompt_tokens": len(prompt_rest),
                    "using_stable_prompt_cache": bool(using_stable_prompt_cache),
                    "max_tokens": int(args.max_tokens),
                    "block_tokens": getattr(cli_args, "block_tokens", None),
                    "prefill_step_size": prefill_step_size,
                    "quantize_kv_cache": bool(
                        getattr(cli_args, "quantize_kv_cache", False)
                    ),
                    "kv_cache_bits": _kv_cache_bits(cli_args),
                    "kv_cache_group_size": _kv_cache_group_size(cli_args),
                },
            )

            event_iter = stream_dflash_generate(
                target_model=model,
                tokenizer=tokenizer,
                draft_model=draft_model,
                prompt="",
                max_new_tokens=args.max_tokens,
                use_chat_template=False,
                stop_token_ids=stop_token_ids,
                prompt_tokens_override=prompt_rest,
                quantize_kv_cache=getattr(cli_args, "quantize_kv_cache", False),
                kv_cache_bits=_kv_cache_bits(cli_args),
                kv_cache_group_size=_kv_cache_group_size(cli_args),
                prefill_step_size=prefill_step_size,
                block_tokens=getattr(cli_args, "block_tokens", None),
                prompt_cache=prompt_cache,
                prompt_cache_count=prompt_cache_count,
                return_prompt_cache=use_dflash_prompt_cache,
            )

            try:
                for event in event_iter:
                    if event.get("event") in ("prefill", "prefill_progress"):
                        processed = int(
                            event.get(
                                "tokens_processed",
                                event.get("prompt_token_count", len(prompt)),
                            )
                        )
                        total = int(
                            event.get(
                                "tokens_total",
                                event.get("prompt_token_count", len(prompt)),
                            )
                        )
                        elapsed_s = (time.perf_counter_ns() - request_start_ns) / 1e9
                        if event.get("event") == "prefill_progress":
                            sys.stderr.write(
                                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefill: {processed}/{total} tokens | {elapsed_s:.1f}s\n"
                            )
                            sys.stderr.flush()
                            _append_dflash_metrics_event(
                                cli_args,
                                {
                                    "event": "prefill_progress",
                                    "request_id": request_id,
                                    "prompt_tokens": len(prompt),
                                    "cached_prompt_tokens": prompt_cache_count,
                                    "tokens_processed": processed,
                                    "tokens_total": total,
                                    "elapsed_ms": elapsed_s * 1_000.0,
                                    "prefill_step_size": event.get(
                                        "prefill_step_size"
                                    ),
                                },
                            )
                            printed_prefill_progress = True
                        else:
                            prefill_elapsed_s = elapsed_s
                            if not printed_prefill_progress:
                                sys.stderr.write(
                                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefill: {processed}/{total} tokens | {elapsed_s:.1f}s\n"
                                )
                                sys.stderr.flush()
                            _append_dflash_metrics_event(
                                cli_args,
                                {
                                    "event": "prefill_done",
                                    "request_id": request_id,
                                    "prompt_tokens": len(prompt),
                                    "cached_prompt_tokens": prompt_cache_count,
                                    "tokens_processed": processed,
                                    "tokens_total": total,
                                    "elapsed_ms": elapsed_s * 1_000.0,
                                    "prefill_step_size": event.get(
                                        "prefill_step_size"
                                    ),
                                },
                            )
                        continue
                    if event.get("event") != "token":
                        if event.get("event") == "summary":
                            summary_event = event
                            generated_token_ids = list(event.get("generated_token_ids", []) or [])
                            if generated_token_ids:
                                last_token = int(generated_token_ids[-1])
                                if last_token in eos_token_ids:
                                    finish_reason = "stop"
                                elif int(event.get("generation_tokens", 0)) >= int(args.max_tokens):
                                    finish_reason = "length"
                                else:
                                    finish_reason = "stop"
                            else:
                                finish_reason = "stop"
                        elif event.get("event") == "adaptive_fallback":
                            elapsed_s = (time.perf_counter_ns() - request_start_ns) / 1e9
                            fallback_generated_tokens = int(
                                event.get("generated_tokens", live_token_count) or 0
                            )
                            fallback_decode_tps = fallback_generated_tokens / max(
                                0.001,
                                elapsed_s - prefill_elapsed_s,
                            )
                            recent_tpc = float(
                                event.get("recent_tokens_per_cycle", 0.0) or 0.0
                            )
                            min_tpc = float(
                                event.get("min_tokens_per_cycle", 0.0) or 0.0
                            )
                            sys.stderr.write(
                                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] adaptive fallback: "
                                f"decode: {fallback_decode_tps:.1f} tok/s | "
                                f"{fallback_generated_tokens} tokens | "
                                f"{int(event.get('cycles_completed', 0) or 0)} cycles | "
                                f"recent {recent_tpc:.1f} tok/cycle < {min_tpc:.1f} | "
                                f"cooldown {int(event.get('cooldown_tokens', 0) or 0)} tokens | "
                                f"reprobe block {int(event.get('reprobe_block_tokens', 0) or 0)}\n"
                            )
                            sys.stderr.flush()
                            _append_dflash_metrics_event(
                                cli_args,
                                {
                                    "event": "adaptive_fallback",
                                    "request_id": request_id,
                                    "prompt_tokens": len(prompt),
                                    "cached_prompt_tokens": prompt_cache_count,
                                    "generated_tokens": fallback_generated_tokens,
                                    "elapsed_ms": elapsed_s * 1_000.0,
                                    "decode_tps": fallback_decode_tps,
                                    "cycles_completed": int(
                                        event.get("cycles_completed", 0) or 0
                                    ),
                                    "recent_tokens_per_cycle": recent_tpc,
                                    "min_tokens_per_cycle": min_tpc,
                                    "cooldown_tokens": int(
                                        event.get("cooldown_tokens", 0) or 0
                                    ),
                                    "reprobe_block_tokens": int(
                                        event.get("reprobe_block_tokens", 0) or 0
                                    ),
                                    "reason": event.get("reason"),
                                },
                            )
                        continue

                    token = int(event["token_id"])
                    live_token_count += 1
                    live_acceptance_pct = float(event.get("acceptance_ratio", 0.0) or 0.0) * 100.0
                    elapsed_s = (time.perf_counter_ns() - request_start_ns) / 1e9
                    live_tok_s = live_token_count / max(0.001, elapsed_s - prefill_elapsed_s)
                    if (
                        metrics_token_interval > 0
                        and live_token_count % metrics_token_interval == 0
                    ):
                        _append_dflash_metrics_event(
                            cli_args,
                            {
                                "event": "decode_progress",
                                "request_id": request_id,
                                "prompt_tokens": len(prompt),
                                "generated_tokens": live_token_count,
                                "elapsed_ms": elapsed_s * 1_000.0,
                                "decode_tps": live_tok_s,
                                "acceptance_ratio": float(
                                    event.get("acceptance_ratio", 0.0) or 0.0
                                ),
                                "cycles_completed": int(
                                    event.get("cycles_completed", 0) or 0
                                ),
                            },
                        )
                    if live_token_count % 2048 == 0:
                        sys.stderr.write(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] decode: {live_tok_s:.1f} tok/s | {live_acceptance_pct:.1f}% accepted | "
                            f"{live_token_count} tokens | {elapsed_s:.1f}s | "
                            f"prompt: {live_prompt_len} tokens\n"
                        )
                        sys.stderr.flush()
                    current_state = "normal"
                    match_sequence: Optional[tuple[int, ...]] = None
                    token_finish_reason: Optional[str] = None
                    if sm is not None:
                        if _state_machine_is_terminal(sm_state):
                            break
                        sm_state, match_sequence, current_state = sm.match(sm_state, token)
                        if match_sequence is not None and current_state is None:
                            token_finish_reason = "stop"

                    text = ""
                    if token not in eos_token_ids:
                        detokenizer.add_token(token)
                        text = detokenizer.last_segment

                    if not first_token_flushed:
                        immediate_finish_reason = token_finish_reason
                        if immediate_finish_reason is None:
                            if token in eos_token_ids:
                                immediate_finish_reason = "stop"
                            elif live_token_count >= int(args.max_tokens):
                                immediate_finish_reason = "length"
                        rqueue.put(
                            self._make_response(
                                text=text,
                                token=token,
                                state=current_state or "normal",
                                match=match_sequence,
                                finish_reason=immediate_finish_reason,
                            )
                        )
                        first_token_flushed = True
                        if ctx._should_stop or immediate_finish_reason is not None:
                            break
                        continue

                    if pending_token is not None:
                        rqueue.put(
                            self._make_response(
                                text=pending_text,
                                token=pending_token,
                                state=pending_state,
                                match=pending_match,
                                finish_reason=pending_finish_reason,
                            )
                        )

                    pending_token = token
                    pending_text = text
                    pending_state = current_state or "normal"
                    pending_match = match_sequence
                    pending_finish_reason = token_finish_reason

                    if ctx._should_stop or token_finish_reason is not None:
                        break
            finally:
                event_iter.close()

            detokenizer.finalize()
            tail = detokenizer.last_segment
            if pending_token is not None:
                rqueue.put(
                    self._make_response(
                        text=pending_text + tail,
                        token=pending_token,
                        state=pending_state,
                        match=pending_match,
                        finish_reason=finish_reason or pending_finish_reason,
                    )
                )

            if summary_event is not None:
                generation_tokens = int(summary_event.get("generation_tokens", 0) or 0)
                elapsed_us = float(summary_event.get("elapsed_us", 0.0) or 0.0)
                phase_timings_us = dict(summary_event.get("phase_timings_us") or {})
                prefill_us = float(phase_timings_us.get("prefill", 0.0) or 0.0)
                elapsed_us += stable_cache_build_us
                prefill_us += stable_cache_build_us
                decode_s = max(0.0, (elapsed_us - prefill_us) / 1_000_000.0)
                tok_s = (generation_tokens / decode_s) if decode_s > 0.0 else 0.0
                acceptance_pct = float(summary_event.get("acceptance_ratio", 0.0) or 0.0) * 100.0
                draft_acceptance_pct = (
                    float(summary_event.get("draft_acceptance_ratio", 0.0) or 0.0)
                    * 100.0
                )
                cycles_completed = int(summary_event.get("cycles_completed", 0) or 0)
                tokens_per_cycle = float(
                    summary_event.get("tokens_per_cycle", 0.0) or 0.0
                )
                total_s = elapsed_us / 1_000_000.0
                sys.stderr.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] decode: {tok_s:.1f} tok/s | {acceptance_pct:.1f}% accepted | "
                    f"draft_accept {draft_acceptance_pct:.1f}% | "
                    f"{cycles_completed} cycles | {tokens_per_cycle:.1f} tok/cycle | "
                    f"{generation_tokens} tokens | {total_s:.1f}s | "
                    f"prompt: {len(prompt)} tokens\n"
                )
                sys.stderr.flush()
                _append_dflash_metrics_event(
                    cli_args,
                    _build_dflash_metrics_record(
                        request_id=request_id,
                        summary_event=summary_event,
                        prompt_len=len(prompt),
                        finish_reason=finish_reason,
                        prompt_cache_count=prompt_cache_count,
                        stable_cache_build_us=stable_cache_build_us,
                        using_stable_prompt_cache=using_stable_prompt_cache,
                    ),
                )
                returned_prompt_cache = summary_event.get("prompt_cache")
                if (
                    use_dflash_prompt_cache
                    and returned_prompt_cache is not None
                ):
                    generated_token_ids = list(
                        summary_event.get("generated_token_ids", []) or []
                    )
                    self.prompt_cache.insert_cache(
                        self.model_provider.model_key,
                        prompt + [int(token_id) for token_id in generated_token_ids],
                        returned_prompt_cache,
                        cache_type="user" if using_stable_prompt_cache else "assistant",
                    )

            rqueue.put(None)
        except Exception as e:
            rqueue.put(e)


class DFlashAPIHandler(mlx_server.APIHandler):
    def handle_completion(self, request, stop_words):
        try:
            return super().handle_completion(request, stop_words)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
            return
        except ValueError as e:
            logging.warning("Tool parser error (likely malformed tool call): %s", e)
            self.close_connection = True
            return

    def generate_response(self, *args, **kwargs):
        response = super().generate_response(*args, **kwargs)
        served_model = (
            self.response_generator.model_provider.model_key[0]
            if self.response_generator.model_provider.model_key is not None
            else None
        )
        if served_model:
            response["model"] = served_model
        return response


def _print_startup_banner(
    *,
    port: int,
    model_provider: DFlashModelProvider,
) -> None:
    dflash_version = _read_project_version()
    server_name = getattr(mlx_server, "__name__", "mlx_lm.server")
    target_ref = None
    draft_ref = None
    if model_provider.model_key is not None:
        target_ref = model_provider.model_key[0]
        draft_ref = model_provider.model_key[2]
    target_ref = target_ref or model_provider.cli_args.model or "(lazy — provide via HTTP)"
    if not draft_ref:
        draft_ref = model_provider.cli_args.draft_model or "(lazy — resolved on first request)"
        draft_suffix = ""
    elif model_provider.cli_args.draft_model:
        draft_suffix = " (explicit)"
    else:
        draft_suffix = " (auto-detected)"
    raw_lines = [
        f"DFlash v{dflash_version} - speculative decoding engine",
        f"Target: {target_ref}",
        f"Draft:  {draft_ref}{draft_suffix}",
        "Mode:   DFlash (speculative decoding active)",
        f"Server: {server_name} on port {port}",
    ]

    width = max(len(line) for line in raw_lines)
    use_color = sys.stderr.isatty()
    reset = "\033[0m" if use_color else ""
    border_color = "\033[38;5;39m" if use_color else ""
    title_color = "\033[1;38;5;51m" if use_color else ""
    body_color = "\033[38;5;252m" if use_color else ""

    def style(text: str, color: str) -> str:
        return f"{color}{text}{reset}" if use_color else text

    border = style("+" + "-" * (width + 2) + "+", border_color)
    lines = [border]
    for index, raw_line in enumerate(raw_lines):
        padded = f"| {raw_line.ljust(width)} |"
        lines.append(style(padded, title_color if index == 0 else body_color))
    lines.append(border)

    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()


def _run_with_dflash_server(host: str, port: int, model_provider: DFlashModelProvider):
    group = mx.distributed.init()
    if model_provider.cli_args.model is not None:
        model_provider.load("default_model", None, "default_model")
    prompt_cache = mlx_server.LRUPromptCache(model_provider.cli_args.prompt_cache_size)
    disk_cache_dir = _dflash_prompt_cache_dir(model_provider.cli_args)
    if disk_cache_dir is not None:
        prompt_cache = DiskBackedPromptCache(
            prompt_cache,
            directory=disk_cache_dir,
            ttl_seconds=max(
                0.0,
                float(model_provider.cli_args.dflash_prompt_cache_ttl_days),
            )
            * 24
            * 60
            * 60,
            max_bytes=(
                int(float(model_provider.cli_args.dflash_prompt_cache_max_disk_gb) * 1e9)
                if model_provider.cli_args.dflash_prompt_cache_max_disk_gb
                else None
            ),
        )
    response_generator = DFlashResponseGenerator(model_provider, prompt_cache)
    if group.rank() == 0:
        _print_startup_banner(port=port, model_provider=model_provider)
        mlx_server._run_http_server(
            host,
            port,
            response_generator,
            handler_class=DFlashAPIHandler,
        )
    else:
        response_generator.join()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DFlash Server.")
    parser.add_argument(
        "--model",
        type=str,
        help="The path or Hugging Face reference for the target model.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the HTTP server (default: 8000)",
    )
    parser.add_argument(
        "--allowed-origins",
        type=lambda x: x.split(","),
        default="*",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--draft-model",
        "--draft",
        dest="draft_model",
        type=str,
        default=None,
        help="Optional DFlash draft model override.",
    )
    parser.add_argument(
        "--dflash-max-ctx",
        type=int,
        default=None,
        help="Maximum prompt token count for DFlash speculative decoding.",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help=argparse.SUPPRESS,
        default=3,
    )
    parser.add_argument(
        "--quantize-draft",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling top-p (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k sampling (default: 0, disables top-k)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Default min-p sampling (default: 0.0, disables min-p)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        default="{}",
        help="""A JSON formatted string of arguments for the tokenizer's apply_chat_template, e.g. '{"enable_thinking":false}'""",
    )
    parser.add_argument(
        "--decode-concurrency",
        type=int,
        default=32,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--prompt-concurrency",
        type=int,
        default=8,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--block-tokens",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quantize-kv-cache",
        action="store_true",
        help="Quantize target full-attention KV cache for long contexts.",
    )
    parser.add_argument(
        "--kv-cache-bits",
        type=int,
        default=8,
        choices=(2, 4, 8),
        help=(
            "Bits for --quantize-kv-cache. Q8 is the measured safe default; "
            "Q4 is experimental and should be checked with evals."
        ),
    )
    parser.add_argument(
        "--kv-cache-group-size",
        type=int,
        default=64,
        choices=(32, 64, 128),
        help="Quantization group size for --quantize-kv-cache.",
    )
    parser.add_argument(
        "--dflash-prompt-cache",
        action="store_true",
        help=(
            "Reuse DFlash target/draft prompt caches across requests. "
            "Use --prompt-cache-size 1 for long single-chat sessions."
        ),
    )
    parser.add_argument(
        "--dflash-prompt-cache-dir",
        type=str,
        default=None,
        help=(
            "Persist DFlash prompt caches to this directory. "
            f"Can also be set with {_PROMPT_CACHE_DIR_ENV}."
        ),
    )
    parser.add_argument(
        "--dflash-prompt-cache-ttl-days",
        type=float,
        default=7.0,
        help="Delete disk prompt caches not used for this many days. 0 disables TTL cleanup.",
    )
    parser.add_argument(
        "--dflash-prompt-cache-max-disk-gb",
        type=float,
        default=0.0,
        help="Maximum disk prompt-cache size in GB. 0 disables size cleanup.",
    )
    parser.add_argument(
        "--dflash-metrics-log",
        type=str,
        default=None,
        help=(
            "Append privacy-safe DFlash session metrics as JSONL. "
            f"Can also be set with {_METRICS_LOG_ENV}."
        ),
    )
    parser.add_argument(
        "--prompt-cache-size",
        type=int,
        default=10,
        help="Maximum number of distinct KV caches to hold in the prompt cache",
    )
    parser.add_argument(
        "--prompt-cache-bytes",
        type=int,
        help="Maximum size in bytes of the KV caches",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.dflash_max_ctx is not None:
        if args.dflash_max_ctx <= 0:
            raise SystemExit("--dflash-max-ctx must be > 0")
        os.environ["DFLASH_MAX_CTX"] = str(args.dflash_max_ctx)
    _stabilize_dflash_prompt_cache_chat_template_args(args)

    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)
        mx.set_cache_limit(wired_limit // 4)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    _run_with_dflash_server(args.host, args.port, DFlashModelProvider(args))


if __name__ == "__main__":
    main()
