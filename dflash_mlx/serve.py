# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import itertools
import logging
import sys
import time
import warnings
from importlib.metadata import PackageNotFoundError, version as package_version

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

from dflash_mlx.server.config import (
    build_parser as _build_parser,
    configure_logging,
    configure_metal_limits,
    normalize_cli_args,
)
from dflash_mlx.server.protocol import (
    STATEFUL_SERVER_API as _STATEFUL_SERVER_API,
    build_generation_context as _build_generation_context,
    match_stream_token as _match_stream_token,
)
from dflash_mlx.bench_logger import (
    enabled as _bench_enabled,
    log_post as _bench_log_post,
)
from dflash_mlx.server.metrics import (
    log_bench_post as _log_bench_post,
    write_summary_line as _write_summary_line,
)
from dflash_mlx.generate import get_stop_token_ids
from dflash_mlx.server.model_provider import (
    DFlashModelProvider,
    wait_for_initial_model_load as _wait_for_initial_model_load,
)
from dflash_mlx.cache.policies import prefix_cache_enabled
from dflash_mlx.engine.config import _resolve_target_fa_window
from dflash_mlx.runtime import stream_dflash_generate
from dflash_mlx.server.prefix_cache_flow import (
    PrefixCacheFlow,
    get_dflash_prefix_cache as _get_dflash_prefix_cache,
    log_prefix_cache_stats,
)
from dflash_mlx.server.prefix_cache_manager import (
    build_prefix_key as _build_prefix_key,
)
from dflash_mlx.server.request_loop import consume_dflash_events


def _read_project_version() -> str:
    try:
        return package_version("dflash-mlx")
    except PackageNotFoundError:
        return "unknown"


_DFLASH_REQUEST_COUNTER = itertools.count(1)


class DFlashResponseGenerator(mlx_server.ResponseGenerator):
    def _serve_single(self, request):
        request_tuple = request
        rqueue, request, args = request_tuple

        request_id = next(_DFLASH_REQUEST_COUNTER)
        bench_active = _bench_enabled()

        if args.max_tokens <= 256:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] fast-path AR | max_tokens={args.max_tokens}\n"
            )
            sys.stderr.flush()
            saved_draft_model = self.model_provider.draft_model
            wall_t0 = time.perf_counter_ns()
            try:
                self.model_provider.draft_model = None
                return super()._serve_single((rqueue, request, args))
            finally:
                self.model_provider.draft_model = saved_draft_model
                if bench_active:
                    wall_ms = (time.perf_counter_ns() - wall_t0) / 1e6
                    _bench_log_post(
                        request_id=request_id,
                        mode_used="ar_fastpath",
                        max_tokens=int(args.max_tokens),
                        wall_ms=wall_ms,
                    )

        try:
            model = self.model_provider.model
            tokenizer = self.model_provider.tokenizer
            draft_model = self.model_provider.draft_model
            tokenized = self._tokenize(tokenizer, request, args)
            if isinstance(tokenized, tuple):
                prompt, _, _, initial_state = tokenized
            else:
                prompt = tokenized
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

            ctx = _build_generation_context(
                tokenizer,
                prompt,
                stop_words=args.stop_words,
                sequences=sequences,
            )
            rqueue.put(ctx)

            if args.seed is not None:
                mx.random.seed(args.seed)

            stop_token_ids = get_stop_token_ids(tokenizer)
            eos_token_ids = set(int(token_id) for token_id in tokenizer.eos_token_ids)
            request_start_ns = time.perf_counter_ns()
            prefix_flow = PrefixCacheFlow.for_request(
                model_provider=self.model_provider,
                draft_model=draft_model,
                tokenizer=tokenizer,
                prompt=prompt,
            )
            ctx.prompt_cache_count = prefix_flow.hit_tokens

            event_iter = stream_dflash_generate(
                target_model=model,
                tokenizer=tokenizer,
                draft_model=draft_model,
                prompt="",
                max_new_tokens=args.max_tokens,
                use_chat_template=False,
                stop_token_ids=stop_token_ids,
                prompt_tokens_override=prompt,
                prefix_snapshot=prefix_flow.snapshot,
                stable_prefix_len=prefix_flow.stable_prefix_len,
            )
            loop_result = consume_dflash_events(
                event_iter=event_iter,
                rqueue=rqueue,
                ctx=ctx,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=int(args.max_tokens),
                eos_token_ids=eos_token_ids,
                request_start_ns=request_start_ns,
                prefix_flow=prefix_flow,
                sm=sm,
                sm_state=sm_state,
                bench_active=bench_active,
                request_id=request_id,
            )
            summary_event = loop_result.summary_event

            if summary_event is not None:
                _write_summary_line(
                    summary_event=summary_event,
                    prompt_token_count=len(prompt),
                )

            if bench_active:
                _log_bench_post(
                    request_id=request_id,
                    summary_event=summary_event,
                    request_start_ns=loop_result.request_start_ns,
                    request_done_ns=time.perf_counter_ns(),
                    first_token_ns=loop_result.first_token_ns,
                    prefill_done_ns=loop_result.prefill_done_ns,
                    prompt_token_count=len(prompt),
                    live_token_count=loop_result.live_token_count,
                    cache_lookup_ms=loop_result.cache_lookup_ms,
                    cache_hit_tokens=loop_result.cache_hit_tokens,
                    cache_insert_ms=loop_result.cache_insert_ms,
                    finish_reason=loop_result.finish_reason,
                    max_tokens=args.max_tokens,
                )
            if hasattr(mx, "get_peak_memory"):
                try:
                    peak_gb = float(mx.get_peak_memory()) / 1e9
                    sys.stderr.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] req#{request_id} peak_memory={peak_gb:.2f} GB\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
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
    target_ref = target_ref or model_provider.cli_args.model or "unknown"
    if not draft_ref:
        raise RuntimeError("DFlash server requires a resolved draft model before startup.")

    if model_provider.cli_args.draft_model:
        draft_suffix = " (explicit)"
    else:
        draft_suffix = " (auto-detected)"
    target_fa_window = _resolve_target_fa_window()
    pc_enabled = prefix_cache_enabled()
    if target_fa_window > 0:
        pc_status = "disabled (--target-fa-window)"
    else:
        pc_status = "enabled" if pc_enabled else "disabled (--no-prefix-cache)"
    target_fa_status = (
        "full KV" if target_fa_window == 0 else f"rotating window {target_fa_window}"
    )
    raw_lines = [
        f"DFlash v{dflash_version} - speculative decoding engine",
        f"Target:       {target_ref}",
        f"Draft:        {draft_ref}{draft_suffix}",
        "Mode:         DFlash (speculative decoding active)",
        f"Prefix cache: {pc_status}",
        f"Target FA KV: {target_fa_status}",
        f"Server:       {server_name} on port {port}",
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
    prompt_cache = mlx_server.LRUPromptCache(model_provider.cli_args.prompt_cache_size)

    response_generator = DFlashResponseGenerator(model_provider, prompt_cache)
    if group.rank() == 0:
        _wait_for_initial_model_load(model_provider, timeout_s=300.0)
        _print_startup_banner(port=port, model_provider=model_provider)
        mlx_server._run_http_server(
            host,
            port,
            response_generator,
            handler_class=DFlashAPIHandler,
        )
    else:
        response_generator.join()

def main() -> None:
    args = normalize_cli_args(_build_parser().parse_args())
    configure_metal_limits()
    configure_logging(args.log_level)
    _run_with_dflash_server(args.host, args.port, DFlashModelProvider(args))

if __name__ == "__main__":
    main()
