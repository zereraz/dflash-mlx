<p align="center">
  <h1 align="center">dflash-mlx</h1>
  <p align="center">DFlash speculative decoding for Apple Silicon</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/MLX-stock-red" alt="Stock MLX">
</p>

Paper: [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) (Chen et al., 2026)

Block-diffusion draft generates 16 tokens in one pass. Target verifies in one pass. Output is lossless — every emitted token is verified against the target model before it is committed.

https://github.com/user-attachments/assets/a9be2b48-3264-4970-b836-c876b0b7fdda
 
## How it works

- A small draft model (~1B params) generates 16 tokens in parallel with block diffusion.
- The target model verifies those 16 tokens in a single forward pass.
- Greedy acceptance keeps the correct prefix and rejects the rest.
- Lossless: every emitted token is the target model's greedy argmax at verification time. Output can still differ from pure AR because of MLX dispatch divergence, but no unverified token is ever emitted.
- Uses stock MLX plus a small number of targeted kernels where rollback and long-context verify need tighter numerical control.

## Technical details

- **Tape-replay rollback**: instead of snapshotting and restoring the full GatedDeltaNet state, dflash-mlx records an innovation tape during verify and replays only the accepted steps through a custom Metal kernel. Keeps rollback cost low and preserves acceptance over long generations.
- **JIT SDPA 2-pass**: long-context verify (`N >= 1024`) uses a custom Metal attention kernel that stays numerically aligned with stock MLX attention.
- **Numerical coherence**: bf16-sensitive paths, including recurrent state replay and small projections, are stabilized across speculative cycles so accepted tokens stay consistent.

## Benchmarks

| Model | Tokens | Baseline | DFlash | Speedup | Acceptance |
|-------|--------|----------|--------|---------|------------|
| Qwen3.5-4B | 1024 | 53.48 tok/s | 197.49 tok/s | 3.69x | 88.67% |
| Qwen3.5-4B | 2048 | 53.74 tok/s | 219.83 tok/s | 4.10x | 89.26% |
| Qwen3.5-4B | 4096 | 53.58 tok/s | 155.19 tok/s | 2.83x | 87.55% |
| Qwen3.5-9B | 1024 | 31.09 tok/s | 127.47 tok/s | 4.10x | 88.96% |
| Qwen3.5-9B | 2048 | 30.96 tok/s | 127.07 tok/s | 4.13x | 89.36% |
| Qwen3.5-9B | 4096 | 31.58 tok/s | 103.90 tok/s | 3.29x | 88.57% |
| Qwen3.5-27B-4bit | 1024 | 33.24 tok/s | 65.80 tok/s | 1.98x | 89.45% |
| Qwen3.5-27B-4bit | 2048 | 32.35 tok/s | 62.78 tok/s | 1.90x | 89.11% |
| Qwen3.5-27B-4bit | 4096 | 29.38 tok/s | 48.89 tok/s | 1.66x | 87.99% |
| Qwen3.5-35B-A3B-4bit | 1024 | 139.97 tok/s | 242.92 tok/s | 1.74x | 89.26% |
| Qwen3.5-35B-A3B-4bit | 2048 | 142.12 tok/s | 240.21 tok/s | 1.69x | 88.67% |
| Qwen3.5-35B-A3B-4bit | 4096 | 140.73 tok/s | 189.62 tok/s | 1.35x | 86.96% |

### Methodology

Hardware: Apple M5 Max, 64GB unified memory. MLX 0.31.1 from the stock pip install.

Protocol: stock `mlx_lm.stream_generate` on a pristine target model vs stock MLX plus the local DFlash runtime, measured sequentially. `3` repeats, median reported, `10s` cooldown between measured runs.

Generation: prompt `"The function $f$ satisfies the functional equation \[ f(x) + f(y) = f(x + y) - xy - 1 \] for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \boxed{}."` with chat templates enabled by default, `--no-eos`, and post-prefill tok/s as the primary metric.

Full per-run JSON reports are available in [`benchmark/results/`](benchmark/results/).

## Install

```bash
pip install dflash-mlx

# or with pipx
pipx install dflash-mlx
```

`dflash-serve` wraps `mlx_lm.server` for full OpenAI-compatible serving semantics, including tools, reasoning, and streaming, while using the DFlash runtime as the generation engine.

## Quick start

```bash
PROMPT='The function $f$ satisfies the functional equation \[ f(x) + f(y) = f(x + y) - xy - 1 \] for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \boxed{}.'

# Generate — draft auto-resolved
dflash --model Qwen/Qwen3.5-9B --prompt "$PROMPT"

# Explicit draft model
dflash --model Qwen/Qwen3.5-9B --draft z-lab/Qwen3.5-9B-DFlash --prompt "$PROMPT"

# Server
dflash-serve --model Qwen/Qwen3.5-9B --port 8000

# Disable visible thinking/reasoning on models that support it
dflash-serve --model Qwen/Qwen3.5-9B --port 8000 \
  --chat-template-args '{"enable_thinking": false}'

# Raise the DFlash fallback threshold for longer prompts
dflash-serve --model mlx-community/Qwen3.5-35B-A3B-4bit --port 8000 \
  --chat-template-args '{"enable_thinking": false}' \
  --dflash-max-ctx 16384

# Benchmark
dflash-benchmark --model Qwen/Qwen3.5-9B --draft z-lab/Qwen3.5-9B-DFlash \
  --prompt "$PROMPT" --max-tokens 1024 --repeat 3 --no-eos

# Live demo — baseline vs DFlash side-by-side
PYTHONPATH=. python3 -m examples.demo --mode dflash \
  --target-model Qwen/Qwen3.5-9B --draft-model z-lab/Qwen3.5-9B-DFlash \
  --prompt "$PROMPT" --max-tokens 2048 --no-eos
```

- Compatible with Open WebUI, Continue, OpenCode, aider, and other OpenAI-compatible clients
- Streaming SSE support
- `dflash-serve` requires a supported DFlash draft model (auto-detected from the registry or passed explicitly with `--draft`)

## Tested models

Any model with a DFlash draft on HuggingFace should work.

Optimized for Qwen3.5 models (hybrid GatedDeltaNet + attention architecture). Qwen3 (pure attention) models work, but without the precision benefits of tape-replay rollback.

| Target | Draft |
|--------|-------|
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | [z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) |
| [mlx-community/Qwen3.5-27B-4bit](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit) | [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) |
| [mlx-community/Qwen3.5-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) | [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) |

Models without a matching DFlash draft are rejected. Pass `--draft` explicitly if you want to override the registry.

## Features

- **Auto draft resolution** — no manual `--draft` flag needed
- **Streaming** — token-by-token output in the CLI and server
- **Chat templates** — enabled by default
- **Recurrent rollback** — `RecurrentRollbackCache` keeps GatedDeltaNet state coherent across speculative verify and rollback

## Roadmap

- Sustained acceptance at 4096+ tokens — draft KV cache window scaling and long-context verify optimization
- Draft model distillation and compression for faster draft forward

## Citation

```bibtex
@misc{chen2026dflash,
  title={DFlash: Block Diffusion for Flash Speculative Decoding},
  author={Jian Chen and Yesheng Liang and Zhijian Liu},
  year={2026},
  eprint={2602.06036},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2602.06036}
}
```

## License

MIT
