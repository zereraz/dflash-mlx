# DFlash MLX Prefill Performance Notes

Date: 2026-04-27

These notes capture the current understanding of where prefill time goes, what we have measured, and which optimizations are real versus still speculative.

## Short Version

- Decode is usually bandwidth-bound because each generated token rereads model weights for a tiny amount of work.
- Prefill is more compute-friendly because it runs many prompt tokens together, but it still pays real memory traffic for quantized weights, dequant scales, activations, KV/GDN state, and unfused intermediate tensors.
- In our measured Qwen target model, the main first-prefill bottleneck is the MLP block, not Python or chat-template handling.
- The best confirmed first-prefill optimization so far is `--hybrid-mlp` plus `--hybrid-gdn-qkv-z-out-proj`, default middle-chunk no-logits, sliced capture, cache-only final-MLP skip, cache-only final-attention K/V update, and the same final-layer skip in no-logits hidden-capture chunks: about 1.11-1.16x versus q4 in controlled 10k-30k A/B runs and real Pi-shaped 10k prompt-cache builds, plus about 2.7-3.4% on checkpoint-heavy cache builds versus disabling final-layer skips.
- The biggest practical chat UX win is prompt prefix caching plus checkpoints: repeated or mostly-shared chat history can skip large already-computed prefixes.

## Model Shape We Are Optimizing

The target model is a Qwen hybrid architecture:

- 64 total layers
- 48 linear-attention/GDN layers
- 16 full-attention layers
- hidden size 5120
- MLP intermediate size 17408
- bf16 activations
- quantized weights in the deployed model

The draft model is much smaller and matters mostly for decode. It does not explain slow first-prefill of the target model.

## What Actually Takes Time In Prefill

A synchronized layer profile on a 512-token prefill showed roughly:

- MLP: 2319 ms across 64 layers, about 64%
- Linear attention/GDN: 922 ms across 48 layers, about 25%
- Full attention: 281 ms across 16 layers, about 8%
- Norm/residual and small ops: 73 ms, about 2%

So for first-prefill, the central question is: how do we make the 64 MLP passes faster without changing model outputs?

Important code paths:

- `dflash_mlx/runtime.py`: `stream_dflash_generate()`
- `dflash_mlx/runtime.py`: `target_forward_with_hidden_states()`
- `dflash_mlx/runtime.py`: target model layer forward, MLP, attention, cache mutation
- `dflash_mlx/serve.py`: prefill progress logging and prompt-cache event handling
- `benchmark/profile_variants.py`: benchmark harness and MLX/Metal capture hooks

## Why Bandwidth Can Still Matter In Prefill

Prefill is not the same as decode. With a large token chunk, matrix-matrix multiplies reuse weights much better than decode's one-token matrix-vector work.

But memory traffic still matters because every layer repeatedly does:

- read quantized weights
- read scales/biases for dequantization
- dequantize or fuse dequantization into matmul
- read and write large activation tensors
- write KV or GDN recurrent state
- sometimes materialize intermediate tensors when operations are not fused

The better statement is:

> Prefill is more compute-heavy than decode, but our prefill can still be slowed by quantization/dequantization overhead and memory traffic around the dominant MLP matmuls.

## What We Have Confirmed

### Apple/MLX MLP Shape Scaling

We added a focused probe:

- `benchmark/probe_apple_prefill.py`
- result: `benchmark/results/apple-prefill-mlp-scaling.json`

This benchmarks one real target-model MLP from the Qwen model across decode-like and prefill-like token counts.

Result on Apple M3 Max:

| Tokens Through One MLP | q4 Best | bf16 Dequant MLP Best | Winner |
| ---: | ---: | ---: | --- |
| 1 | 0.94 ms | 1.63 ms | q4 |
| 4 | 1.43 ms | 3.00 ms | q4 |
| 16 | 1.87 ms | 2.94 ms | q4 |
| 64 | 3.43 ms | 3.47 ms | tie |
| 256 | 12.46 ms | 11.27 ms | bf16 |
| 512 | 24.57 ms | 21.70 ms | bf16 |
| 1024 | 49.06 ms | 43.30 ms | bf16 |
| 2048 | 98.15 ms | 85.87 ms | bf16 |

Interpretation:

- Decode-like small-M work prefers q4 because weight bandwidth dominates and the matmul is too small to fully amortize bf16 weight traffic.
- Prefill-like large-M work can prefer bf16 dequantized MLP because the larger matmul better amortizes weight reads, and q4 dequant/scales overhead becomes visible.
- This is direct evidence that Apple/MLX has shape-dependent behavior: the best representation for decode is not necessarily the best representation for prefill.

Whole-model check:

- 4096-token q4/default prefill: 19.67 s
- 4096-token dequantized-MLP prefill: 19.61 s

So the single-MLP effect is real, but this latest 4096-token whole-model A/B did not show a stable end-to-end win. Earlier 10k runs showed about 5-6%, but global MLP dequantization is the wrong public knob because it can hurt decode-like small `M`.

Follow-up code change:

- Added `--hybrid-mlp`.
- It keeps q4 MLP projections for decode/small verify shapes.
- It also keeps bf16 dequantized MLP projections and uses them only when the
  effective linear row count `M = product(x.shape[:-1])` is at least
  `--hybrid-mlp-threshold`.
- Default threshold is 256. A single isolated MLP starts to favor bf16 around
  M=128, but current whole-path cached-chat suffix tests showed 128-255 token
  suffixes are faster when they stay on q4.
- Smoke validation installed hybrid MLP on all 64 target layers and passed the full test suite.

Why this is better than global `--dequantize-mlp`:

- global dequantization can make decode-like small `M` slower
- hybrid keeps the fast q4 small-M path
- hybrid still unlocks bf16 large-M prefill experiments
- cost is memory: for this Qwen target, the smoke run installed 64 hybrid MLP layers and reported about 34.2 GB of bf16 MLP weights

Early 4096-token smoke numbers:

- hybrid MLP: 18.36 s
- immediate q4 control: 23.04 s
- hybrid repeat: 19.12 s

Controlled 10k A/B:

- q4 first: q4 51.10 s, hybrid 47.51 s
- hybrid first: hybrid 48.18 s, q4 53.32 s
- result: hybrid MLP repeatedly lands in the 47.5-48.2 s band, while q4 lands in the 51.1-53.3 s band
- practical speedup: about 7-10% on this 10k synthetic prefill path

Fresh repeat on 2026-04-27:

- q4 trials: 48.98 s, 51.20 s
- hybrid MLP trials: 45.02 s, 46.54 s
- median speedup: 1.094x, or about 8.6% less prefill time
- aggregate: `benchmark/results/prefill-ab-10000-current/aggregate-1777232829.json`

After enabling middle-chunk no-logits by default:

- q4 trials: 48.42 s, 51.33 s
- hybrid MLP trials: 44.05 s, 45.60 s
- median speedup: 1.113x, or about 10.1% less prefill time
- aggregate: `benchmark/results/prefill-ab-10000-current-default-middle/aggregate-1777234959.json`

After also enabling sliced hidden-state capture for fastpath draft context:

- q4 trials: 49.98 s, 52.28 s
- hybrid MLP trials: 45.68 s, 47.17 s
- median speedup: 1.101x, or about 9.2% less prefill time
- aggregate: `benchmark/results/prefill-ab-10000-current-default-middle-slice/aggregate-1777236305.json`

Server-default mode (`auto_b16`, no explicit fastpath env):

- q4 trials: 49.90 s, 51.36 s
- hybrid MLP trials: 45.69 s, 45.77 s
- median speedup: 1.107x, or about 9.7% less prefill time
- aggregate: `benchmark/results/prefill-ab-10000-auto-current/aggregate-1777236905.json`

Longer 20k server-default spot check:

- q4: 104.1 s
- hybrid MLP: 91.1 s
- speedup: 1.142x, or about 12.5% less prefill time
- aggregate: `benchmark/results/prefill-ab-20000-auto-current/aggregate-1777237549.json`

Memory:

- q4 peak after run: about 20.7 GB
- hybrid MLP peak after run: about 55.0 GB
- extra bf16 MLP weights: about 34.2 GB

Layer-component proof:

- `benchmark/profile_qwen_prefill_layers.py`
- `benchmark/results/layer-profile-q4-2048.json`
- `benchmark/results/layer-profile-hybrid-2048.json`
- `benchmark/results/layer-profile-hybrid-2048-kind.json`

At `M=2048`, synchronized component timing changed:

| Component | q4 | hybrid MLP |
| --- | ---: | ---: |
| attention/GDN | 3496.7 ms | 3541.5 ms |
| MLP | 6857.6 ms | 5686.2 ms |
| total synchronized layer time | 10529.9 ms | 9409.2 ms |

This explains why the full 10k improvement is real but not 30%: MLP improves, but attention/GDN remains a large part of the remaining work.

Per-kind attribution after adding `kind_component_summary_us` and rerunning on the current hybrid path:

- linear/GDN attention: 2.53 s across 48 layers
- full attention: 0.86 s across 16 layers
- linear-layer MLP: 4.08 s across 48 layers
- full-attention-layer MLP: 1.36 s across 16 layers

This says the remaining attention target is mostly the 48 linear/GDN layers by count, not the 16 full-attention layers.

Fresh 2048-token component A/B after the effective-row cleanup:

- q4 core layer work: 10.18 s
- current qkv+z+out stack core layer work: 8.58 s
- layer-loop speedup: 1.19x
- MLP: 6.63 s -> 5.45 s, speedup 1.22x
- linear/GDN attention: 2.52 s -> 2.09 s, speedup 1.21x
- full attention: 0.86 s -> 0.88 s, effectively flat
- results:
  `benchmark/results/layer-profile-q4-control-2048-20260427.json`,
  `benchmark/results/layer-profile-current-qkv-z-out-2048-20260427.json`

Quality check:

- `benchmark/compare_hybrid_logits.py`
- 512-token check: max logit diff 0.0, same argmax, top-10 overlap 10/10
- 2048-token check: max logit diff 0.0, same argmax, top-10 overlap 10/10

MLP dtype check:

- `benchmark/probe_apple_prefill.py`
- result: `benchmark/results/apple-prefill-mlp-fp16-scaling-20260427.json`
- At 2048 tokens through one real MLP:
  - q4: 97.49 ms
  - bf16 dequantized: 84.72 ms
  - fp16 dequantized: 102.57 ms
- Decision: reject fp16 MLP weights. On this Apple GPU / MLX path, bf16 is
  clearly faster than fp16 for the large-M MLP shape.

MLP threshold refinement:

- result: `benchmark/results/apple-prefill-mlp-threshold-refine-20260427.json`
- one real MLP best times:
  - M=96: q4 4.92 ms, bf16 5.48 ms, q4 wins
  - M=128: q4 6.43 ms, bf16 6.04 ms, bf16 wins
  - M=192: q4 9.44 ms, bf16 9.00 ms, bf16 wins
  - M=256: q4 12.42 ms, bf16 11.26 ms, bf16 wins
- earlier whole-model 192-token suffix prefill:
  - threshold 256: 927.2 ms
  - threshold 128: 913.7 ms
- 128-token real-model logit check: max diff 0.0, same argmax, top-10 overlap 10/10.
- current cached-chat-size retest with the max GDN qkv/z/out stack:
  - 96 tokens: threshold 64 was slower than 128, 662.3 ms vs 544.6 ms
  - 130 tokens: threshold 128 was slower than 256, 972.6-974.1 ms vs 817.0-818.7 ms
  - 130-token isolation: MLP128/GDN256 936.9 ms, MLP256/GDN128 847.3 ms,
    MLP256/GDN256 817.0-818.7 ms
  - 192 tokens: threshold 128 was slower than 256, 998.2 ms vs 973.5 ms
  - 452 tokens: threshold 256 was faster than 512, 2161.1 ms vs 2264.1 ms
- Decision: keep the default hybrid MLP/GDN-linear threshold at 256. Long
  2048-token chunks still use bf16, while 128-255 token cached-chat suffixes
  avoid a slower whole-path crossover.

Cache-fraction check:

- 4k hybrid: cache fraction 0.125 -> 18.08 s, 0.25 -> 18.15 s, 0.5 -> 17.70 s
- 10k hybrid: 0.5 -> 46.08 s, immediate 0.25 control -> 46.01 s

Decision: do not change the default MLX cache fraction from 0.25 based on this. The 10k control says the apparent 0.5 win was run-state, not the setting.

Hybrid chunk-size check:

- forward order: 1024 -> 44.26 s, 2048 -> 46.06 s, 4096 -> 46.31 s
- reverse order: 4096 -> 44.94 s, 2048 -> 44.38 s, 1024 -> 46.24 s

Decision: keep 2048 as the measured default. The order flip shows chunk size is still dominated by run-state variance at 10k.

20k confirmation:

- `prefill-step-size 2048`, current best: 90.49 s
- `prefill-step-size 4096`, same current-best settings: 94.73 s
- result: 4096 is worse for this M3 Max run, so keep 2048.
- result file: `benchmark/results/hybrid-gdn-out-step4096-auto-20000-20260427.json`

Current z+out step-size sweep:

- forward 10k: 1024 -> 42.37 s, 2048 -> 42.40 s, 4096 -> 43.03 s
- reverse 10k: 4096 -> 42.96 s, 2048 -> 42.41 s, 1024 -> 42.30 s
- result: 1024 and 2048 are effectively tied; keep 2048 as the user-facing default.
- result files:
  - `benchmark/results/z-out-step-sweep-10000-forward-20260427.json`
  - `benchmark/results/z-out-step-sweep-10000-reverse-20260427.json`

Current default-threshold step-size check:

- 10k max stack after lowering MLP/GDN-linear thresholds to 128:
  - `prefill-step-size 1024`: 41.85 s
  - `prefill-step-size 2048`: 41.85 s
- Decision: still keep 2048. The smaller chunk does not buy measurable first-prefill
  time, and 2048 gives fewer Python loop iterations and fewer checkpoint boundaries.
- result file: `benchmark/results/max-stack-step-sweep-default-thresholds-10000-20260427.json`

Large-step check:

- 10k current max stack in one loaded-model sweep:
  - `prefill-step-size 2048`: 43.42 s
  - `prefill-step-size 4096`: 46.09 s
  - `prefill-step-size 8192`: 45.34 s
  - `prefill-step-size 10000`: 44.97 s
- Decision: keep 2048. Bigger chunks do not win even on the 128 GB M3 Max;
  the larger full-attention working set appears to cost more than the smaller
  Python/chunk loop saves.
- result file: `benchmark/results/max-stack-large-step-sweep-10000-20260427.json`

Middle-chunk no-logits check:

- Before default change: hybrid median 44.55 s, hybrid with middle-no-logits 43.97 s
- speedup: 1.013x, or about 1.3% less prefill time
- aggregate: `benchmark/results/prefill-ab-10000-hybrid-middle/aggregate-1777234723.json`

Decision: enable `DFLASH_PREFILL_MIDDLE_NO_LOGITS` by default. For chunks that
do not need retained draft-context features, this uses
`target_prefill_without_logits()` instead of forcing a hidden-state capture
container. It still evaluates the final hidden state, so cache mutation and
later chunks are preserved.

Sliced hidden-state capture check:

- Old full-chunk capture: 48.22 s, then 45.46 s
- New sliced capture: 46.84 s, then 45.24 s
- median speedup: about 1.017x

Decision: enable `DFLASH_PREFILL_SLICE_CAPTURE` by default. In fastpath prefill,
the draft only needs the sink tokens and trailing window, so selected layer
hidden states are sliced before storing/evaluating instead of retaining the
whole 2048-token chunk.

Deferred sliced capture and cache-only final-MLP skip:

- old hybrid cache path with both disabled: 43.63 s median
- new hybrid default path: 43.39 s median
- speedup: 1.005x, or about 0.5% less prefill time
- aggregate: `benchmark/results/prefill-ab-10000-cacheopt-20260427/aggregate-1777239164.json`

Decision: keep enabled by default. This is small, but exact and low risk:
deferred capture now slices the retained context feature range the same way the
fastpath does, and chunks that only update caches skip the last layer MLP
because no logits, hidden capture, or later layer consumes that output.

Cache-only final-attention K/V update:

- cache-only middle chunks discard the final layer hidden output
- Qwen3.5 hybrid's final layer is full attention, so only that layer's K/V cache
  is needed by future chunks/decode
- the shortcut computes final-layer K/V, updates the cache, and skips final Q,
  SDPA, output projection, and MLP
- real-model next-token cache check: max logit diff 0.0, same argmax, top-10 overlap 10/10
- 10k A/B: current best 45.64 s, shortcut 44.71 s, speedup 1.021x
- 20k A/B: current best 90.61 s, shortcut 89.23 s, speedup 1.015x
- result files:
  - `benchmark/results/cache-only-final-attn-logits-512-20260427.json`
  - `benchmark/results/prefill-ab-10000-skip-final-attn-20260427/aggregate-1777241693.json`
  - `benchmark/results/prefill-ab-20000-skip-final-attn-20260427/aggregate-1777241988.json`

Decision: enable `DFLASH_PREFILL_SKIP_FINAL_ATTENTION` by default. This is a
small hardware-aware win on the 40-core M3 Max, and it only applies when the
final hidden output is known to be discarded.

Selective GDN projection bf16:

- GDN component profile showed the recurrent kernel is not the main GDN cost.
  The large q4 projections dominate: `in_proj_qkv`, `in_proj_z`, and `out_proj`.
- Dequantizing all GDN projections used about 11.1 GB extra memory and was not
  a stable end-to-end win.
- Dequantizing only `out_proj` uses about 3.0 GB extra memory.
- Dequantizing `in_proj_z,out_proj` uses about 6.0 GB extra memory.
- 512-token full-logit check: max diff 0.0, same argmax, top-10 overlap 10/10.
- forward-order 10k A/B: hybrid median 45.52 s, hybrid + GDN out median 43.53 s, speedup 1.046x.
- reverse-order 10k A/B: hybrid median 44.53 s, hybrid + GDN out median 43.74 s, speedup 1.018x.
- `in_proj_z` alone was worse in a one-pass sweep: 45.16 s versus 43.47 s for out-only.
- `in_proj_z,out_proj` was slightly faster than out-only:
  - one-pass sweep: out-only 43.47 s, z+out 42.61 s
  - reverse repeat: out-only median 42.76 s, z+out median 42.47 s, speedup 1.007x
- z+out 512-token full-logit check: max diff 0.0, same argmax, top-10 overlap 10/10.
- `in_proj_qkv,in_proj_z,out_proj` uses about 11.1 GB extra memory and was
  faster again:
  - one-pass sweep: z+out 42.50 s, qkv+z+out 41.93 s
  - clean reverse repeat: z+out median 42.48 s, qkv+z+out median 41.93 s, speedup 1.013x
- qkv+z+out 512-token full-logit check: max diff 0.0, same argmax, top-10 overlap 10/10.
- GDN threshold refinement:
  - component profile q4 medians: qkv 2085 us, z 1408 us, out 1501 us
  - component profile bf16 qkv+z+out medians: qkv 1426 us, z 971 us, out 958 us
  - whole-model 192-token suffix with MLP threshold 128 and GDN threshold 256: 913.7 ms
  - whole-model 192-token suffix with MLP threshold 128 and GDN threshold 128: 908.9 ms
  - 128-token full-logit check: max diff 0.0, same argmax, top-10 overlap 10/10.
  - current 130-token isolation showed GDN128 was slower than GDN256 when MLP
    stayed at 256: 847.3 ms vs 817.0-818.7 ms.
  - Decision: default GDN-linear threshold is 256. Long 2048-token chunks
    still use bf16 either way, and 128-255 token suffixes stay on the safer q4 path.
- q4 vs current best 10k A/B before final-attention K/V shortcut: q4 median 49.28 s, hybrid MLP + GDN out median 43.63 s, speedup 1.129x.
- q4 vs current best 20k before final-attention K/V shortcut: q4 105.28 s, hybrid MLP + GDN out 90.49 s, speedup 1.164x.
- clean-stack 20k A/B after cleanup and async disk-write support:
  q4 105.58 s, hybrid MLP + GDN qkv/z/out 87.44 s, speedup 1.207x.
- with the final-attention K/V shortcut, expected current-best 10k is around 44.7 s in the latest A/B run and 20k is around 89.2 s in the latest spot check.
- fresh q4 vs recommended-stack 10k A/B after enabling the shortcut by default:
  q4 median 49.03 s, out-only recommended median 44.15 s, speedup 1.110x.
- fresh q4 vs max-prefill z+out recommended-stack 10k A/B:
  q4 median 49.51 s, z+out recommended median 44.00 s, speedup 1.125x.
- fresh q4 vs max-prefill qkv+z+out recommended-stack 10k A/B:
  q4 median 47.74 s, qkv+z+out recommended median 42.40 s, speedup 1.126x.
- fresh q4 vs max-prefill qkv+z+out recommended-stack 20k spot check:
  q4 98.17 s, qkv+z+out recommended 87.41 s, speedup 1.123x.
- fresh q4 vs max-prefill qkv+z+out recommended-stack 30k spot check:
  q4 156.58 s, qkv+z+out recommended 136.25 s, speedup 1.149x, saving about 20.3 s.
- real Pi/agent JSONL 30k spot check:
  q4 156.56 s, qkv+z+out recommended 136.22 s, speedup 1.149x, saving about 20.3 s.
- fresh 10k A/B with MLP/GDN-linear thresholds at 128:
  q4 50.78 s, qkv+z+out recommended 43.27 s, speedup 1.174x.
- fresh real Pi-shaped 10k A/B after the checkpoint/server cleanups:
  q4 49.92 s, qkv+z+out recommended 44.73 s, speedup 1.116x.
- fresh 10k A/B after direct K/V dependency evaluation and rollback-kernel
  correctness fixes:
  q4 49.35 s, qkv+z+out recommended 44.09 s, speedup 1.119x.
  aggregate: `benchmark/results/prefill-ab-10000-after-cache-deps-20260427/aggregate-1777257666.json`
- fresh 10k A/B after changing default MLP/GDN-linear thresholds back to 256:
  q4 49.97 s, qkv+z+out recommended 44.52 s, speedup 1.122x.
  aggregate: `benchmark/results/prefill-ab-10000-threshold256-default-20260427/aggregate-1777258379.json`
- fresh 10k A/B after unifying hybrid switch logic around effective linear row
  count `M = product(x.shape[:-1])`:
  q4 49.67 s, qkv+z+out recommended 44.11 s, speedup 1.126x.
  aggregate: `benchmark/results/prefill-ab-10000-row-threshold-cleanup-20260427/aggregate-1777259127.json`
- fresh 20k A/B after the same cleanup:
  q4 105.50 s, qkv+z+out recommended 87.48 s, speedup 1.206x, saving about
  18.0 s.
  aggregate: `benchmark/results/prefill-ab-20000-current-after-cleanups-20260427/aggregate-1777259935.json`
- fresh 30k A/B after the same cleanup:
  q4 153.00 s, qkv+z+out recommended 136.28 s, speedup 1.123x, saving about
  16.7 s.
  aggregate: `benchmark/results/prefill-ab-30000-current-after-cleanups-20260427/aggregate-1777260168.json`
- fresh 10k A/B after async disk-copy cleanup:
  q4 49.60 s, qkv+z+out recommended 44.63 s, speedup 1.111x.
  aggregate: `benchmark/results/prefill-ab-10000-after-disk-copy-cleanup-20260427/aggregate-1777261182.json`
- cache-only final-attention K/V shortcut dependency cleanup:
  the hot path now evaluates the K/V dependency arrays directly instead of
  forcing them through a zero-valued reduction over the new cache slices. This
  is a small overhead cleanup, not a new claimed headline speedup.
  Correctness after the change stayed exact: next-token logit max diff 0.0,
  captured hidden max diff 0.0, same argmax, top-10 overlap 10/10.
  result: `benchmark/results/cache-only-final-attn-logits-threshold256-default-512-20260427.json`
- cache-only eval-root cleanup:
  when direct final-attention K/V dependencies are available, the runtime now
  evaluates those arrays without also making the discarded final hidden tensor
  an eval root. Correctness stayed exact, but one 10k current-stack run measured
  44.62 s, so this is not counted as a confirmed speedup.
  results:
  `benchmark/results/cache-only-final-attn-logits-eval-deps-only-512-20260427.json`,
  `benchmark/results/current-eval-deps-only-prefill-10000-20260427.json`
- checkpoint/capture path final-layer skip check:
  - max-stack 10k cache-only build with 2048-token checkpoints and final-layer skips enabled: 42.26 s, repeat 41.96 s
  - same shape with `--no-skip-final-mlp --no-skip-final-attention`: 43.44 s
  - result: about 2.7-3.4% faster on checkpoint-heavy prefill where every chunk captures hidden states
  - real-model 512-token correctness check: next-token logit max diff 0.0, captured hidden max diff 0.0, same argmax, top-10 overlap 10/10
- aggregates:
  - `benchmark/results/prefill-ab-10000-selective-gdn-20260427/aggregate-1777239777.json`
  - `benchmark/results/prefill-ab-10000-selective-gdn-out-reverse-20260427/aggregate-1777240090.json`
  - `benchmark/results/prefill-ab-10000-q4-vs-hybrid-gdn-out-20260427/aggregate-1777240590.json`
  - `benchmark/results/prefill-ab-20000-q4-vs-hybrid-gdn-out-20260427/aggregate-1777240895.json`
  - `benchmark/results/prefill-ab-10000-current-final-attn-20260427/aggregate-1777242272.json`
  - `benchmark/results/prefill-ab-10000-gdn-z-20260427/aggregate-1777242553.json`
  - `benchmark/results/prefill-ab-10000-gdn-z-out-reverse-20260427/aggregate-1777242711.json`
  - `benchmark/results/hybrid-gdn-z-out-logits-512-20260427.json`
  - `benchmark/results/prefill-ab-10000-current-z-out-20260427/aggregate-1777243032.json`
  - `benchmark/results/prefill-ab-10000-gdn-qkv-z-out-20260427/aggregate-1777243597.json`
  - `benchmark/results/prefill-ab-10000-gdn-qkv-z-out-reverse-clean-20260427/aggregate-1777243872.json`
  - `benchmark/results/hybrid-gdn-qkv-z-out-logits-512-20260427.json`
  - `benchmark/results/gdn-component-profile-128-q4-20260427.json`
  - `benchmark/results/gdn-component-profile-128-qkv-z-out-20260427.json`
  - `benchmark/results/max-stack-prefill-192-mlp128-gdn128-20260427.json`
  - `benchmark/results/hybrid-gdn-qkv-z-out-threshold128-logits-128-20260427.json`
  - `benchmark/results/prefill-ab-10000-current-qkv-z-out-20260427/aggregate-1777244131.json`
  - `benchmark/results/prefill-ab-20000-current-qkv-z-out-20260427/aggregate-1777244369.json`
  - `benchmark/results/prefill-ab-30000-current-qkv-z-out-20260427/aggregate-1777244615.json`
  - `benchmark/results/prefill-ab-30000-realpi-qkv-z-out-20260427/aggregate-1777244998.json`
  - `benchmark/results/prefill-ab-10000-current-default-thresholds-20260427/aggregate-1777247833.json`
  - `benchmark/results/prefill-ab-10000-realpi-current-rerun-20260427/aggregate-1777256109.json`
  - `benchmark/results/max-stack-10000-cacheonly-checkpoint2048-capture-final-skip-20260427.json`
  - `benchmark/results/max-stack-10000-cacheonly-checkpoint2048-capture-final-skip-repeat-20260427.json`
  - `benchmark/results/max-stack-10000-cacheonly-checkpoint2048-no-final-skip-20260427.json`
  - `benchmark/results/cache-only-and-capture-final-skip-logits-512-20260427.json`

Decision: expose the lighter `--hybrid-gdn-out-proj` and the max-prefill
`--hybrid-gdn-z-out-proj` / `--hybrid-gdn-qkv-z-out-proj` variants. Do not hide
them inside `--hybrid-mlp`. They are exact and useful on this M3 Max 128 GB
machine, but the gain is smaller and more run-state-sensitive than hybrid MLP.

Run with:

```bash
dflash-serve \
  --model ~/models/Huihui-Qwen3.6-27B-abliterated-4.5bit-msq \
  --draft ~/models/Qwen3.6-27B-DFlash \
  --port 8000 \
  --prefill-step-size 2048 \
  --block-tokens 16 \
  --prefill-cache-fastpath \
  --hybrid-mlp \
  --hybrid-gdn-qkv-z-out-proj \
  --dflash-prompt-cache \
  --prompt-cache-size 16 \
  --dflash-prompt-cache-dir ~/.cache/dflash-mlx/prompt-cache \
  --dflash-prompt-cache-max-disk-gb 20 \
  --dflash-prompt-cache-checkpoint-tokens 2048 \
  --chat-template-args '{"enable_thinking": false}'
```

### Prompt Cache And Checkpoints

This helps repeated chat sessions, not the first ever computation of a new 30k-token prompt.

Relevant upstream context:

- mlx-lm issue #1162 reports Qwen3-Next hybrid prompt cache misses because
  Gated-DeltaNet recurrent state is not handled like ordinary attention K/V:
  `https://github.com/ml-explore/mlx-lm/issues/1162`
- Our DFlash prompt cache stores the target cache layout plus draft context
  state, so the fix here is model-family-aware rather than assuming every layer
  is pure K/V attention.

Confirmed behavior:

- First request computes the stable prefix and can save reusable cache state.
- Later requests with the same prefix only prefill the suffix.
- Checkpoints inside long prefills let future divergent chats reuse partial prefixes instead of only the final full prompt.
- `benchmark/extract_pi_session_prompt.py` turns a real Pi session JSONL into a
  text-only benchmark prompt without printing private chat contents. It keeps
  text/tool-call content and replaces image blobs with compact placeholders.

Measured examples:

- Full 4096 cache build: about 19.3 s
- Continuing remaining 2048 from a 2048 checkpoint: about 10.25 s
- Full 10k cache build: about 51.7 s
- Reuse from an 8192 checkpoint to 10k: about 9.9 s
- Max-stack cache-only 30k with 4096-token checkpoints: 137.12 s and 7 checkpoints
- This is close to the 136.25 s no-checkpoint max-stack 30k generation-prefill spot check, so 4096-token checkpoints are cheap enough to keep for real chat branching.
- result: `benchmark/results/max-stack-30000-cacheonly-checkpoint4096-20260427.json`
- Max-stack cache-only 30k with 2048-token checkpoints: 135.68 s and 14 checkpoints
- The 2048 result was not slower, so denser checkpoints are cheap on this M3 Max. The tradeoff is cache count/storage, not first-prefill time.
- result: `benchmark/results/max-stack-30000-cacheonly-checkpoint2048-20260427.json`
- After extending the final-layer skip to no-logits hidden-capture chunks, a 10k max-stack cache build with 2048-token checkpoints measured 42.26 s and 41.96 s. Disabling final-layer skips for the same shape measured 43.44 s. This matters for checkpoint-heavy first-prefill because checkpoint chunks need draft hidden captures but not final logits.
- correctness result: `benchmark/results/cache-only-and-capture-final-skip-logits-512-20260427.json`
- Target-only checkpoints were tested as a way to save exact target KV/GDN state
  without materializing persisted draft context at every checkpoint:
  - after fixing no-logits checkpoint emission, a real Pi-shaped 10k run emitted
    all 4 checkpoints
  - target-only checkpoints: 43.52 s
  - full-draft checkpoints rerun in the same run state: 43.49 s
  - result: no first-prefill speedup, and mid-checkpoint reuse would start with
    empty draft context, so this is not part of the recommended path.
  - result files:
    - `benchmark/results/pi-real-cacheonly-profile-10000-target-only-checkpoints-fixed-20260427.json`
    - `benchmark/results/pi-real-cacheonly-profile-10000-full-draft-checkpoints-rerun-20260427.json`
- Server checkpoint handling now stores checkpoint events from the main prefill
  stream too, not just from the separate stable-prefix build. This does not make
  the current request's math faster, but it prevents long active/tool prompts
  from losing reusable partial-prefix checkpoints.
- Current recommended cache-building path uses `--prefill-cache-fastpath`.
  On the max qkv/z/out stack, it measured faster than deferred draft-context
  materialization:
  - 4096 tokens: fastpath 16.81 s, deferred 17.12 s
  - 10k tokens: fastpath 43.70 s, deferred 45.72 s
  - result files:
    - `benchmark/results/fastpath-vs-defer-current-4096-20260427.json`
    - `benchmark/results/fastpath-vs-defer-current-10000-20260427.json`
- Fastpath now supports full prompt-cache checkpoints instead of accidentally
  disabling them. The checkpoint path materializes only the projected
  sink/window draft-context segments that are not already present in the draft
  cache, so it preserves reuse while avoiding duplicate draft-cache fills.
  - 4096 tokens with 2048-token checkpoints: fastpath 16.77 s, deferred 17.57 s
  - both emitted 1 checkpoint
  - regression coverage: `tests/test_dflash_prompt_cache.py`
  - result: `benchmark/results/fastpath-checkpoints-4096-20260427.json`
- Cache-only checkpoint A/B with the updated benchmark harness:
  - 4096 tokens, q4 19.38 s, current qkv/z/out stack 18.27 s, speedup 1.061x
  - 10k forward order: q4 54.01 s, current 49.44 s, speedup 1.093x
  - 10k reverse order: current 54.48 s, q4 61.02 s, speedup 1.120x
  - combined 10k medians: q4 about 57.5 s, current about 52.0 s, speedup about 1.11x
  - result files:
    - `benchmark/results/prefill-ab-4096-cacheonly-checkpoint-current-20260427/aggregate-1777262839.json`
    - `benchmark/results/prefill-ab-10000-cacheonly-checkpoint-current-20260427/aggregate-1777262898.json`
    - `benchmark/results/prefill-ab-10000-cacheonly-checkpoint-current-reverse-20260427/aggregate-1777263026.json`
- Real Pi-shaped prompt extracted from
  `~/.pi/agent/sessions/--Users-zereraz--/2026-04-23T12-52-28-498Z_019dba66-20d2-7186-920c-d58e50653283.jsonl`:
  - source log shape: 1494 messages, 680 tool results, 79 user messages
  - extracted prompt: 1.03M chars, 352k tokens before the benchmark truncates to 10k
  - 10k cache-only checkpoints, forward order: q4 50.89 s, current 47.47 s, speedup 1.072x
  - 10k cache-only checkpoints, reverse order: current 45.24 s, q4 56.65 s, speedup 1.252x
  - combined 10k medians: q4 about 53.77 s, current about 46.35 s, speedup about 1.16x
  - after building a 10k cache, a real-Pi-shaped 1106-token suffix measured q4 6.08 s and current 5.29 s, speedup about 1.15x
  - result files:
    - `benchmark/results/prefill-ab-10000-realpi-cacheonly-checkpoint-20260427/aggregate-1777263428.json`
    - `benchmark/results/prefill-ab-10000-realpi-cacheonly-checkpoint-reverse-20260427/aggregate-1777263549.json`
    - `benchmark/results/pi-real-reuse1106-current-20260427.json`
    - `benchmark/results/pi-real-reuse1106-q4-20260427.json`
- Second real Pi-shaped prompt extracted from
  `~/.pi/agent/sessions/--Users-zereraz--/2026-04-22T12-52-50-896Z_019db540-1c50-73f1-b55c-19b8021a8308.jsonl`:
  - source log shape: 270 messages, 120 tool results, 16 user messages
  - extracted prompt: 183k chars
  - 10k cache-only checkpoints, forward order: q4 53.19 s, current 52.64 s, speedup 1.010x
  - 10k cache-only checkpoints, reverse order: current 61.53 s, q4 73.54 s, speedup 1.195x
  - combined 10k medians: q4 about 63.36 s, current about 57.09 s, speedup about 1.11x
  - result files:
    - `benchmark/results/prefill-ab-10000-realpi2-cacheonly-checkpoint-20260427/aggregate-1777268515.json`
    - `benchmark/results/prefill-ab-10000-realpi2-cacheonly-checkpoint-reverse-20260427/aggregate-1777268682.json`
- Target-only checkpoints were retested on the same real Pi-shaped prompt:
  - q4 55.37 s, current 48.34 s, speedup 1.145x versus q4
  - current target-only 48.34 s did not beat current full-checkpoint runs
    at 45.24-47.47 s
  - decision: do not recommend target-only checkpoints as a default; keeping
    full draft-context checkpoints gives better decode startup after a hit and
    was not slower in the current evidence.
  - result: `benchmark/results/prefill-ab-10000-realpi-target-only-checkpoint-20260427/aggregate-1777263687.json`

Recommended server mode for real chat use:

```bash
dflash-serve \
  --model ~/models/Huihui-Qwen3.6-27B-abliterated-4.5bit-msq \
  --draft ~/models/Qwen3.6-27B-DFlash \
  --port 8000 \
  --prefill-step-size 2048 \
  --block-tokens 16 \
  --prefill-cache-fastpath \
  --hybrid-mlp \
  --hybrid-gdn-qkv-z-out-proj \
  --dflash-prompt-cache \
  --prompt-cache-size 16 \
  --dflash-prompt-cache-dir ~/.cache/dflash-mlx/prompt-cache \
  --dflash-prompt-cache-max-disk-gb 20 \
  --dflash-prompt-cache-checkpoint-tokens 2048 \
  --chat-template-args '{"enable_thinking": false}'
```

### Global MLP Dequantization

This was the earlier exact-output, memory-for-speed experiment. It is now superseded by `--hybrid-mlp`.

Why it can help:

- q4 weights save memory, but each MLP matmul pays dequantization/scales overhead.
- On a 128 GB M3 Max, storing MLP weights in bf16 can be feasible.
- For large prefill chunks, bf16 MLP matmul can be faster even though it moves more raw weight bytes.

Closest measured A/B:

- q4/default 10k prefill: about 51.12 s
- dequantized MLP 10k prefill: about 48.19 s
- speedup: about 5.7%

Cleanup decision: remove the public `--dequantize-mlp` flag. Keep only the internal env path for low-level experiments; normal users should use `--hybrid-mlp` so decode keeps q4.

### Rejected: Hybrid Attention Projections

We tried applying the same shape-aware bf16 idea to large projection linears inside full attention and GDN.

Performance signal:

- `M=2048` synchronized profile improved attention/GDN from 3541.5 ms to 3348.2 ms
- 10k prefill sometimes improved from 47.82 s to 45.78 s
- reverse order was noisy: 50.12 s vs 50.54 s

Correctness/quality signal:

- q4 vs hybrid-attention logits had max absolute diff 10.375
- top-10 overlap was only 4/10, although argmax stayed the same in that one prompt

Decision: remove the code path. The speed signal is too small/noisy for that much logit drift. Hybrid MLP remains because its logit comparison was exact on the 512-token check.

### Mixed: GDN Projection Experiments

The newer MLX Qwen3-Next code shape packs GDN `qkv+z` projections. We tested a
hybrid version that uses packed `qkv+z` only for large prefill chunks.

Correctness:

- 512-token full-logit check was exact after the packed path actually installed
  on all 48 GDN layers.
- result: max logit diff 0.0, top-10 overlap 10/10

Performance:

- hybrid median: 47.85 s
- hybrid + packed GDN median: 47.82 s
- aggregate: `benchmark/results/prefill-ab-10000-hybrid-gdn/aggregate-1777233641.json`

Decision: reject packed GDN. The synchronized profile showed a small linear-GDN
subcomponent improvement, but the end-to-end 10k prefill median was effectively
unchanged.

We also tested all bf16 GDN projection linears for large prefill chunks.

- extra bf16 GDN projection memory: about 11.1 GB
- 512-token full-logit check: max diff 0.0, top-10 overlap 10/10
- hybrid median: 47.41 s
- hybrid + bf16 GDN linears median: 47.72 s
- aggregate: `benchmark/results/prefill-ab-10000-hybrid-gdn-linear/aggregate-1777234049.json`

Decision: reject all-projection bf16 GDN. Exact outputs are not enough; it was
slightly slower and used more memory.

The useful subsets are GDN `out_proj` and, with more memory,
`in_proj_z,out_proj` or `in_proj_qkv,in_proj_z,out_proj`:

- extra bf16 target weights: about 3.0 GB
- 512-token full-logit check: max diff 0.0, top-10 overlap 10/10
- reverse-order 10k A/B vs hybrid: speedup 1.018x
- q4 vs hybrid MLP + GDN out: speedup 1.129x
- `in_proj_z,out_proj` adds about 6.0 GB instead and measured another 1.007x
  over out-only in a reverse-order 10k repeat
- `in_proj_qkv,in_proj_z,out_proj` adds about 11.1 GB instead and measured
  another 1.013x over z+out in a clean reverse-order 10k repeat

Decision: keep `--hybrid-gdn-out-proj` as the lighter opt-in and
`--hybrid-gdn-qkv-z-out-proj` as the max-prefill M3 Max/large-memory tuning.

### Rejected: Packed bf16 MLP Gate/Up

The hybrid MLP large-M path can either run bf16 gate and up as two matmuls or
concatenate those bf16 weights and run one wider gate+up matmul.

Correctness:

- 512-token full-logit check was exact: max diff 0.0, top-10 overlap 10/10

Performance:

- packed off: 45.29 s, then 43.59 s
- packed on: 45.42 s, then 43.46 s
- median result was effectively flat

Decision: keep the packed bf16 MLP gate/up experiment opt-in through
`DFLASH_HYBRID_MLP_PACK_BF16_GATE_UP=1`, but default it off.

### Rejected: Compiled MLP Wrapper

We tested `mx.compile(lambda x: mlp(x))` around one real MLP:

- result: `benchmark/results/apple-prefill-mlp-compile-probe-20260427.json`
- q4 M=2048: 97.05 ms uncompiled, 96.80 ms compiled
- bf16 M=2048: 84.67 ms uncompiled, 85.39 ms compiled
- bf16 M=128: 6.09 ms uncompiled, 6.03 ms compiled

Decision: do not add a runtime compiled-MLP wrapper. The only win was tiny at
M=128, while the long-prefill shape was flat/slower.

### Rejected: Compiled GDN Wrapper

We tested `mx.compile(lambda x: linear_attn(x, None, None))` around one real
Qwen3.5 GDN block. Shape-specific compile was required because shapeless compile
cannot infer the static GDN split outputs.

- q4 GDN, M=2048: uncompiled 48.12 ms best, compiled 48.18 ms best
- q4 GDN, M=2048 median: compiled was only 1.003x, effectively flat
- hybrid qkv/z/out GDN, M=2048: uncompiled 43.36 ms best, compiled 43.27 ms best
- hybrid qkv/z/out GDN, M=2048 median: compiled was slower, 0.989x
- both checks were exact: max diff 0.0
- results:
  `benchmark/results/gdn-compile-q4-512-2048-20260427.json`,
  `benchmark/results/gdn-compile-qkv-z-out-512-2048-20260427.json`

Decision: do not add a compiled GDN runtime wrapper. The current GDN path is
already dominated by projection matmuls and the upstream Metal recurrent
kernel; larger MLX graph compile does not expose a meaningful prefill win.

### Rejected: Target KV-Cache Quantization For Speed

Target KV-cache quantization is different from model weight quantization. It
compresses full-attention K/V activations, so it can affect outputs and decode
scheduling.

- result: `benchmark/results/kv-quality-current-max-2000-with-nosplit-20260427.json`
- 2k current max stack:
  - default split: 8373.6 ms prefill
  - fp no-split control: 8301.8 ms prefill
  - q8 KV: 8366.5 ms prefill
  - q4 KV: 9197.3 ms prefill
  - q2 KV: 9405.4 ms prefill

Decision: reject KV-cache quantization as a prefill speed knob here. q8 is
essentially flat, q4/q2 are slower, and this harness also shows split/no-split
decode drift, so quality needs a separate evaluation before any future KV-cache
change.

### Rejected: Full-Attention Projection Dequantization

We tried extending the large-M hybrid-linear idea from GDN projections to the
16 full-attention layers.

- all full-attention `q_proj,k_proj,v_proj,o_proj` with the current max stack:
  `max_abs_logit_diff=10.375`, same argmax, top-10 overlap 4/10
- `k_proj,v_proj` only:
  `max_abs_logit_diff=10.9375`, same argmax, top-10 overlap 4/10
- `q_proj` only:
  `max_abs_logit_diff=0.0`, same argmax, top-10 overlap 10/10
- `o_proj` only:
  `max_abs_logit_diff=0.0`, same argmax, top-10 overlap 10/10
- `q_proj,o_proj` only:
  `max_abs_logit_diff=0.0`, same argmax, top-10 overlap 10/10
- 10k paired run:
  current max 44.28 s, current max + full-attn `q_proj,o_proj` 45.06 s
- 10k reverse run:
  current max + full-attn `q_proj,o_proj` 41.87 s, current max 41.98 s

Decision: reject this as a user-facing prefill option. K/V are not exact enough
because full-attention softmax amplifies small projection differences; Q/O are
exact but their end-to-end speed is run-state noise, not a confirmed win.

### Rejected: Async Per-Chunk Prefill Eval

We tested using `mx.async_eval` after each prefill chunk and synchronizing at
the prefill boundary instead of blocking on `mx.eval` after each chunk.

- 10k paired run:
  current max 43.27 s, async eval 45.29 s
- 10k reverse run:
  async eval 42.40 s, current max 41.98 s

Decision: reject. The bottleneck is not Python blocking between chunks; keeping
the normal synchronous chunk boundary is simpler and faster on these runs.

### Rejected: Disabling Split SDPA For Prefill

The split-SDPA hook exists for decode/short-query full attention. We tested
whether disabling it during large prefill would let upstream MLX-LM run faster.

- 10k paired run:
  current max 42.36 s, no split SDPA 45.36 s
- 10k reverse run:
  no split SDPA 42.66 s, current max 41.98 s

Decision: keep split SDPA enabled. In the current stack it is not just overhead
for prefill.

### Rejected: Decode-Only Verify Linear For Prefill

We retested deferring verify-linears until decode:

- decode-only verify prefill: 41.97 s
- immediate current-stack control: 41.98 s

Decision: no meaningful prefill change. Keep the current verify-linear setup
unless decode-specific testing says otherwise.

### Rejected: Extra GDN `b/a` Hybrid Linears

We checked whether GDN `in_proj_b` and `in_proj_a` could join the bf16 large-M
hybrid-linear path. They cannot: on this model they are `_ExactSmallProjPad`,
not quantized linears, so the hybrid wrapper has nothing to install. The useful
GDN hybrid set remains `in_proj_qkv,in_proj_z,out_proj`.

Follow-up metadata check:

- allowing `in_proj_b,in_proj_a` in the benchmark attr list still installed
  144 linears and 11.07 GB of bf16 weights, exactly the same as qkv/z/out
- the 4096-token timing differences were run-state noise, not a real b/a path
- result files:
  - `benchmark/results/prefill-ab-4096-gdn-ba-hybrid-20260427/aggregate-1777262460.json`
  - `benchmark/results/prefill-ab-4096-gdn-ba-hybrid-reverse-20260427/aggregate-1777262522.json`

Decision: keep b/a out of the runtime attr allow-list.

### Rejected: Packed bf16 GDN qkv+z

We tried packing the bf16-dequantized GDN `in_proj_qkv` and `in_proj_z` weights
into one wider large-M projection, while keeping `out_proj` as a separate bf16
hybrid linear.

- real-model 512-token logit check:
  `max_abs_logit_diff=0.0`, same argmax, top-10 overlap 10/10
- 10k paired run:
  separate qkv/z/out 44.00 s, packed qkv+z plus out 46.40 s
- 10k reverse run:
  packed qkv+z plus out 43.27 s, separate qkv/z/out 41.98 s
- follow-up one-layer microbench:
  - M=512: separate 7.024 ms best, packed 7.016 ms best, median packed slower
  - M=1024: separate 13.598 ms best, packed 13.603 ms best
  - M=2048: separate 26.865 ms best, packed 26.823 ms best, median flat
  - max diff: 0.0 for all tested sizes
  - result: `benchmark/results/gdn-bf16-qkvz-pack-512-2048-20260427.json`

Decision: reject. Like packed MLP gate/up, one wider projection is not faster
on this MLX/M3 Max path.

### Rejected: Packed GDN b+a

SGLang and vLLM pack small Qwen3.5 GDN `b/a` projections in their CUDA/server
paths, so we tested packing MLX `in_proj_b` and `in_proj_a` into a single
quantized projection.

- first attempt did not install because these projections are wrapped by the
  small-projection padding hook
- after unwrapping and installing on all 48 GDN layers, the 512-token logit
  check failed: `max_abs_logit_diff=12.1875`, top-10 overlap 4/10
- disabling verify-linear did not change the drift, so the issue is the packed
  quantized projection path/order, not the verify wrapper

Decision: reject and remove. This is exactly the kind of server-runtime idea
that looks plausible from SGLang/vLLM but cannot be kept unless the MLX output
is exact.

### Rejected: Explicit MLX Streams For MLP Gate/Up

MLP gate and up projections are independent, so we tested whether launching
them on two explicit MLX streams would overlap better on the M3 Max GPU.

- M=128: sequential best 6.08 ms, streams best 6.17 ms
- M=512: sequential best 21.62 ms, streams best 21.88 ms
- M=2048: sequential best 84.78 ms, streams best 84.98 ms
- output diff: exact

Decision: reject. The normal lazy MLX graph/single-stream scheduling is already
as good or better for this MLP shape; explicit streams add overhead without a
measured prefill win.

### Rejected: GDN Threadgroup-y Change

We tested the upstream MLX gated-delta Metal kernel with the same Qwen-shaped
GDN tensors but different `threadgroup.y` values.

- all variants were exact after using stable synthetic q/k/v ranges
- y=1 best 5.13 ms
- y=2 best 5.10 ms
- y=4 best 5.06 ms
- y=8 best 5.16 ms
- y=16 best 5.09 ms

Decision: keep upstream-style `threadgroup=(32, 4, 1)`. It does not create
duplicate writes; it groups multiple `Dv` rows per threadgroup, and the measured
differences are tiny.

Important distinction: our custom rollback tape/replay kernels are not the same
as upstream MLX-LM's GDN kernel. They index `Dv` with
`thread_position_in_grid.y` but only use `threadgroup.x`, so `threadgroup.y > 1`
does duplicate work there. We changed only those custom rollback kernels to
`threadgroup=(32, 1, 1)` and added regression coverage.

Rollback kernel correctness fixes on 2026-04-27:

- preserve recurrent `state_out` dtype as the input state dtype, matching
  upstream MLX-LM's `StT` path instead of casting fp32 state to bf16 activation
  dtype
- remove per-step state downcasts inside tape/replay, because upstream keeps
  the recurrent update in float and only casts at the output boundary
- write zero `y` for masked tokens instead of leaving the Metal output
  uninitialized
- proof: `tests/test_gdn_kernels.py` covers scalar/vector GDN gates, masked and
  unmasked tape generation, and replay parity against the MLX ops fallback
- validation: `pytest -q` -> 110 passed, 1 skipped; short real generation smoke
  after the fix -> 32 tokens, 12.5 tok/s, 53.1% acceptance
- longer decode smoke after the rollback fix and threshold-256 default:
  128 tokens, 39.1 tok/s, 84.4% acceptance

### Rejected: Final-Layer Last-Token-Only Prefill Logits

We tested computing the last target layer only for the last prompt token when
prefill only needs first-token logits.

- 512-token logit check with current max stack:
  `max_abs_logit_diff` stayed around 0.09-0.13
- next-token cache continuity check:
  exact after advancing from the cache

Decision: reject and remove from runtime. It preserves the cache for later
decode, but it changes the first generated token logits, so it is not an exact
generation path.

### Rejected: Target `ConcatenateKVCache`

We tested `ConcatenateKVCache` for target full-attention layers to avoid the
normal `KVCache` growth/update path during prefill.

- 10k current-stack run: about 41.94 s, effectively tied with nearby control
  runs around 41.98 s
- decode speed in that run did not improve and looked slightly worse

Decision: remove the flag. The prefill signal was noise-level and the cache
choice can affect later single-token decode appends.

### Rejected: Lower-Precision GDN Recurrent State

Qwen3.5's GDN recurrent state is fp32 upstream. We tested whether writing the
state as bf16 at chunk boundaries would reduce cache traffic.

- same-chunk logit checks are not enough, because the Metal kernel accumulates
  the whole chunk in float and only writes the final recurrent state at the end
- chunked 1024-token prefix with 512-token chunks:
  - default fp32 state: 4271.97 ms
  - bf16 state: 4268.39 ms
  - speedup: 1.0008x, effectively flat
  - next-token logit drift: `max_abs_logit_diff=4.65625`
  - argmax stayed the same in this one check, but the drift is too large for an
    exact path.
- result: `benchmark/results/gdn-state-bf16-chunked-1024-step512-20260427.json`

Decision: keep fp32 GDN state. The custom GDN hook now preserves the upstream
fp32 state dtype when it has to initialize state itself, and the rollback tape
kernel preserves the recurrent state dtype when returning `state_out`.

### Neutral: Large-Prefill SDPA Query Tiling

We tested splitting only the full-attention SDPA query dimension while running
larger model chunks. This is exact and is left as a benchmark-only probe, but
it is not a default optimization.

- 10k sweep: 2048-token chunks about 41.98 s, 4096-token chunks about 42.05 s,
  8192-token chunks about 42.08 s, 10000-token chunk about 42.75 s
- 20k sweep: 2048-token chunks about 87.48 s, 4096-token chunks about 87.38 s

Decision: no confirmed first-prefill win. It makes larger chunks less bad, but
does not beat the existing 2048-token scheduling in a stable way.

### Decode Adaptive Fallback

This is a decode improvement, not a prefill improvement.

It watches speculative decode quality. If the draft model is wasting cycles during low-acceptance regions, it temporarily falls back to target-only decode and later probes again.

Observed effect in bad-draft coding/chat regions:

- lower draft acceptance ratio
- higher effective tokens/sec
- fewer wasted speculative cycles

### Async Disk Prompt-Cache Writes

This is not a faster matrix multiply; it removes CPU/disk checkpoint save time
from the request thread.

When using `--dflash-prompt-cache-dir`, checkpoint and final prompt-cache saves
now default to a background writer. The memory cache is still inserted
synchronously, so the current server session can reuse the prefix immediately.

Synthetic 268 MB cache-write probe on the M3 Max:

- synchronous disk insert returned after 434.8 ms
- async disk insert returned after 1.8 ms
- async background wait still took 415.3 ms
- after removing the redundant async disk snapshot copy, the same synthetic
  shape returned in 1.8 ms and the background wait measured 352.8 ms in one
  run; treat this as request-thread cleanup, not a model-compute speedup
- result:
  `benchmark/results/disk-cache-writes-after-copy-cleanup-4x16x8192x128-20260427.json`
- follow-up read-hit probe on the same 268 MB synthetic cache:
  fetch returned in about 1.0-1.3 ms, and forcing the loaded arrays live took
  about 10 ms; persistent cache hits are therefore much cheaper than recomputing
  an 8k+ prefix
- result:
  `benchmark/results/disk-cache-writes-fetch-after-copy-cleanup-4x16x8192x128-20260427.json`

Decision: keep async writes as the server default for disk prompt cache. Use
`--dflash-prompt-cache-sync-writes` only when debugging persistence timing or
when a caller needs disk durability before the response continues.

## What Did Not Prove Out Yet

### Packed MLP Gate+Up

The idea was to combine gate projection and up projection so two MLP matmuls become one larger matmul.

Reality from same-process microbench:

- unpacked MLP at M=2048: about 97.55 ms
- packed MLP at M=2048: about 97.94 ms

So this is not a confirmed first-prefill win. Earlier 15% claims were likely run-state drift, not a stable optimization.

### Tiled MLP SwiGLU+Down Accumulation

The idea was to avoid materializing the full `[B, S, intermediate]` SwiGLU
activation by slicing the intermediate dimension, computing partial
`SwiGLU(gate, up) @ down_weight` products, and accumulating the output.

Reality on one real target MLP:

- M=2048 baseline bf16 MLP: 85.40 ms best
- intermediate tile 2048: 85.97 ms best and `max_abs_diff=1.0`
- intermediate tile 4352: 85.00 ms best and `max_abs_diff=1.0`
- intermediate tile 8704: 85.47 ms best and `max_abs_diff=1.0`
- full-width manual matmul tile 17408: exact, but only 84.69 ms in one short run

Follow-up exact manual-matmul run:

- M=512: baseline 21.647 ms, manual 21.666 ms
- M=1024: baseline 42.622 ms, manual 42.618 ms
- M=2048: baseline 84.702 ms, manual 84.642 ms

Decision: reject for production. Real tiling changes accumulation order enough
to cause visible tensor drift, and the exact full-width manual path is
noise-level versus `nn.Linear`.

### Chunk Size Sweeps

Some runs made smaller chunks look faster, but reverse-order checks removed the signal.

Conclusion:

- chunk size can affect memory and scheduling
- current data does not prove a reliable first-prefill speedup from changing it

### Decode-Only Verify Linear

Moving decode-specific linear wrappers out of prefill looked promising in one short run, but 10k comparisons did not confirm it.

Conclusion:

- keep it experimental
- do not count it as a real prefill win yet

### Startup Warmup / Precompile

We tested whether same-process warmup makes a second 10k prefill materially
faster on the recommended qkv/z/out stack.

- run 1: 43.69 s
- run 2, same loaded models/process: 46.31 s
- result: `benchmark/results/current-same-process-repeat-10000-20260427.json`

Conclusion:

- do not add a startup prefill warmup knob for speed
- the first-prefill wall is mostly real model compute and memory scheduling,
  not one-time MLX compilation overhead

## Why We Cannot Parallelize Everything

Inside one layer, MLX already parallelizes the large matmuls and attention kernels across the Apple GPU.

Across layers, we cannot run layer 12 before layer 11 because each layer consumes the previous layer's output.

Across chunks, later prompt tokens depend on earlier prompt tokens through causal attention and state. We can cache earlier chunks and reuse them later, but for a brand-new prompt the cache/state still has to be built in order.

The useful parallelism is therefore:

- make each layer's kernels faster
- fuse operations to reduce memory traffic
- avoid recomputing prefixes already seen
- use memory-for-speed tradeoffs where the machine can afford them
- improve decode scheduling so the draft model does not fight the target model when acceptance is poor

## Best Next Bets

1. Keep validating the max stack in real Pi Mono sessions and watch memory pressure.
2. Measure live server checkpoint-copy/disk-write overhead for tool-output-heavy chats before moving any writes to background threads.
3. Profile MLP/attention projection matmuls with real Metal capture to see if q4 is dequant-bound, bandwidth-bound, or dispatch-bound on M3 Max.
4. Investigate a custom large-M projection path only if Metal capture shows MLX's quantized matmul leaves obvious GPU utilization on the table.
5. Treat ANE/Core ML as separate research, not a quick flag. It may help only if selected matmuls can be offloaded without data-transfer overhead dominating.

## Current Honest Speedup Summary

- First-prefill exact compute: about 11-21% confirmed from the current max
  stack on 10k-30k A/B/spot checks, depending on prompt length and run state.
  Latest clean runs: 10k 1.126x, 20k 1.206x, 30k 1.123x.
- Medium cached suffixes: default threshold 256 keeps 128-255 token suffixes on
  q4 after current whole-path retests showed threshold 128 was slower there.
- Repeated-prefix prefill: can be much faster when prompt cache/checkpoints hit.
- Decode: adaptive fallback gave around 20% effective improvement in low-acceptance sessions.
- 10x first-prefill on the same exact model and hardware is not currently supported by the data; reaching that would require a much bigger runtime or model-contract change.
