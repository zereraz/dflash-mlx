# Adaptive Decode Controller Notes

Status: deferred. Current focus is prefill.

The next decode idea is a position-aware DFlash controller, not a single
on/off fallback switch.

## Goal

Use live acceptance shape to decide when to keep DFlash, shrink the block,
or briefly run target-only decoding before probing DFlash again.

## Signals

- Isolated miss: one bad draft token after otherwise good cycles. Keep DFlash.
- Good early positions, bad late positions: block is too long. Shrink block.
- Repeated rejection at position 1: draft model is misaligned for the current
  region. Use target-only autoregressive cooldown.
- Recovery streak: grow block again with hysteresis.

## Block Selection

For a candidate block size `B`, estimate expected committed tokens per cycle:

```text
E(tokens for B) = 1 + sum(P(accept position i), i = 1..B-1)
```

Choose the block that maximizes expected tokens per measured cycle cost. Use
hysteresis so the block size does not bounce every few cycles.

## Why This Should Help

Acceptance is not only "good" or "bad". Sometimes the draft gets the first few
tokens right but fails later. In that case disabling DFlash throws away useful
work; a smaller block is better. When position-1 rejection repeats, the draft
is not helping and target-only cooldown avoids wasting GPU time on draft passes.

## Open Questions

- How many cycles are needed before the signal is stable enough?
- Should the controller share history across requests with the same prompt
  cache prefix?
- What is the measured cost curve for block sizes 2, 4, 8, 16 on M3 Max?
