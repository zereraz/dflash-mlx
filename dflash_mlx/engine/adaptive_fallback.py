from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional


_FALSE_VALUES = {"", "0", "false", "no", "off"}


@dataclass(frozen=True)
class AdaptiveFallbackConfig:
    enabled: bool = False
    min_tokens_per_cycle: float = 2.0
    probe_cycles: int = 8
    cooldown_tokens: int = 32
    reprobe_block_tokens: Optional[int] = None


@dataclass
class AdaptiveDecision:
    action: str
    reason: str
    tokens_per_cycle: float
    probe_cycles: int
    probe_tokens: int
    cooldown_tokens: int
    block_tokens: int
    next_block_tokens: int
    fallback_count: int
    reprobe_count: int

    def as_event(self) -> dict[str, object]:
        return {
            "event": "adaptive_fallback",
            "action": self.action,
            "reason": self.reason,
            "tokens_per_cycle": self.tokens_per_cycle,
            "probe_cycles": self.probe_cycles,
            "probe_tokens": self.probe_tokens,
            "cooldown_tokens": self.cooldown_tokens,
            "block_tokens": self.block_tokens,
            "next_block_tokens": self.next_block_tokens,
            "fallback_count": self.fallback_count,
            "reprobe_count": self.reprobe_count,
        }


@dataclass
class AdaptiveFallbackState:
    config: AdaptiveFallbackConfig
    initial_block_tokens: int
    active_block_tokens: int = field(init=False)
    mode: str = field(default="probe", init=False)
    cooldown_remaining: int = field(default=0, init=False)
    pending_reprobe_block_tokens: Optional[int] = field(default=None, init=False)
    probe_cycles_seen: int = field(default=0, init=False)
    probe_tokens: int = field(default=0, init=False)
    last_probe_tokens_per_cycle: float = field(default=0.0, init=False)
    fallback_count: int = field(default=0, init=False)
    reprobe_count: int = field(default=0, init=False)
    fallback_tokens: int = field(default=0, init=False)
    fallback_reason: Optional[str] = field(default=None, init=False)
    decisions: list[AdaptiveDecision] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.initial_block_tokens = max(1, int(self.initial_block_tokens))
        self.active_block_tokens = self.initial_block_tokens

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def in_cooldown(self) -> bool:
        return self.enabled and self.mode == "cooldown"

    def block_tokens_for_cycle(self, remaining_tokens: int) -> int:
        remaining_tokens = max(0, int(remaining_tokens))
        if remaining_tokens <= 0:
            return 0
        if self.in_cooldown:
            return 1
        return max(1, min(self.active_block_tokens, remaining_tokens))

    def record_cycle(
        self,
        *,
        block_len: int,
        commit_count: int,
        can_continue: bool = True,
    ) -> Optional[AdaptiveDecision]:
        if not self.enabled:
            return None

        block_len = int(block_len)
        commit_count = max(0, int(commit_count))

        if self.in_cooldown:
            self.fallback_tokens += commit_count
            self.cooldown_remaining = max(0, self.cooldown_remaining - commit_count)
            if self.cooldown_remaining > 0 or not can_continue:
                return None
            return self._begin_reprobe()

        if block_len <= 1:
            return None

        self.probe_cycles_seen += 1
        self.probe_tokens += commit_count
        if self.probe_cycles_seen < self.config.probe_cycles:
            return None

        tokens_per_cycle = self.probe_tokens / max(1, self.probe_cycles_seen)
        self.last_probe_tokens_per_cycle = tokens_per_cycle
        probe_cycles = self.probe_cycles_seen
        probe_tokens = self.probe_tokens
        self._reset_probe()

        if tokens_per_cycle >= self.config.min_tokens_per_cycle or not can_continue:
            return None

        next_block_tokens = self._next_reprobe_block_tokens()
        self.mode = "cooldown"
        self.cooldown_remaining = max(1, int(self.config.cooldown_tokens))
        self.pending_reprobe_block_tokens = next_block_tokens
        self.fallback_count += 1
        self.fallback_reason = (
            f"tokens_per_cycle={tokens_per_cycle:.3f} "
            f"< {self.config.min_tokens_per_cycle:.3f}"
        )
        decision = AdaptiveDecision(
            action="fallback",
            reason=self.fallback_reason,
            tokens_per_cycle=tokens_per_cycle,
            probe_cycles=probe_cycles,
            probe_tokens=probe_tokens,
            cooldown_tokens=self.cooldown_remaining,
            block_tokens=self.active_block_tokens,
            next_block_tokens=next_block_tokens,
            fallback_count=self.fallback_count,
            reprobe_count=self.reprobe_count,
        )
        self.decisions.append(decision)
        return decision

    def summary_fields(self) -> dict[str, object]:
        return {
            "adaptive_fallback_enabled": bool(self.config.enabled),
            "adaptive_fallback_triggered": self.fallback_count > 0,
            "adaptive_fallback_count": int(self.fallback_count),
            "adaptive_reprobe_count": int(self.reprobe_count),
            "adaptive_fallback_tokens": int(self.fallback_tokens),
            "adaptive_fallback_reason": self.fallback_reason,
            "adaptive_min_tokens_per_cycle": float(self.config.min_tokens_per_cycle),
            "adaptive_probe_window": int(self.config.probe_cycles),
            "adaptive_cooldown_tokens": int(self.config.cooldown_tokens),
            "adaptive_initial_block_tokens": int(self.initial_block_tokens),
            "adaptive_final_block_tokens": int(self.active_block_tokens),
            "adaptive_last_probe_tokens_per_cycle": float(self.last_probe_tokens_per_cycle),
            "adaptive_pending_probe_cycles": int(self.probe_cycles_seen),
            "adaptive_pending_probe_tokens": int(self.probe_tokens),
            "adaptive_events": [
                {k: v for k, v in decision.as_event().items() if k != "event"}
                for decision in self.decisions
            ],
        }

    def _begin_reprobe(self) -> AdaptiveDecision:
        next_block_tokens = max(
            1,
            min(
                int(self.pending_reprobe_block_tokens or self.active_block_tokens),
                self.initial_block_tokens,
            ),
        )
        previous_block_tokens = self.active_block_tokens
        self.active_block_tokens = next_block_tokens
        self.pending_reprobe_block_tokens = None
        self.mode = "probe"
        self.reprobe_count += 1
        self._reset_probe()
        decision = AdaptiveDecision(
            action="reprobe",
            reason="target_ar_cooldown_complete",
            tokens_per_cycle=self.last_probe_tokens_per_cycle,
            probe_cycles=0,
            probe_tokens=0,
            cooldown_tokens=0,
            block_tokens=previous_block_tokens,
            next_block_tokens=self.active_block_tokens,
            fallback_count=self.fallback_count,
            reprobe_count=self.reprobe_count,
        )
        self.decisions.append(decision)
        return decision

    def _reset_probe(self) -> None:
        self.probe_cycles_seen = 0
        self.probe_tokens = 0

    def _next_reprobe_block_tokens(self) -> int:
        configured = self.config.reprobe_block_tokens
        if configured is not None:
            return max(1, min(int(configured), self.initial_block_tokens))
        if self.active_block_tokens <= 2:
            return self.active_block_tokens
        return max(2, self.active_block_tokens // 2)


def resolve_adaptive_fallback_config(
    env: Mapping[str, str],
) -> AdaptiveFallbackConfig:
    enabled_raw = str(env.get("DFLASH_ADAPTIVE_FALLBACK", "")).strip().lower()
    enabled = enabled_raw not in _FALSE_VALUES
    if not enabled:
        return AdaptiveFallbackConfig(enabled=False)
    return AdaptiveFallbackConfig(
        enabled=True,
        min_tokens_per_cycle=_float_env(
            env,
            "DFLASH_ADAPTIVE_MIN_TOKENS_PER_CYCLE",
            2.0,
            min_value=1.0,
        ),
        probe_cycles=_int_env(env, "DFLASH_ADAPTIVE_PROBE_CYCLES", 8, min_value=1),
        cooldown_tokens=_int_env(
            env,
            "DFLASH_ADAPTIVE_TARGET_AR_COOLDOWN",
            32,
            min_value=1,
        ),
        reprobe_block_tokens=_optional_int_env(
            env,
            "DFLASH_ADAPTIVE_REPROBE_BLOCK_TOKENS",
            min_value=1,
        ),
    )


def _int_env(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    min_value: int,
) -> int:
    raw = str(env.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return max(int(min_value), value)


def _optional_int_env(
    env: Mapping[str, str],
    name: str,
    *,
    min_value: int,
) -> Optional[int]:
    raw = str(env.get(name, "")).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(int(min_value), value)


def _float_env(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    min_value: float,
) -> float:
    raw = str(env.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return max(float(min_value), value)
