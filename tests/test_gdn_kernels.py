from __future__ import annotations

import mlx.core as mx
import pytest

from dflash_mlx.kernels import (
    _gated_delta_ops_with_tape,
    _tape_replay_ops,
    gated_delta_kernel_with_tape,
    tape_replay_kernel,
)


pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(),
    reason="custom GDN rollback kernels require Metal",
)


def _small_inputs(*, vectorized_g: bool = False, masked: bool = False):
    mx.random.seed(7)
    batch, steps, h_k, h_v, d_k, d_v = 1, 4, 1, 2, 64, 8
    q = (mx.random.normal((batch, steps, h_k, d_k)) * 0.05).astype(mx.bfloat16)
    k = (mx.random.normal((batch, steps, h_k, d_k)) * 0.05).astype(mx.bfloat16)
    v = (mx.random.normal((batch, steps, h_v, d_v)) * 0.05).astype(mx.bfloat16)
    beta = mx.sigmoid(mx.random.normal((batch, steps, h_v)) * 0.1).astype(
        mx.bfloat16
    )
    if vectorized_g:
        g = (mx.ones((batch, steps, h_v, d_k), dtype=mx.float32) * 0.96) + (
            mx.random.normal((batch, steps, h_v, d_k)) * 0.01
        )
    else:
        g = (mx.ones((batch, steps, h_v), dtype=mx.float32) * 0.96) + (
            mx.random.normal((batch, steps, h_v)) * 0.01
        )
    state = (mx.random.normal((batch, h_v, d_v, d_k)) * 0.05).astype(mx.float32)
    mask = mx.array([[True, False, True, True]]) if masked else None
    return q, k, v, g, beta, state, mask


@pytest.mark.parametrize("vectorized_g", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_gated_delta_tape_kernel_preserves_state_dtype_and_matches_ops(
    vectorized_g: bool,
    masked: bool,
):
    q, k, v, g, beta, state, mask = _small_inputs(
        vectorized_g=vectorized_g,
        masked=masked,
    )

    expected_y, expected_state, expected_tape = _gated_delta_ops_with_tape(
        q, k, v, g, beta, state, mask
    )
    actual_y, actual_state, actual_tape = gated_delta_kernel_with_tape(
        q, k, v, g, beta, state, mask
    )
    mx.eval(expected_y, expected_state, expected_tape, actual_y, actual_state, actual_tape)

    assert actual_state.dtype == state.dtype
    assert actual_tape.dtype == mx.float32
    assert mx.allclose(actual_y, expected_y.astype(actual_y.dtype), atol=1e-2, rtol=1e-2).item()
    assert mx.allclose(actual_state, expected_state, atol=1e-4, rtol=1e-4).item()
    assert mx.allclose(actual_tape, expected_tape, atol=1e-4, rtol=1e-4).item()


@pytest.mark.parametrize("vectorized_g", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_tape_replay_kernel_matches_ops_and_full_tape_state(vectorized_g: bool, masked: bool):
    q, k, v, g, beta, state, mask = _small_inputs(
        vectorized_g=vectorized_g,
        masked=masked,
    )
    _, expected_state, tape = _gated_delta_ops_with_tape(q, k, v, g, beta, state, mask)

    replay_expected = _tape_replay_ops(tape, k, g, state, mask)
    replay_actual = tape_replay_kernel(tape, k, g, state, mask)
    mx.eval(expected_state, replay_expected, replay_actual)

    assert replay_actual.dtype == state.dtype
    assert mx.allclose(replay_actual, replay_expected, atol=1e-4, rtol=1e-4).item()
    assert mx.allclose(replay_actual, expected_state, atol=1e-4, rtol=1e-4).item()
