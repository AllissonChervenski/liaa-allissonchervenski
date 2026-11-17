"""Unit tests for Causal Temporal Majority Voting and Post-Processing Filter."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.temporal_smoothing import (
    CausalTemporalMajorityVote,
    causal_majority_vote,
    causal_soft_voting,
    compute_smoothing_metrics,
    stream_causal_majority_vote,
)


def test_invalid_parameters() -> None:
    """Ensure invalid window_size or tie_breaker raises ValueError."""
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        CausalTemporalMajorityVote(window_size=0)

    with pytest.raises(ValueError, match="Unknown tie_breaker"):
        CausalTemporalMajorityVote(window_size=3, tie_breaker="invalid_strategy")


def test_edge_cases_empty_and_k1() -> None:
    """Empty arrays and window_size=1 should return exact copies."""
    assert len(causal_majority_vote([], window_size=3)) == 0
    assert len(stream_causal_majority_vote([], window_size=3)) == 0

    preds = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    res_k1 = causal_majority_vote(preds, window_size=1)
    np.testing.assert_array_equal(res_k1, preds)


def test_glitch_filtering() -> None:
    """A single transient spike inside a stable plateau should be filtered out."""
    # Sequence of class 1 with a single erroneous glitch of class 2 at t=2
    # [1, 1, 2, 1, 1]
    # For K=3:
    # t=0: [1] -> 1
    # t=1: [1, 1] -> 1
    # t=2: [1, 1, 2] -> 1 (mode is 1, count=2 vs count=1)
    # t=3: [1, 2, 1] -> 1 (mode is 1, count=2 vs count=1)
    # t=4: [2, 1, 1] -> 1 (mode is 1, count=2 vs count=1)
    raw = np.array([1, 1, 2, 1, 1], dtype=np.int64)
    smoothed = causal_majority_vote(raw, window_size=3)
    np.testing.assert_array_equal(smoothed, np.array([1, 1, 1, 1, 1]))


def test_causality_strictness() -> None:
    """Smoothed output at step t must NEVER depend on step t+1 or beyond."""
    raw = np.array([0, 0, 0, 5, 5, 5], dtype=np.int64)
    # At t=2, future values (5, 5, 5) must not influence output at t=2
    smoothed = causal_majority_vote(raw, window_size=5)
    assert smoothed[0] == 0
    assert smoothed[1] == 0
    assert smoothed[2] == 0
    # At t=3: buffer has [0, 0, 0, 5], mode is 0
    assert smoothed[3] == 0
    # At t=4: buffer has [0, 0, 0, 5, 5], mode is 0 (count 3 vs 2)
    assert smoothed[4] == 0
    # At t=5: buffer has [0, 0, 5, 5, 5], mode is 5 (count 3 vs 2)
    assert smoothed[5] == 5


def test_tie_breaking_persistence() -> None:
    """Persistence retains the last output when counts tie."""
    # Buffer of size 2: [1, 2] -> tie between 1 and 2
    # At t=0: pred=1, out=1
    # At t=1: pred=2, buffer=[1, 2], counts tie. Since last_output was 1, winner is 1.
    filter_obj = CausalTemporalMajorityVote(window_size=2, tie_breaker="persistence")
    assert filter_obj.step(1) == 1
    assert filter_obj.step(2) == 1  # Persistence holds 1!
    # At t=2: pred=2, buffer=[2, 2], winner is 2.
    assert filter_obj.step(2) == 2


def test_tie_breaking_latest() -> None:
    """Latest picks the most recent prediction in the buffer during a tie."""
    filter_obj = CausalTemporalMajorityVote(window_size=2, tie_breaker="latest")
    assert filter_obj.step(1) == 1
    assert filter_obj.step(2) == 2  # Latest picks 2!


def test_tie_breaking_smallest() -> None:
    """Smallest picks the minimal integer class during a tie."""
    filter_obj = CausalTemporalMajorityVote(window_size=2, tie_breaker="smallest")
    assert filter_obj.step(4) == 4
    # Buffer has [4, 2]. Tie. Smallest picks 2.
    assert filter_obj.step(2) == 2


def test_stream_boundary_reset() -> None:
    """Verify that buffer clears at segment boundaries, preventing cross-leakage."""
    # Two trials of length 3:
    # Trial A: [8, 8, 8]
    # Trial B: [3, 3, 3]
    # If no reset: at start of Trial B, buffer has [8, 8, 3] -> mode would be 8!
    # With reset at index 3: Trial B starts with fresh buffer [3] -> mode is 3!
    preds = np.array([8, 8, 8, 3, 3, 3], dtype=np.int64)
    boundaries = [3]

    # Without boundary awareness (continuous stream):
    unaware = causal_majority_vote(preds, window_size=3)
    assert unaware[3] == 8  # Leaked! Buffer has [8, 8, 3], so mode is 8!

    # With boundary awareness:
    aware = stream_causal_majority_vote(preds, boundaries=boundaries, window_size=3)
    assert aware[3] == 3  # Fresh start: no leakage!
    np.testing.assert_array_equal(aware, np.array([8, 8, 8, 3, 3, 3]))

    # Same test with segment_ids:
    seg_ids = [0, 0, 0, 1, 1, 1]
    aware_seg = stream_causal_majority_vote(preds, segment_ids=seg_ids, window_size=3)
    np.testing.assert_array_equal(aware_seg, np.array([8, 8, 8, 3, 3, 3]))


def test_causal_soft_voting() -> None:
    """Test confidence accumulation over sliding window."""
    probs = np.array(
        [
            [0.6, 0.4],
            [0.4, 0.6],
            [0.4, 0.6],
            [0.7, 0.3],
        ],
        dtype=np.float32,
    )
    # K=1
    res_k1 = causal_soft_voting(probs, window_size=1)
    np.testing.assert_array_equal(res_k1, [0, 1, 1, 0])

    # K=2:
    # t=0: [0.6, 0.4] -> 0
    # t=1: sum = [1.0, 1.0] -> 0 (argmax ties to first)
    # t=2: sum = [0.8, 1.2] -> 1
    # t=3: sum = [1.1, 0.9] -> 0
    res_k2 = causal_soft_voting(probs, window_size=2)
    np.testing.assert_array_equal(res_k2, [0, 0, 1, 0])


def test_compute_smoothing_metrics() -> None:
    """Ensure metric computation outputs valid delta and delays."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    metrics = compute_smoothing_metrics(
        y_true=y_true,
        y_pred_smoothed=y_pred,
        baseline_acc_pct=80.0,
        baseline_f1_pct=80.0,
        window_size_k=5,
        stride_ms=50.0,
    )
    assert metrics.window_size_k == 5
    assert metrics.algorithmic_delay_ms == 200.0  # (5-1) * 50
    assert metrics.top1_accuracy_pct == 100.0
    assert metrics.delta_accuracy_pct == 20.0
