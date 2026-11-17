"""Causal Temporal Majority Voting and Post-Processing Filter for sEMG Streams.

In real-time prosthetic control and sEMG sequence processing, consecutive windows
(e.g., 200 ms duration, 50 ms stride) exhibit high temporal correlation.
Single-frame inference often exhibits transient jitter and spurious classifications
due to electrode noise or motion artifacts.

This module provides:
1. `CausalTemporalMajorityVote`: A causal streaming mode filter maintaining a circular
   FIFO buffer of size K with configurable tie-breaking policies ("persistence", "latest", "smallest").
2. `causal_majority_vote`: Vectorized batch operation over a sequence of predictions.
3. `stream_causal_majority_vote`: Boundary-aware sequence smoothing that strictly resets
   the filter at segment/repetition/exercise boundaries to prevent cross-trial contamination.
4. `causal_soft_voting`: Causal temporal moving average across probability/score vectors.
5. `compute_smoothing_metrics`: Benchmark evaluator reporting Top-1 Accuracy, Balanced
   Accuracy, Macro F1, Algorithmic Latency, and Delta vs Raw.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


@dataclass(frozen=True)
class TemporalSmoothingResult:
    """Benchmark outcome for a specific filter configuration."""

    window_size_k: int
    stride_ms: float
    algorithmic_delay_ms: float
    top1_accuracy_pct: float
    balanced_accuracy_pct: float
    macro_f1_pct: float
    delta_accuracy_pct: float
    delta_f1_pct: float


class CausalTemporalMajorityVote:
    """Causal sliding-window majority voting filter with zero future-leakage.

    Parameters:
        window_size: Number of past predictions in the temporal window (K >= 1).
        tie_breaker: Strategy when two or more classes tie for the highest frequency:
            - "persistence": Retain the previous smoothed prediction if it is one
              of the tied classes (hysteresis). Otherwise choose the most recent.
            - "latest": Choose the tied class that appeared most recently in the window.
            - "smallest": Choose the class with the smallest integer index (deterministic).
    """

    def __init__(self, window_size: int = 5, tie_breaker: str = "persistence") -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if tie_breaker not in ("persistence", "latest", "smallest"):
            raise ValueError(
                f"Unknown tie_breaker '{tie_breaker}'. Allowed: 'persistence', 'latest', 'smallest'"
            )
        self.window_size = int(window_size)
        self.tie_breaker = tie_breaker
        self.buffer: deque[int] = deque(maxlen=self.window_size)
        self.last_output: int | None = None

    def reset(self) -> None:
        """Clear the FIFO buffer and reset internal state."""
        self.buffer.clear()
        self.last_output = None

    def step(self, pred: int) -> int:
        """Process one incoming prediction causally.

        Args:
            pred: Raw model prediction for the current time step.

        Returns:
            Smoothed prediction as an integer.
        """
        p_val = int(pred)
        self.buffer.append(p_val)

        if self.window_size == 1 or len(self.buffer) == 1:
            self.last_output = p_val
            return p_val

        # Count frequencies in FIFO buffer
        counts = Counter(self.buffer)
        max_count = max(counts.values())
        candidates = [cls for cls, count in counts.items() if count == max_count]

        if len(candidates) == 1:
            winner = candidates[0]
        else:
            # Resolve tie
            if self.tie_breaker == "persistence" and self.last_output in candidates:
                winner = self.last_output
            elif self.tie_breaker in ("persistence", "latest"):
                # Pick the candidate appearing most recently in the buffer
                for item in reversed(self.buffer):
                    if item in candidates:
                        winner = item
                        break
                else:
                    winner = candidates[0]
            else:  # smallest
                winner = min(candidates)

        self.last_output = winner
        return winner

    def process_sequence(self, predictions: Sequence[int] | np.ndarray) -> np.ndarray:
        """Process an entire contiguous sequence causally."""
        out = np.empty(len(predictions), dtype=np.int64)
        for i, p in enumerate(predictions):
            out[i] = self.step(int(p))
        return out


def causal_majority_vote(
    predictions: Sequence[int] | np.ndarray,
    window_size: int = 5,
    tie_breaker: str = "persistence",
) -> np.ndarray:
    """Apply causal temporal majority voting over a sequence of predictions.

    Args:
        predictions: 1D array of predicted class labels.
        window_size: Odd or positive integer window length K.
        tie_breaker: Tie-breaking strategy ("persistence", "latest", "smallest").

    Returns:
        1D array of smoothed predictions of same shape and length.
    """
    preds_arr = np.asarray(predictions, dtype=np.int64)
    if len(preds_arr) == 0:
        return np.empty(0, dtype=np.int64)
    if window_size <= 1:
        return preds_arr.copy()

    filter_obj = CausalTemporalMajorityVote(window_size=window_size, tie_breaker=tie_breaker)
    return filter_obj.process_sequence(preds_arr)


def stream_causal_majority_vote(
    predictions: Sequence[int] | np.ndarray,
    boundaries: Sequence[int] | np.ndarray | None = None,
    segment_ids: Sequence[int | str] | np.ndarray | None = None,
    window_size: int = 5,
    tie_breaker: str = "persistence",
) -> np.ndarray:
    """Boundary-aware causal majority voting filter across segmented streams.

    Resets the temporal history buffer at segment/repetition boundaries so that
    no past information leaks across independent trials or exercises.

    Args:
        predictions: 1D array of predicted class labels.
        boundaries: Explicit start indices of new segments (e.g., [0, 2600, 5200, ...]).
        segment_ids: Array of segment identifiers of same length as predictions.
                     A reset occurs whenever segment_ids[i] != segment_ids[i-1].
        window_size: Number of past windows K.
        tie_breaker: Strategy for ties ("persistence", "latest", "smallest").

    Returns:
        1D array of smoothed predictions.
    """
    preds_arr = np.asarray(predictions, dtype=np.int64)
    n = len(preds_arr)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if window_size <= 1:
        return preds_arr.copy()

    # Determine boundary set
    boundary_set: set[int] = {0}
    if boundaries is not None:
        boundary_set.update(int(b) for b in boundaries if 0 <= b < n)
    if segment_ids is not None:
        seg_arr = np.asarray(segment_ids)
        if len(seg_arr) != n:
            raise ValueError(f"segment_ids length {len(seg_arr)} != predictions length {n}")
        diff_indices = np.where(seg_arr[1:] != seg_arr[:-1])[0] + 1
        boundary_set.update(int(b) for b in diff_indices)

    filter_obj = CausalTemporalMajorityVote(window_size=window_size, tie_breaker=tie_breaker)
    out = np.empty(n, dtype=np.int64)

    for i in range(n):
        if i in boundary_set:
            filter_obj.reset()
        out[i] = filter_obj.step(int(preds_arr[i]))

    return out


def causal_soft_voting(
    scores_or_probs: np.ndarray,
    boundaries: Sequence[int] | np.ndarray | None = None,
    segment_ids: Sequence[int | str] | np.ndarray | None = None,
    window_size: int = 5,
) -> np.ndarray:
    """Causal temporal moving average across probability/logit distributions.

    Accumulates the past K distributions causally and selects the argmax:
        s_t = sum_{j=0}^{K-1} P(t - j)
        y_hat_t = argmax_c (s_t[c])

    Args:
        scores_or_probs: 2D array of shape (N, num_classes).
        boundaries: Explicit boundary indices where accumulation resets.
        segment_ids: 1D array of segment identifiers.
        window_size: Number of past frames K.

    Returns:
        1D array of predicted class labels.
    """
    scores = np.asarray(scores_or_probs, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"scores_or_probs must be 2D (N, C), got ndim={scores.ndim}")
    n, num_classes = scores.shape
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if window_size <= 1:
        return np.argmax(scores, axis=1)

    boundary_set: set[int] = {0}
    if boundaries is not None:
        boundary_set.update(int(b) for b in boundaries if 0 <= b < n)
    if segment_ids is not None:
        seg_arr = np.asarray(segment_ids)
        diff_indices = np.where(seg_arr[1:] != seg_arr[:-1])[0] + 1
        boundary_set.update(int(b) for b in diff_indices)

    out = np.empty(n, dtype=np.int64)
    buf: deque[np.ndarray] = deque(maxlen=window_size)
    running_sum = np.zeros(num_classes, dtype=np.float32)

    for i in range(n):
        if i in boundary_set:
            buf.clear()
            running_sum.fill(0.0)

        vec = scores[i]
        if len(buf) == window_size:
            old_vec = buf.popleft()
            running_sum -= old_vec
        buf.append(vec)
        running_sum += vec

        out[i] = int(np.argmax(running_sum))

    return out


def compute_smoothing_metrics(
    y_true: np.ndarray,
    y_pred_smoothed: np.ndarray,
    baseline_acc_pct: float,
    baseline_f1_pct: float,
    window_size_k: int,
    stride_ms: float = 50.0,
) -> TemporalSmoothingResult:
    """Evaluate accuracy, F1, and algorithmic latency for a smoothed sequence."""
    acc = float(accuracy_score(y_true, y_pred_smoothed)) * 100.0
    bal = float(balanced_accuracy_score(y_true, y_pred_smoothed)) * 100.0
    f1 = float(f1_score(y_true, y_pred_smoothed, average="macro", zero_division=0)) * 100.0
    delay_ms = float(max(0, window_size_k - 1) * stride_ms)

    return TemporalSmoothingResult(
        window_size_k=window_size_k,
        stride_ms=stride_ms,
        algorithmic_delay_ms=delay_ms,
        top1_accuracy_pct=round(acc, 2),
        balanced_accuracy_pct=round(bal, 2),
        macro_f1_pct=round(f1, 2),
        delta_accuracy_pct=round(acc - baseline_acc_pct, 2),
        delta_f1_pct=round(f1 - baseline_f1_pct, 2),
    )
