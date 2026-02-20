"""Safety wrapper for competition robustness.

Provides:
- Latency tracking (P50/P95/P99)
- Exception catching with hover fallback
- Action bounds validation
- Windows-compatible (no signal.alarm)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class SafeActionWrapper:
    """Wraps compute calls with safety guarantees.

    - Catches exceptions and returns hover command
    - Tracks compute latency for monitoring
    - Validates action bounds
    """

    # Default action bounds: [thrust, roll_rate, pitch_rate, yaw_rate]
    ACTION_LOW = np.array([0.0, -8.0, -8.0, -4.0], dtype=np.float32)
    ACTION_HIGH = np.array([1.0, 8.0, 8.0, 4.0], dtype=np.float32)

    # Hover fallback: thrust for ~hover, zero rates
    HOVER_ACTION = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)

    def __init__(
        self,
        action_dim: int = 4,
        max_latency_ms: float = 50.0,
        history_size: int = 1000,
    ) -> None:
        self._action_dim = action_dim
        self._max_latency_ms = max_latency_ms
        self._latencies: deque[float] = deque(maxlen=history_size)
        self._fallback_count = 0
        self._total_calls = 0

    def safe_compute(self, compute_fn: Callable[[], np.ndarray]) -> np.ndarray:
        """Call compute_fn with safety wrapping.

        Args:
            compute_fn: Callable that returns action array

        Returns:
            Valid action array (always within bounds)
        """
        self._total_calls += 1

        try:
            t0 = time.perf_counter()
            action = compute_fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._latencies.append(elapsed_ms)

            if elapsed_ms > self._max_latency_ms:
                logger.warning(
                    "Compute latency %.1fms exceeds max %.1fms",
                    elapsed_ms,
                    self._max_latency_ms,
                )

        except Exception:
            logger.exception("Compute failed, returning hover fallback")
            self._fallback_count += 1
            return self.HOVER_ACTION.copy()

        # Validate
        if not isinstance(action, np.ndarray):
            action = np.asarray(action, dtype=np.float32)

        if action.shape != (self._action_dim,):
            logger.warning("Bad action shape %s, returning hover", action.shape)
            self._fallback_count += 1
            return self.HOVER_ACTION.copy()

        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            logger.warning("NaN/Inf in action, returning hover")
            self._fallback_count += 1
            return self.HOVER_ACTION.copy()

        # Clip to bounds
        action = np.clip(action, self.ACTION_LOW, self.ACTION_HIGH)

        return action

    def get_stats(self) -> dict:
        """Get latency statistics.

        Returns:
            Dict with p50_ms, p95_ms, p99_ms, mean_ms, max_ms,
            fallback_count, total_calls
        """
        if not self._latencies:
            return {
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "mean_ms": 0.0,
                "max_ms": 0.0,
                "fallback_count": self._fallback_count,
                "total_calls": self._total_calls,
            }

        arr = np.array(self._latencies)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(np.mean(arr)),
            "max_ms": float(np.max(arr)),
            "fallback_count": self._fallback_count,
            "total_calls": self._total_calls,
        }
