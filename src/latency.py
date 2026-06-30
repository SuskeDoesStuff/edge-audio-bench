"""Inference latency benchmark for edge deployment.

Real-time on constrained hardware means latency is a first-class metric, not
an afterthought. This harness times any callable (a model's forward pass, a
quantised model, a preprocessing step) with warmup and reports the
distribution, not just the mean, because tail latency is what breaks
real-time guarantees.

Pure numpy/stdlib so it runs without a model present.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class LatencyResult:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_hz: float
    n: int

    def __str__(self) -> str:
        return (f"mean {self.mean_ms:.2f} ms | p50 {self.p50_ms:.2f} | "
                f"p95 {self.p95_ms:.2f} | p99 {self.p99_ms:.2f} | "
                f"{self.throughput_hz:.1f} infer/s (n={self.n})")


def benchmark(fn: Callable[[], object], runs: int = 200, warmup: int = 20) -> LatencyResult:
    """Time `fn` over `runs` calls after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    times = np.empty(runs, dtype=np.float64)
    for i in range(runs):
        t0 = time.perf_counter()
        fn()
        times[i] = (time.perf_counter() - t0) * 1e3  # ms
    return LatencyResult(
        mean_ms=float(times.mean()),
        p50_ms=float(np.percentile(times, 50)),
        p95_ms=float(np.percentile(times, 95)),
        p99_ms=float(np.percentile(times, 99)),
        throughput_hz=1000.0 / float(times.mean()),
        n=runs,
    )


if __name__ == "__main__":
    # demo against a stand-in workload so the harness is runnable today;
    # swap in a real model forward pass once model.py is wired.
    def fake_forward():
        x = np.random.randn(64, 64).astype(np.float32)
        return x @ x.T

    print("stand-in workload:", benchmark(fake_forward))
