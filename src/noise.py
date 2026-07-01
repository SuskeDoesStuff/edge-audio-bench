"""SNR-controlled noise mixing for robustness testing.

The core of the "stress-test under noise" deliverable: take a clean signal,
add noise scaled to a precise signal-to-noise ratio, and you can sweep SNR
from clean down to heavily corrupted and measure how the model degrades.

Pure numpy, no heavy deps, so this runs anywhere.
"""

from __future__ import annotations
import numpy as np


def _power(x: np.ndarray) -> float:
    """Mean power of a signal."""
    return float(np.mean(x.astype(np.float64) ** 2))


def mix_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add `noise` to `signal` scaled to achieve a target SNR in dB.

    SNR_dB = 10 * log10(P_signal / P_noise_scaled), so the scale that
    achieves the target is sqrt(P_signal / (P_noise * 10**(snr_db/10))).

    If `noise` is shorter than `signal` it is tiled; if longer, a random
    offset is taken so repeated calls see different noise segments.
    """
    signal = np.asarray(signal, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)

    if noise.shape[0] < signal.shape[0]:
        reps = int(np.ceil(signal.shape[0] / noise.shape[0]))
        noise = np.tile(noise, reps)
    if noise.shape[0] > signal.shape[0]:
        rng = rng or np.random.default_rng()
        start = int(rng.integers(0, noise.shape[0] - signal.shape[0] + 1))
        noise = noise[start : start + signal.shape[0]]

    p_sig, p_noise = _power(signal), _power(noise)
    if p_noise == 0:
        return signal.copy()
    scale = np.sqrt(p_sig / (p_noise * (10.0 ** (snr_db / 10.0))))
    return (signal + scale * noise).astype(np.float32)


def white_noise(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Gaussian white noise of length n."""
    rng = rng or np.random.default_rng()
    return rng.standard_normal(n).astype(np.float32)


def measured_snr(signal: np.ndarray, noisy: np.ndarray) -> float:
    """Recover the realised SNR (dB) from a clean/noisy pair. Used in tests."""
    noise = np.asarray(noisy, np.float64) - np.asarray(signal, np.float64)
    p_sig, p_noise = _power(signal), _power(noise)
    return 10.0 * np.log10(p_sig / p_noise) if p_noise > 0 else float("inf")


if __name__ == "__main__":
    # self-check: mixing at a target SNR should reproduce that SNR
    rng = np.random.default_rng(0)
    sig = np.sin(np.linspace(0, 200 * np.pi, 16000)).astype(np.float32)
    for target in (20, 10, 5, 0, -5):
        noisy = mix_at_snr(sig, white_noise(len(sig), rng), target, rng)
        got = measured_snr(sig, noisy)
        print(f"target {target:+3d} dB -> measured {got:+6.2f} dB")
