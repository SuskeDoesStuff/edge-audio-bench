"""Accuracy-under-noise sweep -> the headline result table.

This is the deliverable that mirrors the role: one table showing how a
quantised, real-time audio model holds up as conditions get noisy.

Status: harness wired against noise.py (working). Model + dataset hookup is
the active next step, marked TODO below. Running this today prints the table
structure with the noise pipeline exercised on synthetic data.
"""
from __future__ import annotations
import numpy as np
from noise import mix_at_snr, white_noise

SNR_LEVELS_DB = [40, 20, 10, 5, 0, -5]  # "clean" down to heavily corrupted


def evaluate_at_snr(snr_db: float, rng: np.random.Generator) -> dict:
    """Run the model over a noisy eval set at one SNR and return metrics.

    TODO(next): load Speech Commands eval split + trained TinyKWS, extract
    log-mel features, run inference, compute real accuracy. For now this
    exercises the noise pipeline on synthetic clips so the sweep is runnable.
    """
    accs = []
    for _ in range(50):
        clip = np.sin(np.linspace(0, 100 * np.pi, 16000)).astype(np.float32)
        _noisy = mix_at_snr(clip, white_noise(len(clip), rng), snr_db, rng)
        # TODO: feats = logmel(_noisy); pred = model(feats); accs.append(pred == label)
        accs.append(np.nan)  # placeholder until model is wired
    return {"snr_db": snr_db, "accuracy": float(np.nanmean(accs)) if not all(np.isnan(accs)) else None}


def main() -> None:
    rng = np.random.default_rng(0)
    rows = [evaluate_at_snr(s, rng) for s in SNR_LEVELS_DB]
    print(f"{'SNR (dB)':>10} | {'accuracy':>9}")
    print("-" * 23)
    for r in rows:
        acc = "  TODO" if r["accuracy"] is None else f"{r['accuracy']:.3f}"
        print(f"{r['snr_db']:>10} | {acc:>9}")
    print("\nNoise pipeline runs; model/dataset hookup is the next commit.")


if __name__ == "__main__":
    main()
