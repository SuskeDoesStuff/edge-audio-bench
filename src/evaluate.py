"""Accuracy-under-noise sweep -> the headline result table.

Loads the trained TinyKWS, and for each SNR level injects white noise at the
waveform level, extracts features, and measures test accuracy. Writes
results/accuracy_vs_noise.csv and prints the table.
"""
from __future__ import annotations
import sys
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from noise import mix_at_snr, white_noise
from features import logmel
from model import TinyKWS
from data import SpeechCommandsSubset, collate, LABELS

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "tinykws.pt"
SNR_LEVELS_DB = [40, 20, 10, 5, 0, -5]   # ~clean down to heavily corrupted


@torch.no_grad()
def accuracy_at_snr(model, loader, snr_db, dev, rng, max_batches=None) -> float:
    model.eval()
    correct = tot = 0
    for bi, (wav, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        w = wav.numpy()
        noisy = np.stack([
            mix_at_snr(w[i], white_noise(w.shape[1], rng), snr_db, rng)
            for i in range(w.shape[0])
        ])
        x = logmel(torch.from_numpy(noisy).to(dev))
        pred = model(x).argmax(1).cpu()
        correct += (pred == y).sum().item()
        tot += y.numel()
    return correct / max(tot, 1)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not CKPT.exists():
        raise SystemExit(f"no checkpoint at {CKPT}; run train.py first")
    model = TinyKWS(n_classes=len(LABELS)).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev))
    test = DataLoader(SpeechCommandsSubset("testing"), batch_size=256,
                      collate_fn=collate)
    rng = np.random.default_rng(0)
    rows = []
    print(f"{'SNR (dB)':>10} | {'accuracy':>9}")
    print("-" * 23)
    for s in SNR_LEVELS_DB:
        acc = accuracy_at_snr(model, test, s, dev, rng)
        rows.append((s, round(acc, 4)))
        print(f"{s:>10} | {acc:>9.3f}")
    out = ROOT / "results" / "accuracy_vs_noise.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "accuracy"])
        w.writerows(rows)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
