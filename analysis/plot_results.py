"""Turn the result CSVs into the two figures that tell the story.

Reads results/accuracy_vs_noise.csv (from evaluate.py) and
results/latency.csv (from bench_quant.py) and writes:
  results/accuracy_vs_noise.png  - the degradation curve
  results/latency.png            - fp32 vs int8 latency, mean and p99

Run from the repo root:  python analysis/plot_results.py
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
N_CLASSES = 10  # chance level = 1 / N_CLASSES


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_accuracy() -> None:
    rows = _read_csv(RES / "accuracy_vs_noise.csv")
    snr = [float(r["snr_db"]) for r in rows]
    acc = [float(r["accuracy"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(snr, acc, "o-", color="#2b6cb0", lw=2.2, ms=7, zorder=3)
    ax.axhline(1 / N_CLASSES, ls="--", color="#a0aec0", lw=1.3,
               label=f"chance ({1/N_CLASSES:.2f})")

    ax.invert_xaxis()  # clean (high SNR) on the left, noisier to the right
    ax.set_xlabel("SNR (dB)   \u2190 cleaner      noisier \u2192")
    ax.set_ylabel("test accuracy")
    ax.set_title("Keyword-spotting accuracy under additive white noise")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, zorder=0)

    # annotate the parity point, where speech and noise powers are equal
    if 0.0 in snr:
        i = snr.index(0.0)
        ax.annotate("0 dB: signal = noise power",
                    xy=(0, acc[i]), xytext=(-3, acc[i] + 0.22),
                    fontsize=9, color="#4a5568",
                    arrowprops=dict(arrowstyle="->", color="#4a5568"))
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(RES / "accuracy_vs_noise.png", dpi=150)
    print("wrote results/accuracy_vs_noise.png")


def plot_latency() -> None:
    path = RES / "latency.csv"
    if not path.exists():
        print("(no latency.csv; skipping latency plot)")
        return
    rows = {r["variant"]: r for r in _read_csv(path)}
    variants = list(rows.keys())
    means = [float(rows[v]["mean_ms"]) for v in variants]
    p99s = [float(rows[v]["p99_ms"]) for v in variants]

    x = range(len(variants))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar([i - w / 2 for i in x], means, w, label="mean", color="#2b6cb0")
    ax.bar([i + w / 2 for i in x], p99s, w, label="p99 (tail)", color="#f6ad55")
    ax.set_xticks(list(x))
    ax.set_xticklabels([v.upper() for v in variants])
    ax.set_ylabel("latency (ms)  \u2014  batch 1, CPU")
    ax.set_title("Inference latency: fp32 vs int8 (dynamic)")
    ax.grid(alpha=0.25, axis="y")
    for i, (m, p) in enumerate(zip(means, p99s)):
        ax.text(i - w / 2, m + 0.01, f"{m:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, p + 0.01, f"{p:.2f}", ha="center", fontsize=8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "latency.png", dpi=150)
    print("wrote results/latency.png")


if __name__ == "__main__":
    plot_accuracy()
    plot_latency()
