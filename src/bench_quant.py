"""fp32 vs int8 latency for the real-time (batch-1) scenario, on CPU.

Edge inference is single-clip and usually CPU-bound, so we time batch-1
forward passes there. Dynamic quantization here targets the Linear head;
TinyKWS is conv-dominated, so expect a modest speedup and note it honestly:
the real conv-level win comes from static quantization or an exported
runtime, which is a documented next step, not a claim made here.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from model import TinyKWS, quantize_dynamic, count_params
from latency import benchmark
from data import LABELS

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "tinykws.pt"


def load(dev="cpu"):
    m = TinyKWS(n_classes=len(LABELS))
    if CKPT.exists():
        m.load_state_dict(torch.load(CKPT, map_location=dev))
    else:
        print("(no checkpoint; benchmarking an untrained model, timing is valid)")
    return m.eval().to(dev)


def main():
    dev = "cpu"
    m = load(dev)
    mq = quantize_dynamic(m)
    x = torch.randn(1, 1, 40, 101)   # one ~1s clip
    fp32 = benchmark(lambda: m(x))
    int8 = benchmark(lambda: mq(x))
    print(f"params       : {count_params(m):,}")
    print(f"fp32 latency : {fp32}")
    print(f"int8 latency : {int8}")
    print(f"speedup      : {fp32.mean_ms / int8.mean_ms:.2f}x (head-only dynamic quant)")


if __name__ == "__main__":
    main()
