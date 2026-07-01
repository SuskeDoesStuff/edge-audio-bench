"""Train TinyKWS on Speech Commands to 'good enough to deploy'.

Not a leaderboard run. Trains a few epochs, keeps the best val checkpoint,
and stops. The point of the repo is the deployment + noise analysis, so this
exists only to produce a working model to analyse.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import SpeechCommandsSubset, collate, LABELS
from features import logmel
from model import TinyKWS, count_params

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "tinykws.pt"


@torch.no_grad()
def eval_loader(model, loader, dev) -> float:
    model.eval()
    correct = tot = 0
    for wav, y in loader:
        wav, y = wav.to(dev), y.to(dev)
        pred = model(logmel(wav)).argmax(1)
        correct += (pred == y).sum().item()
        tot += y.numel()
    return correct / max(tot, 1)


def run(epochs: int = 12, batch_size: int = 256, lr: float = 1e-3):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DataLoader(SpeechCommandsSubset("training"), batch_size=batch_size,
                    shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
    va = DataLoader(SpeechCommandsSubset("validation"), batch_size=batch_size,
                    collate_fn=collate, num_workers=2)
    model = TinyKWS(n_classes=len(LABELS)).to(dev)
    print(f"TinyKWS params: {count_params(model):,} | device {dev} | "
          f"train {len(tr.dataset)} val {len(va.dataset)}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    CKPT.parent.mkdir(exist_ok=True)
    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        for wav, y in tr:
            wav, y = wav.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(model(logmel(wav)), y)
            loss.backward()
            opt.step()
        acc = eval_loader(model, va, dev)
        print(f"epoch {ep:2d}  val_acc {acc:.3f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), CKPT)
    print(f"best val_acc {best:.3f}  saved -> {CKPT}")


if __name__ == "__main__":
    run()
