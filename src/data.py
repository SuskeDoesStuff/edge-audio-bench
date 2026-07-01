"""Speech Commands, filtered to a fixed keyword set.

Uses torchaudio's built-in downloader (no scraping, clean license). We keep
the 10 core command words so the task is a clean N-class keyword spotter;
"unknown"/"silence" buckets are deliberately left out to keep the task a
clean N-class problem, and can be added later if wanted.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchaudio

LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class SpeechCommandsSubset(Dataset):
    """torchaudio SPEECHCOMMANDS restricted to LABELS. subset in
    {'training','validation','testing'}."""

    def __init__(self, subset: str):
        DATA_DIR.mkdir(exist_ok=True)
        self.ds = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(DATA_DIR), download=True, subset=subset)
        # filter by label parsed from path, without loading audio
        self.indices = [i for i, p in enumerate(self.ds._walker)
                        if Path(p).parent.name in LABEL_TO_IDX]
        if not self.indices:
            raise RuntimeError(
                "no samples matched LABELS; check torchaudio _walker format")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        wav, _sr, label, *_ = self.ds[self.indices[k]]
        return wav.squeeze(0), LABEL_TO_IDX[label]   # (T,), int


def collate(batch):
    """Pad/crop each clip to 1s and stack -> (B, 16000), (B,)."""
    from features import fix_length, SAMPLE_RATE
    wavs = torch.stack([fix_length(w, SAMPLE_RATE) for w, _ in batch])
    labels = torch.tensor([l for _, l in batch], dtype=torch.long)
    return wavs, labels
