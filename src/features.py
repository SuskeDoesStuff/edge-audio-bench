"""Waveform -> log-mel features, the bridge between raw audio and TinyKWS.

Device-aware so it works whether the batch is on CPU or GPU. The model uses
an adaptive pool, so the exact frame count does not need to be fixed, but we
fix the *waveform* to 1s so batches stack cleanly.
"""
from __future__ import annotations
import torch
import torchaudio

SAMPLE_RATE = 16000
N_MELS = 40

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=N_MELS)
_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)


def fix_length(wav: torch.Tensor, n: int = SAMPLE_RATE) -> torch.Tensor:
    """Pad or crop the last dim to exactly n samples."""
    t = wav.shape[-1]
    if t < n:
        return torch.nn.functional.pad(wav, (0, n - t))
    if t > n:
        return wav[..., :n]
    return wav


def logmel(wav: torch.Tensor) -> torch.Tensor:
    """(T,) or (B,T) waveform -> (1,N_MELS,frames) or (B,1,N_MELS,frames)."""
    dev = wav.device
    mel = _mel.to(dev)
    to_db = _to_db.to(dev)
    wav = fix_length(wav)
    spec = to_db(mel(wav))
    mean = spec.mean(dim=(-2, -1), keepdim=True)
    std = spec.std(dim=(-2, -1), keepdim=True)
    spec = (spec - mean) / (std + 1e-5)
    if spec.dim() == 2:            # (N_MELS,frames)
        return spec.unsqueeze(0)   # -> (1,N_MELS,frames)
    return spec.unsqueeze(1)       # (B,N_MELS,frames) -> (B,1,N_MELS,frames)
