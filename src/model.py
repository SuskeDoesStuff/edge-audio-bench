"""Small keyword-spotting model + quantization path.

A compact CNN over log-mel features, sized to be plausible on constrained
hardware. The point of this repo is not a novel architecture; it is the
deployment path: define small, quantise, measure latency, measure accuracy
under noise. Training lives in train.py; this file is the model and the
quantisation step.

Requires torch + torchaudio (see requirements.txt).
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # keep the rest of the repo runnable without torch
    raise SystemExit("model.py needs torch: pip install -r requirements.txt") from e


class TinyKWS(nn.Module):
    """Compact CNN for keyword spotting on log-mel spectrograms.

    Input:  (batch, 1, n_mels, time)
    Output: (batch, n_classes) logits
    """

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, n_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.head(self.features(x).flatten(1))


def quantize_dynamic(model: "nn.Module") -> "nn.Module":
    """Post-training dynamic quantization of the Linear layers (int8).

    Note: this only quantizes Linear layers. TinyKWS is conv-dominated, so the
    effect on latency is negligible (see the README quantization finding); the
    real lever is static quantization or an exported runtime.
    """
    model.eval()
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def count_params(model: "nn.Module") -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = TinyKWS()
    print(f"TinyKWS params: {count_params(m):,}")
    x = torch.randn(1, 1, 40, 101)  # ~1s of 40-mel audio
    print("output shape:", tuple(m(x).shape))
    mq = quantize_dynamic(m)
    print("quantized OK; linear head is now int8 dynamic")
