"""Small, randomly initialized CNN baselines for prepared NightLab inputs.

Models return unnormalized class logits. Preprocessing and training-set-only
normalization belong to the input pipeline, not to these networks.
"""

import torch
from torch import nn


def _validate_input(x: torch.Tensor, sample_shape: tuple[int, ...]) -> None:
    expected = f"[N, {', '.join(map(str, sample_shape))}]"
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor with shape {expected}")
    if x.ndim != len(sample_shape) + 1 or tuple(x.shape[1:]) != sample_shape:
        raise ValueError(f"Expected input shape {expected}; got {tuple(x.shape)}")
    if x.shape[0] < 1:
        raise ValueError("Expected a nonempty batch (N >= 1)")


class AudioCNN(nn.Module):
    """Binary audio classifier accepting float tensors shaped [N, 1, 32, 100]."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(1), nn.Dropout(0.1), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_input(x, (1, 32, 100))
        return self.classifier(self.features(x))


class WearableCNN(nn.Module):
    """Epoch classifier for [N, C, 30], including HR/accel/fused C=2/5/7."""

    def __init__(self, in_channels: int = 7, num_classes: int = 5) -> None:
        super().__init__()
        for name, value in (("in_channels", in_channels), ("num_classes", num_classes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, 5, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1), nn.Dropout(0.1), nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_input(x, (self.in_channels, 30))
        return self.classifier(self.features(x))


def build_model(task: str, in_channels: int | None = None) -> nn.Module:
    """Build the audio or wearable baseline without allocating GPU resources."""
    if task == "audio":
        if in_channels is not None and (type(in_channels) is not int or in_channels != 1):
            raise ValueError("audio requires in_channels=1 (or None)")
        return AudioCNN()
    if task == "wearable":
        return WearableCNN(in_channels=7 if in_channels is None else in_channels)
    raise ValueError(f"Unknown task {task!r}; expected 'audio' or 'wearable'")
