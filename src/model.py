"""Neural network architectures used by the MNIST project."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """A fully connected classifier for 28x28 grayscale images."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalised class scores for a batch of images."""
        return self.network(x)


class CNN(nn.Module):
    """A compact convolutional classifier for 28x28 grayscale images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalised class scores for a batch of images."""
        return self.classifier(self.features(x))


def create_model(name: str) -> nn.Module:
    """Create an MLP or CNN from a case-insensitive model name."""
    normalized_name = name.lower()
    if normalized_name == "mlp":
        return MLP()
    if normalized_name == "cnn":
        return CNN()
    raise ValueError(f"Unknown model '{name}'. Expected 'mlp' or 'cnn'.")