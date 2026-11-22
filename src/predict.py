"""Model loading and inference helpers shared by the application."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.model import create_model


def load_model(
    model_name: str,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> nn.Module:
    """Load a named MNIST checkpoint for CPU or CUDA inference."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name)
    state_dict = torch.load(
        Path(checkpoint_path),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_topk(
    model: nn.Module,
    image: torch.Tensor,
    k: int = 3,
    device: torch.device | None = None,
) -> tuple[list[int], list[float]]:
    """Return top-k digit labels and probabilities for one 28x28 image."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if tuple(image.shape[-2:]) != (28, 28):
        raise ValueError(f"Expected a 28x28 image, received shape {tuple(image.shape)}")

    probabilities = torch.softmax(model(image.to(device)), dim=1)[0]
    values, indices = torch.topk(probabilities, k=min(k, 10))
    return indices.cpu().tolist(), values.cpu().tolist()