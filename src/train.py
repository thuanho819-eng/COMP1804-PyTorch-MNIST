"""Training utilities and command-line entry point for MNIST models."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import create_model


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch for repeatable experiments."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    *,
    download: bool = True,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Create normalized MNIST training and test data loaders."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data = datasets.MNIST(data_dir, train=True, transform=transform, download=download)
    test_data = datasets.MNIST(data_dir, train=False, transform=transform, download=download)
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (predictions.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate a model and return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        total_loss += criterion(predictions, labels).item() * labels.size(0)
        correct += (predictions.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def fit(
    model: nn.Module,
    train_loader: DataLoader[Any],
    test_loader: DataLoader[Any],
    *,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Train a model and return train/test loss and accuracy per epoch."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}
    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        for key, value in (
            ("train_loss", train_loss),
            ("train_accuracy", train_accuracy),
            ("test_loss", test_loss),
            ("test_accuracy", test_accuracy),
        ):
            history[key].append(value)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.2%}, "
            f"test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.2%}"
        )
    return history


def save_history(history: dict[str, list[float]], output_dir: str | Path) -> None:
    """Save metrics as CSV and create loss and accuracy plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *history])
        for index in range(len(history["train_loss"])):
            writer.writerow([index + 1, *(history[key][index] for key in history)])

    epochs = range(1, len(history["train_loss"]) + 1)
    for metric, title, filename, ylabel in (
        ("loss", "Training and test loss", "loss.png", "Loss"),
        ("accuracy", "Training and test accuracy", "accuracy.png", "Accuracy"),
    ):
        figure, axis = plt.subplots()
        axis.plot(epochs, history[f"train_{metric}"], label="Train")
        axis.plot(epochs, history[f"test_{metric}"], label="Test")
        axis.set(xlabel="Epoch", ylabel=ylabel, title=title)
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path / filename, dpi=150)
        plt.close(figure)


def main() -> None:
    """Parse command-line options, train a model, and save its artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("cnn", "mlp"), default="cnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    train_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)
    model = create_model(args.model)
    history = fit(
        model,
        train_loader,
        test_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    args.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_dir / f"{args.model}.pt")
    save_history(history, args.output_dir)
    print(f"Saved model to {args.model_dir / f'{args.model}.pt'}")


if __name__ == "__main__":
    main()