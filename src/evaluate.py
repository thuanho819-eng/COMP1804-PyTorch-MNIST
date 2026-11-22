"""Evaluate a saved MNIST model on the real test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model import create_model
from src.train import get_data_loaders


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    """Calculate test metrics and collect predictions for evaluation."""
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())

    predicted = torch.cat(all_predictions)
    actual = torch.cat(all_labels)
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    for label, prediction in zip(actual, predicted):
        confusion[label.item(), prediction.item()] += 1

    per_digit: dict[str, dict[str, float | int]] = {}
    for digit in range(10):
        true_positive = int(confusion[digit, digit].item())
        support = int(confusion[digit].sum().item())
        predicted_count = int(confusion[:, digit].sum().item())
        per_digit[str(digit)] = {
            "accuracy": true_positive / support if support else 0.0,
            "precision": true_positive / predicted_count if predicted_count else 0.0,
            "support": support,
            "correct": true_positive,
        }

    metrics: dict[str, object] = {
        "model": model.__class__.__name__,
        "test_loss": total_loss / total,
        "test_accuracy": correct / total,
        "test_samples": total,
        "per_digit": per_digit,
    }
    return metrics, confusion, actual != predicted


def save_confusion_matrix(confusion: torch.Tensor, output_path: Path) -> None:
    """Save a labelled confusion matrix."""
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(confusion.numpy(), cmap="Blues")
    axis.set(
        xlabel="Predicted digit",
        ylabel="Actual digit",
        title="MNIST confusion matrix",
        xticks=range(10),
        yticks=range(10),
    )
    for row in range(10):
        for column in range(10):
            axis.text(
                column,
                row,
                str(confusion[row, column].item()),
                ha="center",
                va="center",
                color="white" if confusion[row, column] > confusion.max() / 2 else "black",
                fontsize=7,
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    """Evaluate a saved checkpoint and write JSON and plot artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("cnn", "mlp"), default="cnn")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_data_loaders(args.data_dir, download=False)
    model = create_model(args.model)
    checkpoint = args.model_dir / f"{args.model}.pt"
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    metrics, confusion, _ = evaluate_model(model, test_loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    save_confusion_matrix(confusion, args.output_dir / "confusion_matrix.png")

    print(
        f"{args.model.upper()} test accuracy: {float(metrics['test_accuracy']):.2%}; "
        f"test loss: {float(metrics['test_loss']):.6f}"
    )
    print(f"Saved metrics to {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()