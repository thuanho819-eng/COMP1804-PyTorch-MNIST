import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

from src.model import create_model
from src.predict import PROJECT_DIR, predict, preprocess
from src.train import evaluate, set_seed, train_one_epoch

EPOCHS = 3          # số epoch cho mỗi cấu hình (tăng lên nếu máy mạnh)
BATCH_SIZE = 128
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SAMPLES_DIR = os.path.join(PROJECT_DIR, "samples")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# Các cấu hình cần so sánh
CONFIGS = [
    {"name": "Adam lr=0.001", "optimizer": "Adam", "lr": 0.001, "augment": False},
    {"name": "Adam lr=0.01", "optimizer": "Adam", "lr": 0.01, "augment": False},
    {"name": "SGD lr=0.01", "optimizer": "SGD", "lr": 0.01, "augment": False},
    {"name": "SGD lr=0.1", "optimizer": "SGD", "lr": 0.1, "augment": False},
    {"name": "Adam lr=0.001 + tăng cường", "optimizer": "Adam", "lr": 0.001, "augment": True},
]

normalize = transforms.Normalize((0.1307,), (0.3081,))

plain_transform = transforms.Compose([transforms.ToTensor(), normalize])

augment_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    normalize,
])


def samples_accuracy(model):
    """Tỷ lệ đoán đúng trên ảnh tự viết (tên file bắt đầu bằng chữ số đúng)."""
    files = [f for f in os.listdir(SAMPLES_DIR)
             if f.lower().endswith((".png", ".jpg", ".jpeg")) and f[0].isdigit()]
    if len(files) == 0:
        return None
    model.cpu()
    correct = 0
    for name in files:
        probs = predict(model, preprocess(Image.open(os.path.join(SAMPLES_DIR, name))))
        if probs.index(max(probs)) == int(name[0]):
            correct += 1
    return correct / len(files)


def run_config(config, test_loader, device):
    """Train CNN theo một cấu hình, trả về kết quả từng epoch và kết quả cuối."""
    set_seed(42)  # cùng điểm xuất phát để so sánh công bằng

    transform = augment_transform if config["augment"] else plain_transform
    train_data = datasets.MNIST(DATA_DIR, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model = create_model("cnn").to(device)
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    history = []
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        history.append({
            "Cấu hình": config["name"],
            "Epoch": epoch,
            "Train loss": round(train_loss, 4),
            "Test accuracy (%)": round(test_acc * 100, 2),
        })
        print(f"  Epoch {epoch}/{EPOCHS}: train loss {train_loss:.4f}, test accuracy {test_acc:.2%}")
    minutes = (time.time() - start) / 60

    sample_acc = samples_accuracy(model)
    result = {
        "Cấu hình": config["name"],
        "Optimizer": config["optimizer"],
        "Learning rate": config["lr"],
        "Tăng cường dữ liệu": "Có" if config["augment"] else "Không",
        "Test accuracy (%)": history[-1]["Test accuracy (%)"],
        "Ảnh tự viết đúng (%)": None if sample_acc is None else round(sample_acc * 100, 1),
        "Thời gian (phút)": round(minutes, 1),
    }
    return history, result


def save_charts(history_df, result_df):
    """Vẽ 2 biểu đồ so sánh các cấu hình."""
    # 1. Độ chính xác trên tập test qua từng epoch
    figure, axis = plt.subplots(figsize=(8, 5))
    for name, group in history_df.groupby("Cấu hình", sort=False):
        axis.plot(group["Epoch"], group["Test accuracy (%)"], marker="o", label=name)
    axis.set(xlabel="Epoch", ylabel="Test accuracy (%)",
             title="So sánh độ chính xác trên tập test theo cấu hình")
    axis.set_xticks(range(1, EPOCHS + 1))
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS_DIR, "experiments_test_accuracy.png"), dpi=150)
    plt.close(figure)

    # 2. Cột: độ chính xác cuối trên tập test và trên ảnh tự viết
    figure, axis = plt.subplots(figsize=(9, 5))
    x = range(len(result_df))
    axis.bar([i - 0.2 for i in x], result_df["Test accuracy (%)"], width=0.4, label="Tập test MNIST")
    if result_df["Ảnh tự viết đúng (%)"].notna().any():
        axis.bar([i + 0.2 for i in x], result_df["Ảnh tự viết đúng (%)"].fillna(0),
                 width=0.4, label="Ảnh tự viết (samples)")
    axis.set_xticks(list(x))
    axis.set_xticklabels(result_df["Cấu hình"], rotation=15, ha="right")
    axis.set(ylabel="Độ chính xác (%)", ylim=(0, 105),
             title="Độ chính xác cuối cùng của từng cấu hình")
    axis.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(RESULTS_DIR, "experiments_summary.png"), dpi=150)
    plt.close(figure)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = datasets.MNIST(DATA_DIR, train=False, transform=plain_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    all_history = []
    results = []
    for i, config in enumerate(CONFIGS, start=1):
        print(f"[{i}/{len(CONFIGS)}] {config['name']}")
        history, result = run_config(config, test_loader, device)
        all_history.extend(history)
        results.append(result)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    history_df = pd.DataFrame(all_history)
    result_df = pd.DataFrame(results)
    history_df.to_csv(os.path.join(RESULTS_DIR, "experiments_history.csv"), index=False, encoding="utf-8-sig")
    result_df.to_csv(os.path.join(RESULTS_DIR, "experiments.csv"), index=False, encoding="utf-8-sig")
    save_charts(history_df, result_df)

    print()
    print(result_df.to_string(index=False))
    best = result_df.loc[result_df["Test accuracy (%)"].idxmax()]
    print(f"\nCấu hình tốt nhất trên tập test: {best['Cấu hình']} ({best['Test accuracy (%)']}%)")
    print("Đã lưu results/experiments.csv và 2 biểu đồ experiments_*.png")


if __name__ == "__main__":
    main()
