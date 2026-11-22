import os

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from src.model import CNN, MLP

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_FILES = {"CNN": "cnn.pt", "MLP": "mlp.pt"}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def load_model(name):
    """Tạo mô hình CNN hoặc MLP, nạp trọng số đã huấn luyện nếu có file."""
    if name == "CNN":
        model = CNN()
    else:
        model = MLP()

    path = os.path.join(PROJECT_DIR, "models", MODEL_FILES[name])
    trained = os.path.exists(path)
    if trained:
        model.load_state_dict(torch.load(path, map_location="cpu"))

    model.eval() 
    return model, trained


def preprocess(image):
    """Chuyển ảnh người dùng tải lên thành ảnh 28x28 giống bộ dữ liệu MNIST."""
    gray = ImageOps.exif_transpose(image).convert("L")

    if np.array(gray).mean() > 127:
        gray = ImageOps.invert(gray)

    pixels = np.array(gray)
    pixels[pixels < 50] = 0

    rows, cols = np.nonzero(pixels)

    result = Image.new("L", (28, 28))  # ảnh đen 28x28
    if len(rows) > 0:
        digit = Image.fromarray(pixels[rows.min():rows.max() + 1,
                                       cols.min():cols.max() + 1])

        scale = 20 / max(digit.width, digit.height)
        new_width = max(1, round(digit.width * scale))
        new_height = max(1, round(digit.height * scale))
        digit = digit.resize((new_width, new_height))

        x = (28 - new_width) // 2
        y = (28 - new_height) // 2
        result.paste(digit, (x, y))

        arr = np.array(result, dtype=float)
        total = arr.sum()
        center_y = (arr.sum(axis=1) * np.arange(28)).sum() / total
        center_x = (arr.sum(axis=0) * np.arange(28)).sum() / total
        shifted = Image.new("L", (28, 28))
        shifted.paste(result, (round(14 - center_x), round(14 - center_y)))
        result = shifted

    return result


def predict(model, image28):
    """Trả về danh sách 10 xác suất, phần tử thứ i là xác suất ảnh là chữ số i."""
    x = transform(image28).unsqueeze(0) 
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)[0]
    return probs.tolist()
