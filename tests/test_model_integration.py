from PIL import Image, ImageDraw

from src.model import CNN, MLP, create_model
from src.predict import load_model, predict, preprocess


def test_create_model_returns_expected_architectures():
    assert isinstance(create_model("mlp"), MLP)
    assert isinstance(create_model("cnn"), CNN)


def test_saved_checkpoints_load_and_return_probabilities():
    for name in ("MLP", "CNN"):
        model, trained = load_model(name)
        assert trained, f"Không tìm thấy file mô hình {name} trong thư mục models"

        probs = predict(model, Image.new("L", (28, 28)))
        assert len(probs) == 10
        assert abs(sum(probs) - 1) < 1e-4
        assert all(0.0 <= p <= 1.0 for p in probs)


def test_preprocess_returns_28x28_white_digit_on_black():
    # Ảnh giống ảnh chụp giấy: nền trắng, nét đen, kích thước bất kỳ
    image = Image.new("RGB", (300, 200), (250, 250, 250))
    ImageDraw.Draw(image).line([(150, 30), (150, 170)], fill=(0, 0, 0), width=15)

    result = preprocess(image)
    assert result.size == (28, 28)
    assert result.mode == "L"
    assert result.getpixel((0, 0)) == 0          # góc ảnh là nền đen
    assert result.getextrema()[1] > 200          # có nét chữ trắng


def test_cnn_recognises_a_drawn_digit_one():
    image = Image.new("RGB", (300, 300), (245, 245, 240))
    ImageDraw.Draw(image).line([(150, 50), (150, 250)], fill=(15, 15, 15), width=20)

    model, _ = load_model("CNN")
    probs = predict(model, preprocess(image))
    assert probs.index(max(probs)) == 1
