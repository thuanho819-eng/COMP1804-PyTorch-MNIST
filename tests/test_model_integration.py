import torch

from src.model import CNN, MLP, create_model
from src.predict import load_model, predict_topk


def test_create_model_returns_expected_architectures():
    assert isinstance(create_model("mlp"), MLP)
    assert isinstance(create_model("cnn"), CNN)


def test_saved_checkpoints_load_and_return_top_predictions():
    for name in ("mlp", "cnn"):
        model = load_model(name, f"models/{name}.pt")
        image = torch.rand(1, 28, 28)
        labels, probabilities = predict_topk(model, image, k=3)

        assert len(labels) == 3
        assert len(probabilities) == 3
        assert len(set(labels)) == 3
        assert all(0 <= label <= 9 for label in labels)
        assert all(0.0 <= probability <= 1.0 for probability in probabilities)


def test_predict_rejects_wrong_image_shape():
    model = load_model("cnn", "models/cnn.pt")
    try:
        predict_topk(model, torch.rand(32, 32))
    except ValueError as error:
        assert "28x28" in str(error)
    else:
        raise AssertionError("Expected predict_topk to reject a 32x32 image")
