import os

from streamlit.testing.v1 import AppTest

APP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_starts_without_error_and_has_three_tabs():
    at = AppTest.from_file(APP_FILE, default_timeout=120).run()
    assert len(at.exception) == 0
    assert [tab.label for tab in at.tabs] == ["Dự đoán", "Lịch sử dự đoán", "Đánh giá mô hình"]


def test_evaluation_tab_shows_test_accuracy_of_both_models():
    at = AppTest.from_file(APP_FILE, default_timeout=120).run()
    labels = [metric.label for metric in at.metric]
    assert "CNN" in labels
    assert "MLP" in labels
