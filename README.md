# Ứng dụng nhận diện chữ số viết tay bằng PyTorch

## Giới thiệu

Dự án COMP1804 sử dụng bộ dữ liệu MNIST để huấn luyện hai mô hình phân loại chữ số từ 0 đến 9:

- MLP: mạng nơ ron nhiều tầng.
- CNN: mạng tích chập.

Ứng dụng được xây dựng bằng Streamlit và hỗ trợ tải ảnh, vẽ chữ số trực tiếp, xem xác suất dự đoán, xem lịch sử và đánh giá mô hình.

## Yêu cầu

- Python 3.10 hoặc mới hơn.
- Hệ điều hành Windows, macOS hoặc Linux.
- Khuyến nghị máy có CPU 64-bit; GPU CUDA là tùy chọn.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Chạy ứng dụng

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Truy cập địa chỉ do Streamlit hiển thị, mặc định là `http://localhost:8501`.

## Huấn luyện mô hình

Huấn luyện CNN:

```powershell
.\.venv\Scripts\python.exe -m src.train --model cnn --epochs 5
```

Huấn luyện MLP:

```powershell
.\.venv\Scripts\python.exe -m src.train --model mlp --epochs 5
```

Checkpoint được lưu tại `models/cnn.pt` và `models/mlp.pt`. Biểu đồ và lịch sử huấn luyện được lưu tại `results/`.

## Đánh giá mô hình

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --model cnn
.\.venv\Scripts\python.exe -m src.evaluate --model mlp
.\.venv\Scripts\python.exe -m src.check_samples
.\.venv\Scripts\python.exe -m src.experiments
```

Thư mục `samples/` chứa các ảnh mẫu tự viết. Tên file phải bắt đầu bằng chữ số đúng, ví dụ `7.png` hoặc `7_tay_minh.png`.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Cấu trúc thư mục

```text
app.py                 Giao diện Streamlit
src/model.py           Kiến trúc MLP và CNN
src/train.py           Huấn luyện và lưu mô hình
src/evaluate.py        Đánh giá tập test MNIST
src/predict.py         Tiền xử lý và dự đoán
src/experiments.py     So sánh cấu hình huấn luyện
src/check_samples.py   Kiểm tra ảnh trong thư mục samples
tests/                 Unit test và test giao diện
models/                File mô hình .pt
results/               Số liệu và biểu đồ
samples/               10 ảnh mẫu 0–9
```

## Kết quả hiện tại

- CNN: **98.94%** accuracy trên 10.000 ảnh test.
- MLP: **97.73%** accuracy trên 10.000 ảnh test.
- 5 cấu hình thí nghiệm được lưu trong `results/experiments.csv`.
- Cấu hình tốt nhất trên tập test: **Adam learning rate 0.001 có tăng cường dữ liệu**, đạt **99.11%** sau 3 epoch.
