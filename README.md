# COMP1804 - Nhận diện chữ số viết tay bằng PyTorch

Đồ án sử dụng bộ dữ liệu MNIST để huấn luyện và so sánh hai mô hình phân loại chữ số viết tay:

- **CNN (Convolutional Neural Network)**.
- **MLP (Multilayer Perceptron)**.

Ứng dụng **Streamlit** cho phép tải ảnh hoặc vẽ trực tiếp một chữ số, chọn mô hình, xem xác suất dự đoán và theo dõi lịch sử dự đoán.

## Kết quả hiện tại

Hai checkpoint đã được huấn luyện và lưu trong thư mục `models/`.

| Mô hình | Độ chính xác trên 10.000 ảnh test |
|---|---:|
| CNN | **98.94%** |
| MLP | **97.73%** |

Các kết quả chi tiết gồm confusion matrix, loss, accuracy và ảnh dự đoán sai nằm trong `results/`:

- CNN: `results/metrics.json`, `results/confusion_matrix.png`, `results/wrong_predictions.png`.
- MLP: `results/mlp/metrics.json`, `results/mlp/confusion_matrix.png`, `results/mlp/wrong_predictions.png`.
- So sánh thí nghiệm CNN: `results/experiments.csv`, `results/experiments_summary.png`, `results/experiments_test_accuracy.png`.

## Cấu trúc thư mục

```text
COMP1804-PyTorch-MNIST/
├── app.py                 # Giao diện Streamlit
├── requirements.txt       # Các thư viện cần thiết
├── data/MNIST/raw/        # Tự tải dữ liệu MNIST ở lần chạy đầu
├── models/                # cnn.pt và mlp.pt
├── notebooks/             # Notebook phục vụ phân tích/thực nghiệm
├── results/               # Biểu đồ, metrics và bảng kết quả
├── samples/               # Ảnh chữ số tự viết để kiểm thử
├── src/
│   ├── model.py           # Định nghĩa MLP và CNN
│   ├── train.py           # Huấn luyện mô hình
│   ├── evaluate.py        # Đánh giá trên tập test
│   ├── predict.py         # Tiền xử lý và dự đoán
│   ├── experiments.py     # So sánh cấu hình huấn luyện CNN
│   └── check_samples.py   # Kiểm thử trên ảnh trong samples/
└── tests/                 # Kiểm thử ứng dụng và mô hình
```

## Yêu cầu

- Python 3.10 trở lên.
- Windows, macOS hoặc Linux.
- Có thể sử dụng GPU nếu PyTorch nhận diện được CUDA; nếu không, chương trình sẽ tự dùng CPU.

## Cài đặt

Mở PowerShell tại thư mục dự án rồi tạo môi trường Python ảo:

```powershell
cd D:\COMP1804-PyTorch-MNIST
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn lệnh kích hoạt môi trường, chạy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Dữ liệu MNIST chưa được đưa lên GitHub; lần chạy đầu tiên, các script sẽ tự tải dữ liệu về `data/MNIST/raw/`.

## Chạy ứng dụng

```powershell
streamlit run app.py
```

Sau đó mở địa chỉ Streamlit được hiển thị trong terminal (thường là `http://localhost:8501`).

Ứng dụng gồm 3 tab:

1. **Dự đoán:** tải ảnh hoặc vẽ chữ số, chọn CNN/MLP và xem Top 3 xác suất.
2. **Lịch sử dự đoán:** xem và tải lịch sử ra CSV.
3. **Đánh giá mô hình:** xem độ chính xác, metrics từng chữ số, confusion matrix và biểu đồ huấn luyện.

## Huấn luyện mô hình

Huấn luyện CNN mặc định trong 5 epoch:

```powershell
python -m src.train --model cnn --epochs 5 --batch-size 128 --learning-rate 0.001
```

Huấn luyện MLP:

```powershell
python -m src.train --model mlp --epochs 5 --batch-size 128 --learning-rate 0.001 --output-dir results/mlp
```

Các tùy chọn:

- `--epochs`: số epoch.
- `--batch-size`: kích thước batch.
- `--learning-rate`: learning rate cho Adam.
- `--data-dir`: thư mục dữ liệu, mặc định là `data`.
- `--output-dir`: thư mục lưu lịch sử và biểu đồ, mặc định là `results`.
- `--model-dir`: thư mục lưu checkpoint, mặc định là `models`.
- `--seed`: seed để tạo kết quả lặp lại hơn.

Ví dụ đặt thư mục kết quả riêng cho MLP:

```powershell
python -m src.train --model mlp --epochs 5 --output-dir results/mlp
```

Checkpoint được lưu là `models/cnn.pt` hoặc `models/mlp.pt`.

## Đánh giá mô hình

Đánh giá CNN:

```powershell
python -m src.evaluate --model cnn --output-dir results
```

Đánh giá MLP:

```powershell
python -m src.evaluate --model mlp --output-dir results
```

Lệnh đánh giá tạo `metrics.json`, confusion matrix và danh sách ảnh dự đoán sai.

## Kiểm thử ảnh tự viết

Đặt ảnh có tên bắt đầu bằng chữ số đúng, ví dụ `0.png`, `1_digit.png`, `7.jpg`, trong thư mục `samples/`, rồi chạy:

```powershell
python -m src.check_samples
```

Kết quả được lưu tại `results/samples_result.csv`.

## Chạy thí nghiệm so sánh

Script `experiments.py` huấn luyện CNN với các cấu hình Adam/SGD, learning rate và data augmentation khác nhau trong 3 epoch mỗi cấu hình:

```powershell
python -m src.experiments
```

Kết quả được lưu trong `results/experiments.csv`, `results/experiments_history.csv` và các biểu đồ `experiments_*.png`.

## Chạy kiểm thử

```powershell
pytest -q
```

Các test kiểm tra kiến trúc mô hình, khả năng nạp checkpoint, tiền xử lý ảnh, dự đoán mẫu và khởi động ứng dụng Streamlit.

## Quy trình tiền xử lý ảnh

Ảnh tải lên hoặc được vẽ được chuyển sang thang xám, đảo màu nếu nền sáng, loại bỏ pixel nhiễu, cắt vùng chứa chữ số, căn giữa, resize còn 20 x 20 pixel trong khung 28 x 28 và chuẩn hóa bằng:

- Mean: `0.1307`
- Standard deviation: `0.3081`
