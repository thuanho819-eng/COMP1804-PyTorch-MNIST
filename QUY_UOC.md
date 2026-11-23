# Quy ước dự án

- Ảnh đầu vào cần là một chữ số từ 0 đến 9.
- Không gian xử lý nội bộ luôn có kích thước 28x28 pixel.
- Ảnh grayscale được chuẩn hóa bằng mean `0.1307` và standard deviation `0.3081`.
- Tên checkpoint bắt buộc: `models/cnn.pt` và `models/mlp.pt`.
- Tên kết quả CNN nằm trực tiếp trong `results/`; kết quả MLP nằm trong `results/mlp/`.
- Tên ảnh mẫu bắt đầu bằng chữ số đúng, ví dụ `3.png` hoặc `3_tay_minh.jpg`.
- Lịch sử dự đoán được lưu tại `lich_su_du_doan.csv` và không commit file này.
- Trước khi nộp code: chạy `pytest -q`, kiểm tra app, kiểm tra README và dọn các file cache.
