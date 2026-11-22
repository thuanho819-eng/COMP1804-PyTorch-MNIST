import os

import pandas as pd
from PIL import Image

from src.predict import PROJECT_DIR, load_model, predict, preprocess

SAMPLES_DIR = os.path.join(PROJECT_DIR, "samples")
RESULT_FILE = os.path.join(PROJECT_DIR, "results", "samples_result.csv")


def main():
    # Lấy danh sách ảnh có tên bắt đầu bằng một chữ số
    files = []
    for name in sorted(os.listdir(SAMPLES_DIR)):
        if name.lower().endswith((".png", ".jpg", ".jpeg")) and name[0].isdigit():
            files.append(name)

    if len(files) == 0:
        print("Thư mục samples chưa có ảnh nào (tên file phải bắt đầu bằng chữ số, ví dụ 7_a.png)")
        return

    rows = []
    for model_name in ["CNN", "MLP"]:
        model, trained = load_model(model_name)
        if not trained:
            print(f"Chưa có file mô hình {model_name}, bỏ qua")
            continue

        for name in files:
            image = Image.open(os.path.join(SAMPLES_DIR, name))
            probs = predict(model, preprocess(image))
            digit = probs.index(max(probs))
            rows.append({
                "Mô hình": model_name,
                "Tên file": name,
                "Chữ số đúng": int(name[0]),
                "Dự đoán": digit,
                "Độ tin cậy (%)": round(max(probs) * 100, 1),
                "Đúng": digit == int(name[0]),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Tỷ lệ đúng của từng mô hình
    print()
    for model_name, group in df.groupby("Mô hình"):
        correct = group["Đúng"].sum()
        print(f"{model_name}: đúng {correct}/{len(group)} ảnh ({correct / len(group) * 100:.1f}%)")

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu kết quả vào {RESULT_FILE}")


if __name__ == "__main__":
    main()
