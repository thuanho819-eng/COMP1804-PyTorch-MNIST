import datetime
import json
import os

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.predict import load_model, predict, preprocess

APP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(APP_DIR, "results")

HISTORY_FILE = os.path.join(APP_DIR, "lich_su_du_doan.csv")


def save_history():
    """Ghi toàn bộ lịch sử ra file CSV để không mất khi tắt app."""
    df = pd.DataFrame(st.session_state["history"])
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


def result_path(model_name, file_name):
    """Đường dẫn file kết quả: CNN nằm trong results/, MLP nằm trong results/mlp/."""
    if model_name == "CNN":
        return os.path.join(RESULTS_DIR, file_name)
    return os.path.join(RESULTS_DIR, "mlp", file_name)


def load_metrics(model_name):
    """Đọc file metrics.json do evaluate.py tạo ra, chưa có thì trả về None."""
    path = result_path(model_name, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


st.title("Nhận diện chữ số viết tay")
st.write("Đồ án COMP1804 - Mô hình PyTorch huấn luyện trên bộ dữ liệu MNIST")

if "models" not in st.session_state:
    st.session_state["models"] = {
        "CNN": load_model("CNN"),
        "MLP": load_model("MLP"),
    }
if "history" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        old = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
        st.session_state["history"] = old.to_dict("records")
    else:
        st.session_state["history"] = []

tab_predict, tab_history, tab_eval = st.tabs(
    ["Dự đoán", "Lịch sử dự đoán", "Đánh giá mô hình"]
)

with tab_predict:
    model_name = st.selectbox("Chọn mô hình", ["CNN", "MLP"])
    model, trained = st.session_state["models"][model_name]
    if not trained:
        st.warning("Chưa có file mô hình trong thư mục models, kết quả chỉ để thử giao diện")

    input_mode = st.radio("Cách nhập chữ số", ["Tải ảnh lên", "Vẽ trực tiếp"], horizontal=True)

    image = None
    source = ""
    if input_mode == "Tải ảnh lên":
        uploaded = st.file_uploader("Chọn ảnh chữ số (png, jpg)", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            try:
                image = Image.open(uploaded)
                source = uploaded.name
            except Exception:
                st.error("Không đọc được ảnh này, hãy chọn file khác")
        else:
            st.info("Hãy tải lên ảnh một chữ số từ 0 đến 9")
    else:
        st.write("Vẽ **một** chữ số to, nằm giữa khung trắng. Bấm biểu tượng thùng rác để xóa.")
        canvas = st_canvas(
            stroke_width=18,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            return_image_data=True,
            key="canvas",
        )
        if canvas.image_data is not None:
            image = Image.fromarray(canvas.image_data.astype("uint8")).convert("RGB")
            source = "Vẽ trực tiếp"

    if image is not None:
        image28 = preprocess(image)

        if image28.getextrema()[1] == 0:
            st.info("Chưa có nét chữ nào, hãy vẽ một chữ số")
        else:
            # Hiển thị ảnh gốc và ảnh 28x28 sau khi xử lý
            col1, col2 = st.columns(2)
            col1.image(image, caption="Ảnh gốc", width=200)
            col2.image(image28.resize((200, 200), Image.NEAREST),
                       caption="Ảnh 28x28 đưa vào mô hình", width=200)

            if st.button("Dự đoán"):
                probs = predict(model, image28)
                digit = probs.index(max(probs))
                confidence = max(probs) * 100

                st.success(f"Chữ số dự đoán: {digit}")
                st.metric("Độ tin cậy", f"{confidence:.1f} %")

                # Top 3 chữ số có xác suất cao nhất
                top3 = sorted(range(10), key=lambda i: probs[i], reverse=True)[:3]
                st.write("Top 3:", ", ".join(f"{i} ({probs[i] * 100:.1f}%)" for i in top3))

                # Biểu đồ xác suất của 10 chữ số
                chart = pd.DataFrame({
                    "Chữ số": [str(i) for i in range(10)],
                    "Xác suất": probs,
                })
                st.bar_chart(chart, x="Chữ số", y="Xác suất")

                st.session_state["history"].append({
                    "Thời gian": datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
                    "Tên file": source,
                    "Mô hình": model_name,
                    "Kết quả": digit,
                    "Độ tin cậy (%)": round(confidence, 1),
                })
                save_history()

with tab_history:
    if len(st.session_state["history"]) == 0:
        st.write("Chưa có lần dự đoán nào")
    else:
        df = pd.DataFrame(st.session_state["history"])
        st.dataframe(df, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Tải CSV", csv_bytes,
                           file_name="lich_su_du_doan.csv", mime="text/csv")

        if st.button("Xóa lịch sử"):
            st.session_state["history"] = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()
with tab_eval:
    st.subheader("Độ chính xác trên tập test MNIST (10.000 ảnh)")
    col1, col2 = st.columns(2)
    for col, name in [(col1, "CNN"), (col2, "MLP")]:
        metrics = load_metrics(name)
        if metrics is None:
            col.write(f"Chưa có kết quả của {name}, hãy chạy src/evaluate.py")
        else:
            col.metric(name, f"{metrics['test_accuracy'] * 100:.2f} %")

    eval_model = st.radio("Xem chi tiết mô hình", ["CNN", "MLP"], horizontal=True)
    metrics = load_metrics(eval_model)

    if metrics is not None:
        rows = []
        for digit, value in metrics["per_digit"].items():
            rows.append({
                "Chữ số": digit,
                "Số ảnh": value["support"],
                "Đoán đúng": value["correct"],
                "Độ chính xác (%)": round(value["accuracy"] * 100, 2),
            })
        df_digit = pd.DataFrame(rows)
        st.dataframe(df_digit, hide_index=True)
        st.bar_chart(df_digit, x="Chữ số", y="Độ chính xác (%)")

        worst = df_digit.loc[df_digit["Độ chính xác (%)"].idxmin()]
        st.write(f"Chữ số khó nhận diện nhất: **{worst['Chữ số']}** "
                 f"({worst['Độ chính xác (%)']}%)")

    for title, file_name in [
        ("Loss qua từng epoch", "loss.png"),
        ("Accuracy qua từng epoch", "accuracy.png"),
        ("Confusion matrix", "confusion_matrix.png"),
        ("Một số ảnh bị đoán sai", "wrong_predictions.png"),
    ]:
        path = result_path(eval_model, file_name)
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(path, width=600)

    experiments_file = os.path.join(RESULTS_DIR, "experiments.csv")
    if os.path.exists(experiments_file):
        st.subheader("Thí nghiệm so sánh cấu hình huấn luyện CNN")
        st.dataframe(pd.read_csv(experiments_file, encoding="utf-8-sig"), hide_index=True)
        summary_chart = os.path.join(RESULTS_DIR, "experiments_summary.png")
        if os.path.exists(summary_chart):
            st.image(summary_chart, width=650)

    samples_file = os.path.join(RESULTS_DIR, "samples_result.csv")
    if os.path.exists(samples_file):
        st.subheader("Kết quả trên ảnh tự viết (thư mục samples)")
        df_samples = pd.read_csv(samples_file, encoding="utf-8-sig")
        summary = df_samples.groupby("Mô hình")["Đúng"].agg(["sum", "count"]).reset_index()
        summary.columns = ["Mô hình", "Số ảnh đúng", "Tổng số ảnh"]
        summary["Tỷ lệ đúng (%)"] = round(summary["Số ảnh đúng"] / summary["Tổng số ảnh"] * 100, 1)
        st.dataframe(summary, hide_index=True)
