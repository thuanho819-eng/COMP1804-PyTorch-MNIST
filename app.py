import datetime
import os    

import pandas as pd
import streamlit as st
from PIL import Image

from src.predict import load_model, predict, preprocess

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lich_su_du_doan.csv")


def save_history():
    """Ghi toàn bộ lịch sử ra file CSV để không mất khi tắt app."""
    df = pd.DataFrame(st.session_state["history"])
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")

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

model_name = st.selectbox("Chọn mô hình", ["CNN", "MLP"])
model, trained = st.session_state["models"][model_name]
if not trained:
    st.warning("Chưa có file mô hình trong thư mục models, kết quả chỉ để thử giao diện")

uploaded = st.file_uploader("Chọn ảnh chữ số (png, jpg)", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image28 = None
    try:
        image = Image.open(uploaded)
        image28 = preprocess(image)
    except Exception:
        st.error("Không đọc được ảnh này, hãy chọn file khác")

    if image28 is not None:
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

            top3 = sorted(range(10), key=lambda i: probs[i], reverse=True)[:3]
            st.write("Top 3:", ", ".join(f"{i} ({probs[i] * 100:.1f}%)" for i in top3))

            chart = pd.DataFrame({
                "Chữ số": [str(i) for i in range(10)],
                "Xác suất": probs,
            })
            st.bar_chart(chart, x="Chữ số", y="Xác suất")

            st.session_state["history"].append({
                "Thời gian": datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
                "Tên file": uploaded.name,
                "Mô hình": model_name,
                "Kết quả": digit,
                "Độ tin cậy (%)": round(confidence, 1),
            })
            save_history() 
else:
    st.info("Hãy tải lên ảnh một chữ số từ 0 đến 9")

st.subheader("Lịch sử dự đoán")
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
