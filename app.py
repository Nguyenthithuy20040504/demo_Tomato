import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown
import json

# ============ Cấu hình trang ============
st.set_page_config(
    page_title="🍅 Phân loại bệnh lá cà chua",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ CSS =============
st.markdown("""
    <style>
        body {
            background-color: #f4f6f7;
        }
        .main-title {
            font-size: 40px;
            color: #2d6a4f;
            text-align: center;
            margin-bottom: 0px;
        }
        .subtitle {
            font-size: 18px;
            color: #6c757d;
            text-align: center;
            margin-bottom: 40px;
        }
        .result-box {
            background-color: #d1e7dd;
            color: #0f5132;
            padding: 25px;
            border-radius: 15px;
            font-size: 22px;
            text-align: center;
            margin-top: 20px;
            border: 1px solid #badbcc;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: gray;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

# ============ Tải mô hình =============
MODEL_PATH = "plant_disease_model_update.h5"
CLASS_INDEX_PATH = "class_indices.json"
FILE_ID = "1qK6cnyVpIwYfuzhC-qtCdisJPmyfaXgs"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    with st.spinner("⏳ Đang tải mô hình..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

# ============ Tiêu đề =============
st.markdown("<div class='main-title'>🍅 Phân loại Bệnh Lá Cà Chua</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Nhận diện các bệnh phổ biến trên lá cà chua bằng mô hình học sâu</div>", unsafe_allow_html=True)

# ============ Sidebar =============
with st.sidebar:
    st.image("logo.png", use_container_width=True)  # nếu có logo
    st.markdown("## 📥 Tải ảnh lá cà chua")
    uploaded_file = st.file_uploader("Chọn ảnh (JPG/PNG)", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    st.markdown("📌 Định dạng hỗ trợ: .jpg, .jpeg, .png")
    st.markdown("🧠 Mô hình: EfficientNet (cập nhật)")
    st.markdown("👨‍💻 Dành cho mục đích nghiên cứu và giáo dục.")

# ============ Khu vực chính ============
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Ảnh đã tải lên", use_container_width=True)

    # Tiền xử lý ảnh
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Đang phân tích ảnh..."):
        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        predicted_class = index_to_class[predicted_index]
        confidence = float(np.max(prediction)) * 100

    st.markdown(
        f"<div class='result-box'>✅ <strong>{predicted_class}</strong><br/>🎯 Độ chính xác: {confidence:.2f}%</div>",
        unsafe_allow_html=True
    )

    # Mở rộng: thêm mô tả bệnh nếu muốn
    disease_info = {
        "Tomato___Bacterial_spot": "Bệnh đốm vi khuẩn - gây ra các đốm đen tròn nhỏ, dễ lây lan qua nước.",
        "Tomato___Early_blight": "Bệnh mốc sương sớm - các đốm nâu lớn, làm lá héo nhanh.",
        "Tomato___Leaf_Mold": "Bệnh mốc lá - mốc màu xanh ô liu, thường ở mặt dưới lá.",
        # Thêm các bệnh khác nếu cần
    }
    if predicted_class in disease_info:
        st.info(f"📝 **Thông tin về bệnh:** {disease_info[predicted_class]}")

else:
    st.info("📤 Vui lòng tải lên một ảnh trong thanh bên để bắt đầu.")

# ============ Footer ============
st.markdown("<div class='footer'>🌱 Ứng dụng demo - Được phát triển bởi Nhóm AI Nông Nghiệp - 2025</div>", unsafe_allow_html=True)
