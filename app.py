import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown
import json

# === Cấu hình trang ===
st.set_page_config(
    page_title="Phân loại bệnh lá cà chua",
    page_icon="🍅",
    layout="wide",
)

# === CSS tùy chỉnh ===
st.markdown("""
    <style>
        .title {
             font-size: 40px;
    text-align: center;
    color: white;
    background-color: 		#b8e0d2;  /* Màu xanh dịu mắt */
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 30px;
        }
        .result-box {
            text-align: center;
            background-color: #dff0d8;
            color: #155724;
            padding: 20px;
            margin-top: 20px;
            border-radius: 12px;
            font-size: 22px;
            border: 1px solid #c3e6cb;
        }
        .upload-box {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #dee2e6;
            text-align: center;
            margin-bottom: 20px;
        }
        .stButton > button {
            border-radius: 8px;
            background-color: #dc3545;
            color: white;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #c82333;
        }
    </style>
""", unsafe_allow_html=True)

# === Đường dẫn model ===
MODEL_PATH = "plant_disease_model_update.h5"
CLASS_INDEX_PATH = "class_indices.json"
FILE_ID = "1UEheXekm6EakPq8COAKumPoHXQuOsetP"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# === Tải model nếu chưa có ===
if not os.path.exists(MODEL_PATH):
    with st.spinner("⏳ Đang tải mô hình..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

# === Load model ===
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

# === Sidebar ===
with st.sidebar:
    st.image("logo.png", use_container_width=True)  # nếu có ảnh logo
    st.markdown("### 📤 Tải ảnh lá cà chua")
    uploaded_file = st.file_uploader("Chọn ảnh (jpg/png)...", type=["jpg", "jpeg", "png"])
   
        

# === Main Area ===
st.markdown("<div class='title'>🍅 Phân loại bệnh lá cà chua bằng VGG16</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Ảnh đã chọn", use_container_width=True)

        # Tiền xử lý
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    with col2:
        with st.spinner("🔍 Đang dự đoán..."):
            prediction = model.predict(img_array)
            predicted_index = int(np.argmax(prediction))
            predicted_class = index_to_class[predicted_index]
            confidence = float(np.max(prediction)) * 100

        st.markdown(
            f"<div class='result-box'>✅Kết quả: <strong>{predicted_class}</strong><br/>🎯 Độ chính xác: {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )
else:
    st.info("Vui lòng tải ảnh trong sidebar để bắt đầu dự đoán.")
