import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown
import json


# === Thiết lập cấu hình trang ===
st.set_page_config(
    page_title="Phân loại bệnh lá cà chua",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# === CSS tùy chỉnh cho giao diện đẹp hơn ===
st.markdown("""
    <style>
        body {
             background-color: #e6f2ff;
        }
        .stApp {
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #1e3d59;
        }
        .upload-section {
            text-align: center;
            margin-top: 20px;
        }
        .result {
            text-align: center;
            font-size: 22px;
            margin-top: 30px;
            padding: 15px;
            border-radius: 10px;
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .image-container img {
            border-radius: 10px;
            border: 2px solid #ddd;
        }
    </style>
""", unsafe_allow_html=True)

# === Đường dẫn mô hình và class ===
MODEL_PATH = "plant_disease_model_update.h5"
CLASS_INDEX_PATH = "class_indices.json"

FILE_ID = "1UEheXekm6EakPq8COAKumPoHXQuOsetP"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# === Tải mô hình nếu chưa có ===
if not os.path.exists(MODEL_PATH):
    with st.spinner("⏳ Đang tải mô hình từ Google Drive..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

# === Load model và class indices ===

    model = tf.keras.models.load_model('MODEL_PATH')
with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

# === Giao diện chính ===
st.markdown("<div class='title'>🌿 Phân loại bệnh lá cà chua bằng VGG16</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Vui lòng chọn ảnh lá cà chua (jpg/png)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # Hiển thị ảnh
    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
    st.image(img, caption="🖼️ Ảnh đã chọn", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Tiền xử lý
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán
    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    predicted_class = index_to_class[predicted_index]
    confidence = float(np.max(prediction)) * 100

    # Hiển thị kết quả
    st.markdown(f"<div class='result'>✅ <strong>{predicted_class}</strong> ({confidence:.2f}%)</div>", unsafe_allow_html=True)
