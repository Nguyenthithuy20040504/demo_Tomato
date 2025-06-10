import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown
import json

# ============ Cấu hình trang ============ #
st.set_page_config(
    page_title="🍅 Phân loại bệnh lá cà chua",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ CSS giao diện ============ #
st.markdown("""
    <style>
        .main-title {
            font-size: 60px;
            color: #2d6a4f;
            text-align: center;
        }
        .subtitle {
            font-size: 20px;
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
            font-size: 20px;
            color: gray;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

# ============ Tải mô hình ============ #
MODEL_PATH = "plant_disease_model_update.h5"
CLASS_INDEX_PATH = "class_indices.json"

# Tải mô hình từ Google Drive nếu chưa có
FILE_ID = "1qK6cnyVpIwYfuzhC-qtCdisJPmyfaXgs"
URL = f"https://drive.google.com/uc?id={FILE_ID}"
if not os.path.exists(MODEL_PATH):
    with st.spinner("⏳ Đang tải mô hình phân loại..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

# Load feature extractor (ResNet50 base)
@st.cache_resource
def load_feature_extractor():
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    global_pooling = tf.keras.layers.GlobalAveragePooling2D()
    return base_model, global_pooling

feature_extractor, global_pooling = load_feature_extractor()

# Load classifier model
@st.cache_resource
def load_classifier():
    return tf.keras.models.load_model(MODEL_PATH)

classifier_model = load_classifier()


# Load class index
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

# ============ Tiêu đề ============ #
st.markdown("<div class='main-title'>🍅 Phân loại Bệnh Lá Cà Chua</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Nhận diện các bệnh phổ biến trên lá cà chua bằng mô hình học sâu</div>", unsafe_allow_html=True)

# ============ Sidebar ============ #
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("## 📥 Tải ảnh lá cà chua")

    # Khởi tạo biến trạng thái
    if "use_camera" not in st.session_state:
        st.session_state.use_camera = False
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "camera_image" not in st.session_state:
        st.session_state.camera_image = None

    # Hai nút chọn phương thức
    method = st.radio("Chọn phương thức", ["📂 Tải ảnh", "📸 Chụp ảnh"])

    if method == "📂 Tải ảnh":
        # Nếu chọn tải ảnh, reset camera
        st.session_state.use_camera = False
        uploaded_file = st.file_uploader("Chọn ảnh (JPG/PNG)", type=["jpg", "jpeg", "png"])
        st.session_state.uploaded_file = uploaded_file
        st.session_state.camera_image = None
    else:
        # Nếu chọn camera, reset ảnh upload
        st.session_state.use_camera = True
        camera_image = st.camera_input("Chụp ảnh lá cà chua")
        st.session_state.camera_image = camera_image
        st.session_state.uploaded_file = None

    st.markdown("---")
    st.markdown("📌 Định dạng hỗ trợ: .jpg, .jpeg, .png")
    st.markdown("🧠 Mô hình: ResNet50 + Classifier")
    st.markdown("👨‍💻 Dành cho mục đích nghiên cứu và giáo dục.")


# ============ Dữ liệu mô tả bệnh (nếu có) ============ #
disease_info = {
    # Ví dụ: "Tomato___Late_blight": "Là một bệnh nấm gây ra các đốm nâu trên lá..."
}

# ============ Xử lý ảnh & Dự đoán ============ #
image_source = st.session_state.camera_image if st.session_state.use_camera else st.session_state.uploaded_file

if image_source is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        img = Image.open(image_source)
        st.image(img, caption="🖼️ Ảnh được sử dụng", use_container_width=True)

    with col2:
        img = img.convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        with st.spinner("🔍 Đang phân tích ảnh..."):
            features = feature_extractor.predict(img_array)
            features = global_pooling(features)
            prediction = classifier_model.predict(features)
            predicted_index = int(np.argmax(prediction))
            predicted_class = index_to_class[predicted_index]
            confidence = float(np.max(prediction)) * 100

        st.markdown(
            f"<div class='result-box'>✅ Kết quả dự đoán: <strong>{predicted_class}</strong><br/>🎯 Độ chính xác: {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )

        if predicted_class in disease_info:
            st.info(f"📝 **Thông tin về bệnh:**\n{disease_info[predicted_class]}")
else:
    st.info("📤 Vui lòng tải lên ảnh hoặc chụp ảnh để bắt đầu.")


# ============ Footer ============ #
st.markdown("<div class='footer'>🌱 Ứng dụng demo - Được phát triển bởi Nhóm 6 AI - 2025</div>", unsafe_allow_html=True)
