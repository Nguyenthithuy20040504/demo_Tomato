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
            font-size: 60px;
            color: #2d6a4f;
            text-align: center;
            margin-bottom: 0px;
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

# ============ Tải mô hình ============
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

# ============ Tiêu đề ============
st.markdown("<div class='main-title'>🍅 Phân loại Bệnh Lá Cà Chua</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Nhận diện các bệnh phổ biến trên lá cà chua bằng mô hình học sâu</div>", unsafe_allow_html=True)

# ============ Sidebar ============
with st.sidebar:
    st.image("logo.png", use_container_width=True)  # nếu có logo
    st.markdown("## 📥 Tải ảnh lá cà chua")
    uploaded_file = st.file_uploader("Chọn ảnh (JPG/PNG)", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    st.markdown("📌 Định dạng hỗ trợ: .jpg, .jpeg, .png")
    st.markdown("🧠 Mô hình:  ResNet50 kết hợp classifier ")
    st.markdown("👨‍💻 Dành cho mục đích nghiên cứu và giáo dục.")

# ============ Thông tin bệnh ============
disease_info = {
    "Tomato___Bacterial_spot": """Tomato___Bacterial_spot  
**Tên tiếng Việt**: Bệnh đốm vi khuẩn  
**Nguyên nhân**: Vi khuẩn *Xanthomonas campestris*  
**Triệu chứng**:  
- Đốm tròn nhỏ màu nâu hoặc đen trên lá, thân và quả.  
- Lá có thể cháy viền và rụng sớm.  
**Xử lý**:  
- Không trồng cây bị bệnh, sử dụng hạt giống sạch.  
- Phun thuốc gốc đồng định kỳ.""",

    "Tomato___Early_blight": """Tomato___Early_blight  
**Tên tiếng Việt**: Bệnh mốc sương sớm  
**Nguyên nhân**: Nấm *Alternaria solani*  
**Triệu chứng**:  
- Đốm tròn màu nâu có vòng tròn đồng tâm.  
- Lá vàng, rụng từ dưới lên.  
**Xử lý**:  
- Luân canh cây trồng, cắt bỏ lá bệnh.  
- Phun thuốc trị nấm như Mancozeb.""",

    "Tomato___Late_blight": """Tomato___Late_blight  
**Tên tiếng Việt**: Bệnh mốc sương muộn  
**Nguyên nhân**: Nấm *Phytophthora infestans*  
**Triệu chứng**:  
- Vết nước trên lá lan rộng, có mốc trắng dưới mặt lá.  
- Quả bị thối nhũn, cây nhanh chết.  
**Xử lý**:  
- Tiêu hủy cây bệnh, không tưới đẫm lá.  
- Phun thuốc như Metalaxyl.""",

    "Tomato___Leaf_Mold": """Tomato___Leaf_Mold  
**Tên tiếng Việt**: Bệnh mốc lá  
**Nguyên nhân**: Nấm *Fulvia fulva*  
**Triệu chứng**:  
- Mặt trên lá có đốm vàng, mặt dưới có lớp mốc ô liu.  
- Lá héo nhanh khi trời ẩm.  
**Xử lý**:  
- Cắt tỉa lá bệnh, giữ thông thoáng.  
- Phun thuốc gốc đồng hoặc Chlorothalonil.""",

    "Tomato___Septoria_leaf_spot": """Tomato___Septoria_leaf_spot  
**Tên tiếng Việt**: Bệnh đốm lá Septoria  
**Nguyên nhân**: Nấm *Septoria lycopersici*  
**Triệu chứng**:  
- Đốm tròn nhỏ, màu xám nâu, có viền sẫm.  
- Thường xuất hiện ở lá dưới trước.  
**Xử lý**:  
- Cắt tỉa lá bệnh, tăng độ thông thoáng.  
- Phun Mancozeb hoặc Chlorothalonil.""",

    "Tomato___Spider_mites Two-spotted_spider_mite": """Tomato___Spider_mites Two-spotted_spider_mite  
**Tên tiếng Việt**: Bệnh nhện đỏ hai chấm  
**Nguyên nhân**: Nhện *Tetranychus urticae*  
**Triệu chứng**:  
- Chấm vàng li ti trên lá, lá khô và rụng.  
- Mạng tơ mỏng dưới mặt lá.  
**Xử lý**:  
- Phun thuốc trừ nhện như Abamectin.  
- Duy trì độ ẩm đất ổn định.""",

    "Tomato___Target_Spot": """Tomato___Target_Spot  
**Tên tiếng Việt**: Bệnh đốm mục tiêu  
**Nguyên nhân**: Nấm *Corynespora cassiicola*  
**Triệu chứng**:  
- Đốm tròn lớn, có vòng tròn đồng tâm.  
- Thường lan rộng khi ẩm độ cao.  
**Xử lý**:  
- Loại bỏ lá bệnh, không tưới vào lá.  
- Phun thuốc nấm phổ rộng.""",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": """Tomato___Tomato_Yellow_Leaf_Curl_Virus  
**Tên tiếng Việt**: Virus vàng xoăn lá cà chua  
**Nguyên nhân**: Virus TYLCV, lây truyền qua bọ phấn trắng  
**Triệu chứng**:  
- Lá xoăn, cuộn vào trong.  
- Cây còi cọc, chậm lớn, không ra hoa.  
**Xử lý**:  
- Phòng bọ phấn bằng lưới chắn, bẫy vàng.  
- Sử dụng giống kháng virus.  
- Loại bỏ cây bệnh sớm.""",

    "Tomato___Tomato_mosaic_virus": """Tomato___Tomato_mosaic_virus  
**Tên tiếng Việt**: Virus khảm cà chua  
**Nguyên nhân**: *Tomato mosaic virus* (ToMV)  
**Triệu chứng**:  
- Lá loang lổ màu xanh nhạt – đậm.  
- Biến dạng lá, cây kém phát triển.  
**Xử lý**:  
- Không có thuốc trị, cần tiêu hủy cây bệnh.  
- Dùng giống kháng, khử trùng dụng cụ trồng.""",

    "Tomato___healthy": """Tomato___healthy  
**Tên tiếng Việt**: Cây khỏe mạnh  
**Mô tả**:  
- Không có dấu hiệu bệnh.  
- Lá xanh đều, cây phát triển bình thường.  
- Tiếp tục chăm sóc đúng cách để duy trì sức khỏe."""
}

# ============ Khu vực chính - Chia 2 cột ============
if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Ảnh đã tải lên", use_container_width=True)

    with col2:
        # Tiền xử lý ảnh
        img = img.convert("RGB") 
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array) 

        with st.spinner("🔍 Đang phân tích ảnh..."):
            prediction = model.predict(img_array)
            predicted_index = int(np.argmax(prediction))
            predicted_class = index_to_class[predicted_index]
            confidence = float(np.max(prediction)) * 100

        # Hiển thị kết quả dự đoán
        st.markdown(
            f"<div class='result-box'>✅Kết quả dự đoán: <strong>{predicted_class}</strong><br/>🎯 Độ chính xác: {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )

        # Hiển thị thông tin bệnh nếu có
        if predicted_class in disease_info:
            st.info(f"📝 **Thông tin về bệnh:**\n{disease_info[predicted_class]}")
else:
    st.info("📤 Vui lòng tải lên một ảnh trong thanh bên để bắt đầu.")

# ============ Footer ============
st.markdown("<div class='footer'>🌱 Ứng dụng demo - Được phát triển bởi Nhóm 6 AI - 2025 </div>", unsafe_allow_html=True)
