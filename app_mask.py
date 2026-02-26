import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

st.set_page_config(page_title="Mask Detection App", layout="centered")

st.title("😷 Mask Detection App ")
st.write("Upload ảnh và hệ thống sẽ nhận diện người có đeo khẩu trang hay không.")


@st.cache_resource
def load_model():
    return YOLO("best (1).pt")  

model = load_model()


uploaded_file = st.file_uploader("Tải ảnh lên...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.image(img, caption="Ảnh đã upload", width=800)


    st.subheader("🔍 Đang nhận diện...")
    results = model.predict(img_np, verbose=False)


    annotated_img = results[0].plot()  

    st.image(annotated_img, caption="Kết quả nhận diện", width=800)


    st.subheader("📌 Chi tiết detection:")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls]
        st.write(f"- **{cls_name}** – độ tin cậy: `{conf:.2f}`")