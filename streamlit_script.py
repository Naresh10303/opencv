import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io


def apply_pattern_to_flag_cv(flag_img, pattern_img):
    # Resize pattern to match flag dimensions
    pattern_resized = cv2.resize(pattern_img, (flag_img.shape[1], flag_img.shape[0]))
    rows, cols = flag_img.shape[:2]

    # 1. Create a mask for the flag area (assuming white flag on white bg)
    gray = cv2.cvtColor(flag_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. Simulate waving
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    amplitude = 15
    frequency = 2
    phase = np.pi / 3
    offset_x = amplitude * np.sin(2 * np.pi * map_y / rows * frequency + phase)
    offset_y = amplitude * 0.2 * np.sin(2 * np.pi * map_x / cols * frequency + phase)
    map_x_warped = (map_x + offset_x).astype(np.float32)
    map_y_warped = (map_y + offset_y).astype(np.float32)
    warped_pattern = cv2.remap(pattern_resized, map_x_warped, map_y_warped, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 3. Blend where the mask is active
    mask_3ch = cv2.merge([mask, mask, mask])
    mask_norm = mask_3ch / 255.0
    output_img = (warped_pattern * mask_norm + flag_img * (1 - mask_norm)).astype(np.uint8)

    return output_img


# --- Streamlit UI ---
st.set_page_config(page_title="Flag Pattern Waver", layout="centered")
st.title("🧵 Flag Pattern Waver")
st.write("Upload a white flag image and a pattern image to create a waving effect.")

uploaded_flag = st.file_uploader("Upload White Flag Image", type=["jpg", "jpeg", "png"])
uploaded_pattern = st.file_uploader("Upload Pattern Image (e.g., American flag)", type=["jpg", "jpeg", "png"])

if uploaded_flag and uploaded_pattern:
    # Convert images to OpenCV format
    flag_pil = Image.open(uploaded_flag).convert("RGB")
    pattern_pil = Image.open(uploaded_pattern).convert("RGB")
    flag_cv = cv2.cvtColor(np.array(flag_pil), cv2.COLOR_RGB2BGR)
    pattern_cv = cv2.cvtColor(np.array(pattern_pil), cv2.COLOR_RGB2BGR)

    with st.spinner("Processing the waving flag..."):
        result_cv = apply_pattern_to_flag_cv(flag_cv, pattern_cv)

    # Show result
    result_rgb = cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption="Waving Flag", use_column_width=True)

    # Convert result to PIL and BytesIO for download
    result_pil = Image.fromarray(result_rgb)
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG")
    buf.seek(0)

    st.download_button(
        label="📥 Download Waving Flag Image",
        data=buf,
        file_name="waving_flag.jpg",
        mime="image/jpeg"
    )