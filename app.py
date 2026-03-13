import streamlit as st
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import cv2
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

# -------- Load Model (cached) --------
@st.cache_resource
def load_autoencoder():
    return load_model("model.keras", compile=False)

model = load_autoencoder()

st.set_page_config(page_title="Satellite Image Compression", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

[data-testid="stImage"] button{
display:none !important;
}

[data-testid="stImageToolbar"]{
display:none !important;
}

.main-title{
font-size:48px;
font-weight:bold;
text-align:center;
color:#00E5FF;
}

.subtitle{
font-size:20px;
text-align:center;
margin-bottom:40px;
color:white;
}

h1,h2,h3,h4,h5{
color:white !important;
}

label{
color:white !important;
font-weight:bold;
}

[data-testid="stMetricValue"]{
color:#00E5FF !important;
font-size:40px;
font-weight:bold;
}

/* Fix Browse Files button text visibility */
.stFileUploader button {
    color: black !important;
    background-color: #00E5FF !important;
    font-weight: bold !important;
    border-radius: 6px !important;
}

.stFileUploader button:hover {
    color: black !important;
    background-color: #00c6d7 !important;
}
            
/* Fix uploaded filename color */
[data-testid="stFileUploaderFileName"] {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Fix uploaded file size color */
[data-testid="stFileUploaderFile"] small {
    color: #E0E0E0 !important;
}

/* Fix file icon color */
[data-testid="stFileUploaderFile"] svg {
    color: #FFFFFF !important;
}

/* Fix Download Button visibility */
.stDownloadButton button {
    background-color:#00E5FF !important;
    color:black !important;
    font-weight:bold !important;
    border-radius:8px !important;
    padding:10px 20px !important;
}

.stDownloadButton button:hover {
    background-color:#00c6d7 !important;
    color:black !important;
}           
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">Satellite Image Compression System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Semantic-Aware CNN Autoencoder for High-Resolution Satellite Imagery</p>', unsafe_allow_html=True)

st.write("### Upload Sentinel-2 Spectral Bands")

# ---------------- FILE UPLOAD ----------------
col1, col2, col3 = st.columns(3)

with col1:
    b02_file = st.file_uploader("Upload B02 (Blue)", type=["jp2"])

with col2:
    b03_file = st.file_uploader("Upload B03 (Green)", type=["jp2"])

with col3:
    b04_file = st.file_uploader("Upload B04 (Red)", type=["jp2"])


# ---------------- PROCESS ----------------
if b02_file and b03_file and b04_file:

    with st.spinner("Running semantic-aware compression... please wait"):

        # -------- Load JP2 bands --------
        with MemoryFile(b02_file.read()) as memfile:
            with memfile.open() as src:
                b02 = src.read(1).astype(np.float32)

        with MemoryFile(b03_file.read()) as memfile:
            with memfile.open() as src:
                b03 = src.read(1).astype(np.float32)

        with MemoryFile(b04_file.read()) as memfile:
            with memfile.open() as src:
                b04 = src.read(1).astype(np.float32)

        # -------- Normalize --------
        b02 = b02 / 10000
        b03 = b03 / 10000
        b04 = b04 / 10000

        # -------- RGB creation --------
        rgb = np.dstack((b04, b03, b02))
        rgb = np.clip(rgb, 0, 1)

        # resize for faster inference
        rgb = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA)

        # -------- Semantic preprocessing --------
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        mask = edges > np.mean(edges)
        blur = cv2.GaussianBlur(rgb, (5, 5), 0)

        semantic = rgb.copy()
        mask3 = np.stack([mask] * 3, axis=-1)

        semantic[~mask3] = blur[~mask3]

        # -------- Patch extraction --------
        patch_size = 128
        patches = []

        h, w, _ = semantic.shape

        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                patches.append(semantic[i:i + patch_size, j:j + patch_size])

        patches = np.array(patches)

        # -------- Model inference --------
        pred = model.predict(patches)

        # -------- Reconstruction --------
        reconstructed = np.zeros_like(semantic)

        patch_id = 0

        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):

                reconstructed[i:i + patch_size, j:j + patch_size] = pred[patch_id]
                patch_id += 1

        # -------- Metrics --------
        psnr = peak_signal_noise_ratio(semantic, reconstructed, data_range=1)

        ssim = structural_similarity(
            semantic,
            reconstructed,
            channel_axis=-1,
            data_range=1
        )

    # ---------------- RESULTS ----------------
    st.write("### Reconstruction Results")

    c1, c2 = st.columns(2)

    with c1:
        st.image(semantic, caption="Original Image", width="stretch")

    with c2:
        st.image(reconstructed, caption="Reconstructed Image", width="stretch")

    st.write("### Compression Performance")

    m1, m2 = st.columns(2)

    with m1:
        st.metric("PSNR", f"{psnr:.2f}")

    with m2:
        st.metric("SSIM", f"{ssim:.3f}")

    # -------- Download --------
    reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)
    success, buffer = cv2.imencode(".png", reconstructed_uint8)

    if success:
        st.download_button(
            label="Download Reconstructed Image",
            data=buffer.tobytes(),
            file_name="compressed_image.png",
            mime="image/png"
        )

else:
    st.info("Please upload B02, B03, and B04 Sentinel-2 bands to begin.")