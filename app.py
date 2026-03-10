import streamlit as st
import numpy as np
import imageio.v2 as imageio
import cv2
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

model = load_model("model.keras")

st.set_page_config(page_title="Satellite Image Compression", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>

/* ---------------- BACKGROUND ---------------- */
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

button[title="Fullscreen"]{
display:none !important;
}

button[aria-label="Fullscreen"]{
display:none !important;
}

/* ---------------- TITLES ---------------- */
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

/* ---------------- HEADINGS ---------------- */
h1,h2,h3,h4,h5{
color:white !important;
}

/* ---------------- NORMAL TEXT ---------------- */
p,span,div{
color:white;
}

/* ---------------- FILE UPLOADER ---------------- */
label{
color:white !important;
font-weight:bold;
}

/* uploaded filename */
[data-testid="stFileUploaderFileName"]{
color:white !important;
font-weight:bold;
}

/* uploaded file size */
[data-testid="stFileUploaderFile"] small{
color:#E0E0E0 !important;
}

/* browse button */
.stFileUploader button{
color:black !important;
font-weight:bold;
}

/* ---------------- SPINNER ---------------- */
.stSpinner{
color:white !important;
}

/* ---------------- IMAGE CAPTION ---------------- */
figcaption{
color:white !important;
}

/* ---------------- METRICS ---------------- */
[data-testid="stMetricLabel"]{
color:white !important;
}

[data-testid="stMetricValue"]{
color:#00E5FF !important;
font-size:40px;
font-weight:bold;
}

/* ---------------- DOWNLOAD BUTTON ---------------- */
.stDownloadButton button{
background-color:#00E5FF !important;
color:black !important;
font-weight:bold;
border-radius:8px;
padding:10px 20px;
}

.stDownloadButton button:hover{
background-color:#00c6d7 !important;
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

    with st.spinner("Processing satellite image..."):

        b02 = imageio.imread(b02_file).astype(np.float32)

        b03 = imageio.imread(b03_file).astype(np.float32)

        b04 = imageio.imread(b04_file).astype(np.float32)

        b02 = b02/10000
        b03 = b03/10000
        b04 = b04/10000

        rgb = np.dstack((b04,b03,b02))
        rgb = np.clip(rgb,0,1)

        rgb = cv2.resize(rgb,(2048,2048),interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor((rgb*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(gray,50,150)

        mask = edges > np.mean(edges)

        blur = cv2.GaussianBlur(rgb,(5,5),0)

        semantic = rgb.copy()

        mask3 = np.stack([mask]*3,axis=-1)

        semantic[~mask3] = blur[~mask3]

        patch_size = 128
        patches=[]

        h,w,_ = semantic.shape

        for i in range(0,h-patch_size+1,patch_size):
            for j in range(0,w-patch_size+1,patch_size):
                patches.append(semantic[i:i+patch_size,j:j+patch_size])

        patches = np.array(patches)

        pred = model.predict(patches)

        reconstructed = np.zeros_like(semantic)

        patch_id = 0

        for i in range(0,h-patch_size+1,patch_size):
            for j in range(0,w-patch_size+1,patch_size):

                reconstructed[i:i+patch_size,j:j+patch_size] = pred[patch_id]

                patch_id += 1

        psnr = peak_signal_noise_ratio(semantic,reconstructed,data_range=1)

        ssim = structural_similarity(
            semantic,
            reconstructed,
            channel_axis=-1,
            data_range=1
        )

    # ---------------- RESULTS ----------------
    st.write("### Reconstruction Results")

    c1,c2 = st.columns(2)

    with c1:
        st.image(semantic,caption="Original Image",width="stretch")

    with c2:
        st.image(reconstructed,caption="Reconstructed Image",width="stretch")

    st.write("### Compression Performance")

    m1,m2 = st.columns(2)

    with m1:
        st.metric("PSNR",f"{psnr:.2f}")

    with m2:
        st.metric("SSIM",f"{ssim:.3f}")

    reconstructed_uint8=(reconstructed*255).astype(np.uint8)

    success,buffer=cv2.imencode(".png",reconstructed_uint8)

    if success:
        st.download_button(
            label="Download Reconstructed Image",
            data=buffer.tobytes(),
            file_name="compressed_image.png",
            mime="image/png"
        )