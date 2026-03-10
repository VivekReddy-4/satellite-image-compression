# Semantic-Aware Deep Learning Based Compression of High-Resolution Satellite Images

## Project Overview

High-resolution satellite imagery generates massive volumes of data, creating challenges for efficient storage and transmission. Traditional compression methods such as JPEG and JPEG2000 often struggle to preserve important spatial structures present in satellite images.

This project proposes a deep learning based compression system using a Convolutional Autoencoder combined with semantic-aware preprocessing. The approach focuses on preserving important regions while smoothing less significant areas, enabling higher compression efficiency without significantly degrading image quality.

## Key Idea

The system identifies important regions using edge detection and applies region-aware preprocessing before compressing image patches using a CNN-based autoencoder. This allows the model to achieve higher compression while maintaining structural and visual integrity of satellite imagery.

## Dataset

The project uses real Sentinel-2 satellite imagery in JP2 format.

Bands used:
B02 (Blue)  
B03 (Green)  
B04 (Red)

These bands are combined to generate RGB satellite images.

Note: Due to the large size of satellite datasets, the images are not included in this repository. The dataset was accessed from Google Drive during model training.

## Methodology

The proposed system follows the pipeline below:

1. Satellite Image Acquisition
2. RGB Image Generation from Sentinel-2 Bands
3. Important Region Identification using Edge Detection
4. Semantic-Aware Preprocessing
5. Patch Extraction (128 × 128)
6. CNN Autoencoder Compression
7. Patch Reconstruction
8. Full Image Reconstruction
9. Performance Evaluation

## Model Architecture

The compression model is a Convolutional Autoencoder.

Encoder:
Conv2D → MaxPooling  
Conv2D → MaxPooling  
Conv2D → MaxPooling

Latent Representation:
16 × 16 × 32

Decoder:
Conv2D → UpSampling  
Conv2D → UpSampling  
Conv2D → UpSampling

## Compression Details

Original Patch Size:  
128 × 128 × 3

Encoded Representation:  
16 × 16 × 32

Original Size:  
49152 values

Compressed Size:  
8192 values

Compression Ratio:  
≈ 6×

## Performance Metrics

The model is evaluated using the following metrics:

PSNR (Peak Signal-to-Noise Ratio)  
SSIM (Structural Similarity Index)  
Compression Ratio

## Results

Average PSNR:  
≈ 37.5

Average SSIM:  
≈ 0.92

Compression Ratio:  
≈ 6×

The results show that the proposed method effectively compresses high-resolution satellite imagery while preserving critical visual structures.

## Technologies Used

Python  
TensorFlow / Keras  
OpenCV  
Rasterio  
NumPy  
Matplotlib  
Streamlit

## Project Structure

satellite-compression
│
├── app.py  
├── model.keras  
├── satellite-compression.ipynb  
├── requirements.txt  
└── README.md

## How to Run

### Training (Google Colab)

1. Upload the notebook to Google Colab
2. Mount Google Drive
3. Place Sentinel-2 band images in a dataset folder
4. Update the dataset path
5. Run the notebook to train the model

### Running the Streamlit Application

Run the following command:

streamlit run app.py

This will launch the web interface where users can upload Sentinel-2 band images and obtain reconstructed compressed outputs along with PSNR and SSIM evaluation metrics.

## Future Work

Future improvements may include:

Supporting additional Sentinel-2 spectral bands  
Improving compression ratio using advanced neural architectures  
Integrating cloud-based processing for large-scale satellite datasets  
Developing real-time compression pipelines for satellite data streams
