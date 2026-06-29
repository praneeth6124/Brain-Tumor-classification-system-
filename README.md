# 🧠Brain-Tumor-classification-system
An end-to-end Deep Learning application for **multi-class Brain Tumor Classification** from MRI images using **EfficientNet-B0**. The project includes a trained model, FastAPI backend, Streamlit frontend, Dockerized deployment, and confidence-based predictions.

## Overview

This project is an end-to-end **Brain Tumor Classification System** that uses **EfficientNet-B0** and Transfer Learning to classify brain MRI scans into **Glioma, Meningioma, Pituitary Tumor,** and **No Tumor**.

The application features a **FastAPI backend**, an interactive **Streamlit frontend**, and is **Dockerized for deployment on Render**. Along with tumor classification, it provides **confidence scores**, **class probability distributions**, and **misclassification analysis** to improve model interpretability and reliability.

## Features

- Multi-class Brain MRI Classification
- EfficientNet-B0 Transfer Learning (PyTorch)
- **97.5% Test Accuracy**
- Confidence Score for every prediction
- Probability Breakdown for all classes
- Misclassification Analysis
- FastAPI REST API
- Interactive Streamlit Web Interface
- Dockerized Application for Consistent Deployment
- Deployed on Render using Docker Containers


## Tumor Classes

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor


## Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **97.5%** |
| Architecture | EfficientNet-B0 |
| Framework | PyTorch |
| Number of Classes | 4 |
| Deployment | Docker + Render |


### Misclassification Analysis

- Only **27 incorrect predictions** out of **1080** test images.
- Most confusion occurred between **Glioma** and **Meningioma**, indicating their visual similarity.
- Remaining errors were limited and primarily associated with challenging MRI scans.


## Application Features

- Upload Brain MRI Image
- Instant Prediction
- Confidence Score
- Probability Distribution
- Prediction Reliability Indicator
- MRI Upload Guidelines
- Model Information


## Tech Stack

- Python
- PyTorch
- EfficientNet-B0
- FastAPI
- Streamlit
- Docker
- Render
- Pillow
- NumPy

## Application Demo








## 🚀 Future Enhancement

The next phase of this project is to integrate a **U-Net-based Brain Tumor Segmentation** model for precise tumor localization.

- Generate **pixel-level segmentation masks** to accurately highlight tumor boundaries within MRI scans.
- Combine **tumor classification and localization** into a unified Computer-Aided Diagnosis (CAD) pipeline, providing both the predicted tumor type and its exact location.
- Improve model interpretability by enabling visual verification of predictions, making the system more transparent and easier to analyze.











