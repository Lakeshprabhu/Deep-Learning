# Digits Classification: CNN vs Traditional ML Models

## Overview
This project classifies handwritten digits using the **Digits dataset** from `sklearn.datasets`. The performance of a **Convolutional Neural Network (CNN)** is compared with traditional machine learning models such as **Random Forest**, **Support Vector Machine (SVM)**, and **K-Nearest Neighbors (KNN)**.

The goal is to highlight the difference between deep learning and classical ML approaches on small-scale image data.

---

## Dataset
- **Source:** `sklearn.datasets.load_digits`
- **Number of samples:** 1,797 images
- **Image size:** 8×8 pixels
- **Number of classes:** 10 (digits 0–9)
- **Input format:** Grayscale images

---

## Preprocessing
1. **Normalization**: Pixel values scaled to [0,1] for CNN.
2. **Reshaping**:
   - CNN: `(n_samples, 8, 8, 1)`
   - ML models: flattened to `(n_samples, 64)`
3. **Train-test split**: 80% train, 20% test.

---

## Models Implemented

### 1. Convolutional Neural Network (CNN)
- Architecture:
  - Conv2D → ReLU → MaxPooling
  - Flatten
  - Dense → ReLU
  - Dense → Softmax (10 classes)
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `Adam`
- Epochs: 20
- Input shape: `(8,8,1)`

### 2. Random Forest Classifier
- Input: Flattened images `(64,)`
- Number of estimators: 100
- Max depth: None (default)
- Suitable for small tabular image data.
