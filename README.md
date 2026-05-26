# 🫁 Lung Cancer Classification Using Deep Learning and Transfer Learning (VGG19)

<div align="center">

### AI-Powered Medical Image Classification for Early Lung Cancer Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-ImageProcessing-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Dataset Description](#-dataset-description)
- [Project Workflow](#-project-workflow)
- [Data Preprocessing](#-data-preprocessing)
- [Data Augmentation](#-data-augmentation)
- [Transfer Learning using VGG19](#-transfer-learning-using-vgg19)
- [Model Architecture](#-model-architecture)
- [Training Strategy](#-training-strategy)
- [Evaluation Metrics](#-evaluation-metrics)
- [Experimental Results](#-experimental-results)
- [Applications](#-applications)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Contributors](#-contributors)
- [License](#-license)

---

# 🎯 Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection significantly increases survival rates and improves treatment outcomes.

This project presents an Artificial Intelligence-based solution that automatically classifies lung CT scan images into:

- ✅ Normal
- ✅ Benign
- ✅ Malignant

using a Deep Learning model based on **Transfer Learning with VGG19**.

The system assists radiologists by providing fast and accurate classification of lung abnormalities and can serve as a clinical decision-support tool.

---

# 🚨 Problem Statement

Manual interpretation of CT scans is time-consuming and prone to diagnostic variability. Detecting subtle lung abnormalities requires significant expertise and effort.

The goal of this project is to develop a robust deep learning system capable of:

- Detecting lung abnormalities
- Differentiating benign and malignant nodules
- Reducing diagnostic workload
- Supporting early cancer detection

---

# 🎯 Project Objectives

### Primary Objectives

- Classify lung CT scans into Normal, Benign, and Malignant categories
- Improve diagnostic accuracy
- Assist healthcare professionals
- Reduce screening time

### Secondary Objectives

- Utilize Transfer Learning for efficient learning
- Handle class imbalance
- Improve model generalization using augmentation
- Build a scalable framework for deployment

---

# 📂 Dataset Description

## Dataset Source

**Kaggle Lung Cancer CT Scan Dataset**

The dataset contains more than **14,600 CT scan images** categorized into three classes.

### Classes

| Class | Description |
|---------|---------|
| Normal | Healthy lung tissue |
| Benign | Non-cancerous lung nodule |
| Malignant | Cancerous lung nodule |

---

# 🔄 Project Workflow

```text
CT Scan Images
       │
       ▼
Data Collection
       │
       ▼
Image Preprocessing
       │
       ▼
Data Augmentation
       │
       ▼
Transfer Learning (VGG19)
       │
       ▼
Custom Classification Head
       │
       ▼
Model Training
       │
       ▼
Performance Evaluation
       │
       ▼
Prediction & Classification
```

---

# 🖼 Data Preprocessing

To improve model performance, the following preprocessing steps were applied:

### Image Resizing

- Converts images into a fixed dimension suitable for VGG19
- Reduces computational complexity

### Pixel Normalization

- Scales pixel values into a standard range
- Improves convergence speed

### Dataset Cleaning

- Removes corrupted images
- Eliminates duplicate samples
- Ensures dataset consistency

---

# 🔄 Data Augmentation

Medical datasets are often limited and imbalanced.

To overcome this issue, augmentation techniques were applied:

### Rotation

Creates rotated versions of images to improve rotational invariance.

### Horizontal Flip

Generates mirrored samples to increase diversity.

### Brightness Adjustment

Introduces illumination variations for robust learning.

### Zoom Transformations

Allows learning at multiple scales.

### Benefits

- Reduces overfitting
- Improves generalization
- Increases dataset diversity
- Enhances robustness

---

# 🧠 Transfer Learning using VGG19

## What is Transfer Learning?

Transfer Learning is a technique where a model trained on a large dataset is reused for a different but related task.

Instead of training a CNN from scratch:

```text
Random Initialization
       ↓
Feature Learning
       ↓
Classification
```

we leverage a pretrained VGG19 network:

```text
Pretrained VGG19
       ↓
Feature Extraction
       ↓
Fine Tuning
       ↓
Classification
```

---

# 🏗 Model Architecture

The original VGG19 classification layers were removed and replaced with a custom classification head.

### Architecture

```text
Input CT Scan
       │
       ▼
VGG19 Base Model
(Pretrained on ImageNet)
       │
       ▼
Flatten Layer
       │
       ▼
Dense Layer
       │
       ▼
Batch Normalization
       │
       ▼
Dropout Layer
       │
       ▼
Dense Layer
       │
       ▼
Softmax Output Layer
```

### Added Components

#### Dense Layer

Learns high-level medical image representations.

#### Batch Normalization

Benefits:

- Faster convergence
- Stable training
- Better gradient propagation

#### Dropout

Benefits:

- Prevents overfitting
- Improves generalization

#### Softmax Layer

Outputs class probabilities for:

- Normal
- Benign
- Malignant

---

# ⚙ Training Strategy

## Optimizer

### Stochastic Gradient Descent (SGD)

Advantages:

- Stable learning
- Better generalization
- Widely adopted for image classification

---

## Loss Function

### Categorical Cross Entropy

Used for multi-class classification problems.

Measures the difference between:

- Actual labels
- Predicted probabilities

---

## Class Weighting

To address class imbalance, weighted learning was employed.

Benefits:

- Improves minority class recognition
- Reduces prediction bias
- Enhances malignant detection

---

# 📊 Evaluation Metrics

Multiple performance metrics were used.

### Accuracy

Measures overall prediction correctness.

### Precision

Measures prediction reliability.

> Of all predicted positive cases, how many were actually positive?

### Recall (Sensitivity)

Measures detection capability.

> Of all actual positive cases, how many were correctly identified?

### F1 Score

Balances Precision and Recall.

---

# 🏆 Experimental Results

The proposed model achieved excellent performance.

| Metric | Score |
|----------|----------|
| Training Accuracy | 99.68% |
| Validation Accuracy | 98.11% |
| Test Accuracy | 97.93% |

---

# 📈 Confusion Matrix

The confusion matrix demonstrates:

- High true positive rate
- Minimal false positives
- Minimal false negatives
- Strong separation between classes

<p align="center">
  <img src="images/confusion_matrix.png" width="600">
</p>

---

# 💡 Applications

## Early Cancer Detection

Detect suspicious nodules at an early stage.

## Radiologist Assistance

Acts as a second-opinion system.

## Screening Prioritization

Helps hospitals prioritize high-risk patients.

## Surgical Planning

Supports treatment planning decisions.

## Telemedicine Integration

Can be integrated into remote healthcare systems.

---

# ⚠ Limitations

Current limitations include:

- Dependence on dataset quality
- Single-model architecture
- Limited explainability
- No external clinical validation
- Potential dataset bias

---

# 🚀 Future Enhancements

### Ensemble Learning

Combine:

- VGG19
- ResNet50
- DenseNet121

### Explainable AI

Implement:

- Grad-CAM
- Saliency Maps

### API Deployment

Deploy using:

- FastAPI
- Flask

### Web Application

Develop a user-friendly interface for clinicians.

### Clinical Validation

Validate performance on external hospital datasets.

---

# 🛠 Technology Stack

## Programming Language

- Python

## Deep Learning

- TensorFlow
- Keras

## Data Processing

- NumPy
- Pandas

## Computer Vision

- OpenCV
- Pillow (PIL)

## Visualization

- Matplotlib
- Seaborn

## Development Environment

- Jupyter Notebook
- Google Colab

---

# 📁 Project Structure

```bash
Lung-Cancer-Classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── training.ipynb
│   └── evaluation.ipynb
│
├── models/
│   └── vgg19_model.h5
│
├── images/
│   ├── confusion_matrix.png
│   └── architecture.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# ⚡ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/lung-cancer-classification.git

cd lung-cancer-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶ Usage

### Train Model

```bash
python train.py
```

### Evaluate Model

```bash
python evaluate.py
```

### Predict on New Images

```bash
python predict.py
```

---

# 📷 Results Visualization

### Sample Prediction

```text
Input CT Scan
        ↓
Prediction: Malignant
Confidence: 98.6%
```

### Model Performance

✔ High Accuracy

✔ Strong Recall

✔ Low Misclassification Rate

✔ Robust Generalization

---
