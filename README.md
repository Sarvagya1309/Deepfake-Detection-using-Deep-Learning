# Deepfake Detection using ResNet50 + Temporal Attention + MLP-EWC

## 📌 Overview
This project focuses on detecting deepfake videos using frame-based spatial feature extraction
combined with temporal attention and continual learning.

## 🧠 Methodology
- Face Detection: SCRFD
- Feature Extraction: ResNet50 + Multi-Frame Temporal Attention Residual Block (MF-TARB)
- Classifier: MLP with Elastic Weight Consolidation (EWC)
- Learning Paradigm: Continual Learning

## 📂 Datasets
- FaceForensics++ (FF++)
- Deepfake Detection
- Celeb-df(v2)

⚠️ Datasets are not included due to size and licensing constraints.

## ⚙️ Installation
```bash
pip install -r requirements.txt


## Pipeline Overview

1. Face extraction using SCRFD
2. Spatial feature extraction using ResNet50
3. Temporal modeling using MF-TARB
4. Classification using MLP
5. Continual learning using Elastic Weight Consolidation
