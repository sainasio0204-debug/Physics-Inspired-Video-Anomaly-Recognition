# Physics-Inspired Temporal Learning Framework for Multi-Class Video Anomaly Recognition with Uniformer-GRU

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Research](https://img.shields.io/badge/Research-ASSIC%202026-green)
![Task](https://img.shields.io/badge/Task-Video%20Anomaly%20Recognition-orange)

## Overview

This repository contains the implementation of our research paper:

**Physics-Inspired Temporal Learning Framework for Multi-Class Video Anomaly Recognition with Uniformer-GRU**

The proposed framework integrates:

- Uniformer-Mix spatio-temporal feature extraction
- GRU-based temporal modeling
- Physics-inspired entropy and kinetic energy fusion
- Multi-class anomaly classification
- Comparative evaluation against a CNN-ViT baseline

The goal is to improve anomaly recognition performance while introducing physically interpretable representations for surveillance video understanding.

---

## Abstract

Video anomaly detection is a critical component of intelligent surveillance systems used in public safety, crime prevention, and security monitoring. Traditional deep learning methods often lack interpretability and physically grounded reasoning.

To address this challenge, we propose a **Physics-Inspired Temporal Learning Framework** that integrates entropy and kinetic energy concepts into anomaly classification. The framework utilizes Uniformer-Mix for spatio-temporal feature extraction and a lightweight GRU-based temporal encoder for sequence modeling.

Experimental evaluation on a custom 15-class surveillance dataset demonstrates that the proposed Uniformer-GRU architecture significantly outperforms a CNN-ViT baseline.

---

## Key Contributions

- Physics-inspired anomaly representation using entropy and kinetic energy fusion.
- Uniformer-GRU temporal learning architecture.
- Custom multi-class surveillance anomaly dataset with 15 event categories.
- Lightweight design suitable for real-time deployment.
- Comprehensive comparison with CNN-ViT baseline.

---

## Architecture

```text
Raw Video
    ↓
Frame Sampling
    ↓
Video Preprocessing
    ↓
Uniformer-Mix Feature Extraction
    ↓
512-D Spatio-Temporal Features
    ↓
GRU Temporal Encoder
    ↓
Entropy + Kinetic Energy Fusion
    ↓
Fully Connected Layer
    ↓
Softmax Classification
    ↓
15-Class Prediction
```

---

## Dataset

The experiments were conducted on a custom surveillance video dataset.

### Dataset Statistics

| Property | Value |
|-----------|--------|
| Total Videos | 4024 |
| Number of Classes | 15 |
| Feature Dimension | 512 |
| Resolution | 224 × 224 |
| Sequence Length | 8 Frames |
| Feature Extractor | Uniformer-Mix |

### Example Classes

- Chain Snatching
- Fighting
- Robbery
- Road Accident
- Theft
- Vandalism
- Crowd Disturbance
- Suspicious Activity
- Multiple other surveillance anomalies

---

## Repository Structure

```text
Physics-Inspired-Video-Anomaly-Recognition/
│
├── uniformer_features/
│   ├── Chain_Snatching/
│   ├── Fighting/
│   ├── Robbery/
│   └── ...
│
├── models/
│   ├── uniformer_gru.py
│   ├── cnn_vit.py
│   └── physics_classifier.py
│
├── training/
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── roc_curves/
│   ├── confusion_matrices/
│   └── training_plots/
│
├── requirements.txt
└── README.md
```

---

## Training Configuration

| Parameter | Value |
|------------|---------|
| Epochs | 50 |
| Batch Size | 32 |
| Feature Dimension | 512 |
| Temporal Encoder | GRU |
| Optimizer | Adam |
| Framework | PyTorch |

---

## Results

### Uniformer-GRU

| Metric | Score |
|----------|---------|
| Training Accuracy | 99.93% |
| Test Accuracy | 87.78% |
| Precision | 82.98% |
| Recall | 80.33% |
| F1-Score | 81.24% |

### CNN-ViT Baseline

| Metric | Score |
|----------|---------|
| Training Accuracy | 81.25% |
| Test Accuracy | 71.24% |
| Precision | 63.82% |
| Recall | 64.71% |
| F1-Score | 63.89% |

The proposed Uniformer-GRU architecture achieves significantly higher classification performance compared to the CNN-ViT baseline.

---

## Installation

```bash
git clone https://github.com/yourusername/Physics-Inspired-Video-Anomaly-Recognition.git

cd Physics-Inspired-Video-Anomaly-Recognition

pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Evaluation

```bash
python evaluate.py
```

---

## Inference

```bash
python predict.py
```

---

## Future Work

- Physics-aware Graph Neural Networks
- Self-supervised anomaly representation learning
- Real-time edge deployment
- Explainable anomaly localization
- Multi-camera anomaly recognition

---

## Citation

```bibtex
@inproceedings{Sio2026PhysicsInspired,
  title={Physics-Inspired Temporal Learning Framework for Multi-Class Video Anomaly Recognition with Uniformer-GRU},
  author={Sio, Saina and Padhi, Amit Kumar and Moharana, Meena and Padhi, Dushmanta Kumar},
  booktitle={ASSIC 2026},
  year={2026}
}
```

