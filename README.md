# MNIST Digit Classification — AI 100 Midterm Project

**Author:** Paras Rathi
**Course:** AI 100 — Deep Learning Practice
**Submitted:** March 1, 2026

---

## Overview

This project trains a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits (0–9) from the MNIST dataset.

**Final Results:**
- Training Accuracy: **99.18%**
- Test Accuracy: **99.21%**
- Test Loss: **0.0261** (after 10 epochs)

---

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training script — downloads data, trains CNN, saves plots |
| `Paras_Rathi_Midterm_Report.pdf` | Full project report (problem, model, results, lessons) |
| `README.md` | This file |

---

## Requirements

```bash
pip install torch torchvision scikit-learn matplotlib
```

> Tested with Python 3.9+. Uses CPU by default; automatically uses CUDA GPU if available.

---

## How to Run

```bash
python train.py
```

This will:
1. Download the MNIST dataset automatically (~11 MB) to `./data/`
2. Train the CNN for 10 epochs
3. Print training/test accuracy per epoch and a classification report
4. Save the following files:
   - `training_curves.png` — loss and accuracy plots
   - `confusion_matrix.png` — per-class confusion matrix
   - `sample_predictions.png` — sample test predictions
   - `results.json` — all metrics as JSON
   - `mnist_cnn.pth` — saved model weights

---

## Model Architecture

```
Input: 1×28×28 grayscale image

Conv Block 1:  Conv2d(1→32, 3×3) → ReLU → MaxPool(2×2)  →  32×14×14
Conv Block 2:  Conv2d(32→64, 3×3) → ReLU → MaxPool(2×2) →  64×7×7
Flatten        →  3,136
FC 1:          Linear(3136→128) → ReLU → Dropout(0.5)
FC 2 (output): Linear(128→10)

Total parameters: 421,642
Loss: CrossEntropyLoss | Optimizer: Adam (lr=0.001) | Epochs: 10 | Batch: 64
```
