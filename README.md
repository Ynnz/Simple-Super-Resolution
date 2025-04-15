# 🖼️ Simple Super-Resolution with SRCNN

This project is a minimal implementation of image super-resolution using a basic convolutional neural network (SRCNN) on the CIFAR-10 dataset.

---

## 🚀 Features

- Lightweight model (SRCNN architecture)
- Trained on downsampled CIFAR-10 images
- Includes both training and testing scripts
- Uses PyTorch and torchvision

---

## 🧠 Model Overview

The model learns to reconstruct a high-resolution image from a low-resolution input using a 3-layer convolutional neural network based on the classic SRCNN paper.

---

## 🗂️ Project Structure

```
simple-super-resolution/
├── scripts/           # Training and testing scripts
│   ├── train.py
│   └── test.py
├── models/            # Model definition
│   └── srcnn.py
├── checkpoints/       # Saved model weights
├── data/              # CIFAR-10 dataset (auto-downloaded)
├── environment.yml    # Conda environment setup
└── README.md
```

---

## 🔧 Setup environment

```bash
conda env create -f environment.yml
conda activate superres-env
```

---

## 🏋️‍♀️ Training

```bash
python scripts/train.py
```

This will train SRCNN on a small subset (1000 images) of CIFAR-10 and save the model to:
```
checkpoints/srcnn_cifar10_subset.pth
```

---

## 🧪 Testing

```bash
python scripts/test.py
```

This will:
- Load a test image
- Create a low-resolution version
- Run super-resolution
- Show low-res, predicted, and ground truth images

---