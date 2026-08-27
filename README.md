# Computer Vision Classification Studio & Python Pipeline

A suite for computer vision classification with **PyTorch**, **TensorFlow**, **Hugging Face**, and **OpenCV**.

---

## 📁 Repository Structure

```
├── scripts/
│   ├── train_pytorch.py       # Full PyTorch ImageFolder trainer with AMP & Cosine Annealing
│   ├── train_tensorflow.py    # TensorFlow / Keras training pipeline with EfficientNet
│   └── inference_webcam.py    # Real-time OpenCV webcam classifier with FPS telemetry
├── requirements.txt           # Python dependencies (torch, torchvision, tensorflow, etc.)
├── src/                       # React & Tailwind interactive workbench web app
└── package.json
```

---

## 🚀 Quickstart Guide (Python)

### 1. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Organize Your Dataset
Organize training images into standard `ImageFolder` format:
```
data/
  ├── train/
  │   ├── cats/
  │   └── dogs/
  └── val/
      ├── cats/
      └── dogs/
```

### 3. Run PyTorch Training
```bash
python scripts/train_pytorch.py --data_dir data --epochs 20 --batch_size 32
```

### 4. Run TensorFlow Training
```bash
python scripts/train_tensorflow.py --data_dir data --epochs 20 --batch_size 32
```

### 5. Run Real-Time Webcam Inference
```bash
# Using standard pretrained ImageNet ResNet-50:
python scripts/inference_webcam.py

# Or using your fine-tuned checkpoint:
python scripts/inference_webcam.py --checkpoint best_model.pth
```
