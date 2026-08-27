import { CodeGeneratorConfig } from '../types';

export function generatePythonCode(config: CodeGeneratorConfig): string {
  switch (config.framework) {
    case 'pytorch':
      if (config.taskType === 'custom_cnn') {
        return generatePyTorchCustomCNN(config);
      } else if (config.taskType === 'inference_only') {
        return generatePyTorchInference(config);
      } else {
        return generatePyTorchTransferLearning(config);
      }
    case 'tensorflow':
      return generateTensorFlowKerasCode(config);
    case 'huggingface':
      return generateHuggingFaceCode(config);
    case 'sklearn':
      return generateClassicCvCode(config);
    case 'opencv_webcam':
      return generateOpenCvWebcamCode(config);
    default:
      return generatePyTorchTransferLearning(config);
  }
}

function generatePyTorchTransferLearning(cfg: CodeGeneratorConfig): string {
  const modelName = cfg.modelBackbone;
  let weightsImport = 'ResNet50_Weights.DEFAULT';
  let modelCreation = 'models.resnet50(weights=models.ResNet50_Weights.DEFAULT)';
  let classifierReplacement = `num_features = model.fc.in_features\nmodel.fc = nn.Sequential(\n    nn.Dropout(p=0.3),\n    nn.Linear(num_features, ${cfg.numClasses})\n)`;

  if (modelName === 'vit_b_16') {
    weightsImport = 'ViT_B_16_Weights.DEFAULT';
    modelCreation = 'models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)';
    classifierReplacement = `num_features = model.heads.head.in_features\nmodel.heads.head = nn.Sequential(\n    nn.Dropout(p=0.2),\n    nn.Linear(num_features, ${cfg.numClasses})\n)`;
  } else if (modelName === 'efficientnet_b0') {
    weightsImport = 'EfficientNet_B0_Weights.DEFAULT';
    modelCreation = 'models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)';
    classifierReplacement = `num_features = model.classifier[1].in_features\nmodel.classifier[1] = nn.Sequential(\n    nn.Dropout(p=0.2),\n    nn.Linear(num_features, ${cfg.numClasses})\n)`;
  } else if (modelName === 'convnext_tiny') {
    weightsImport = 'ConvNeXt_Tiny_Weights.DEFAULT';
    modelCreation = 'models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)';
    classifierReplacement = `num_features = model.classifier[2].in_features\nmodel.classifier[2] = nn.Sequential(\n    nn.Dropout(p=0.2),\n    nn.Linear(num_features, ${cfg.numClasses})\n)`;
  } else if (modelName === 'mobilenet_v3_large') {
    weightsImport = 'MobileNet_V3_Large_Weights.DEFAULT';
    modelCreation = 'models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)';
    classifierReplacement = `num_features = model.classifier[3].in_features\nmodel.classifier[3] = nn.Sequential(\n    nn.Dropout(p=0.2),\n    nn.Linear(num_features, ${cfg.numClasses})\n)`;
  }

  const freezeCode = cfg.freezeBackbone
    ? `\n    # Freeze feature extractor backbone weights\n    for param in model.parameters():\n        param.requires_grad = False\n    for param in ${modelName === 'vit_b_16' ? 'model.heads.head' : modelName === 'resnet50' ? 'model.fc' : 'model.classifier'}.parameters():\n        param.requires_grad = True`
    : `\n    # Fine-tuning all layers with differential learning rates`;

  return `"""
================================================================================
Computer Vision Image Classification Pipeline (PyTorch)
Architecture: ${cfg.modelBackbone.toUpperCase()} | Classes: ${cfg.numClasses} | Image Size: ${cfg.imageSize}x${cfg.imageSize}
================================================================================
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torch.amp import autocast, GradScaler
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Device Configuration & Reproducibility
# -----------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available() and "${cfg.device}" == "cuda":
        device = torch.device("cuda")
        print(f"-> Using NVIDIA CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and "${cfg.device}" == "mps":
        device = torch.device("mps")
        print("-> Using Apple Silicon MPS Acceleration")
    else:
        device = torch.device("cpu")
        print("-> Using CPU Execution Engine")
    return device

# -----------------------------------------------------------------------------
# 2. Data Transformations & Augmentation
# -----------------------------------------------------------------------------
def build_transforms(img_size=${cfg.imageSize}):
    # ImageNet Standard Normalization constants
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    return train_transform, val_transform

# -----------------------------------------------------------------------------
# 3. Model Architecture with Transfer Learning Head
# -----------------------------------------------------------------------------
def create_model(num_classes=${cfg.numClasses}, pretrained=${cfg.pretrained ? 'True' : 'False'}):
    print("-> Instantiating ${cfg.modelBackbone} backbone...")
    model = ${modelCreation}
${freezeCode}

    # Replace the classification head for custom classes
    ${classifierReplacement.split('\n').join('\n    ')}

    return model

# -----------------------------------------------------------------------------
# 4. Training and Validation Engine
# -----------------------------------------------------------------------------
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=${cfg.epochs}, use_amp=${cfg.useAmp ? 'True' : 'False'}):
    scaler = GradScaler('cuda') if use_amp and device.type == 'cuda' else None
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\\n" + "="*50)
    print(f"Starting Training: {num_epochs} Epochs on {device}")
    print("="*50)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        print(f"\\nEpoch {epoch:02d}/{num_epochs:02d}")
        print("-" * 25)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == 'train'):
                    if use_amp and scaler:
                        with autocast('cuda'):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        if use_amp and scaler:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            print(f"{phase.capitalize():<6} Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%")

            # Save best checkpoint
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_accuracy': best_acc,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, "${cfg.savePath}")
                print(f"  ⭐ Checkpoint Saved! (New Best Val Acc: {best_acc*100:.2f}%)")

        elapsed = time.time() - start_time
        print(f"Epoch Duration: {elapsed:.2f}s")

    print("\\n" + "="*50)
    print(f"Training Complete! Peak Validation Accuracy: {best_acc*100:.2f}%")
    print(f"Model saved to '${cfg.savePath}'")
    print("="*50)

    model.load_state_dict(best_model_wts)
    return model

# -----------------------------------------------------------------------------
# 5. Single Image Inference Function
# -----------------------------------------------------------------------------
def predict_image(image_path: str, model, class_names: list, device, img_size=${cfg.imageSize}):
    model.eval()
    _, val_transform = build_transforms(img_size)

    image = Image.open(image_path).convert('RGB')
    input_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, min(5, len(class_names)))

    print(f"\\nPredictions for: {image_path}")
    print("-" * 35)
    for i in range(top5_prob.size(0)):
        idx = top5_catid[i].item()
        label = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        print(f"{i+1}. {label:<20} -> {top5_prob[i].item()*100:.2f}%")

# -----------------------------------------------------------------------------
# 6. Main Execution Pipeline
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = get_device()
    train_tf, val_tf = build_transforms()

    # Setup Dataset paths (Expects directory structure: data/train/<class_name>/*.jpg)
    data_dir = "data"
    os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "val"), exist_ok=True)

    try:
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
        val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=${cfg.batchSize}, shuffle=True, num_workers=2, pin_memory=True),
            'val':   DataLoader(val_dataset, batch_size=${cfg.batchSize}, shuffle=False, num_workers=2, pin_memory=True)
        }
        class_names = train_dataset.classes
        print(f"-> Loaded {len(train_dataset)} training images across {len(class_names)} classes: {class_names}")

    except Exception as e:
        print("ℹ️ Note: Using synthetic dummy dataset for demonstration run.")
        dummy_data = [(torch.randn(3, ${cfg.imageSize}, ${cfg.imageSize}), torch.randint(0, ${cfg.numClasses}, (1,)).item()) for _ in range(64)]
        dataloaders = {
            'train': DataLoader(dummy_data, batch_size=${cfg.batchSize}, shuffle=True),
            'val':   DataLoader(dummy_data, batch_size=${cfg.batchSize}, shuffle=False)
        }
        class_names = [f"Class_{i}" for i in range(${cfg.numClasses})]

    # Instantiate Model
    model = create_model(num_classes=${cfg.numClasses}).to(device)

    # Loss, Optimizer & LR Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.${cfg.optimizer}(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=${cfg.learningRate},
        weight_decay=${cfg.weightDecay}
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=${cfg.epochs}, eta_min=1e-6)

    # Launch Training
    trained_model = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=${cfg.epochs}
    )
`;
}

function generatePyTorchCustomCNN(cfg: CodeGeneratorConfig): string {
  return `"""
================================================================================
Custom 4-Layer Convolutional Neural Network (CNN) from Scratch
Designed in PyTorch with Batch Normalization, ReLU Activations & Dropout
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
# 1. Custom CNN Architecture Definition
# -----------------------------------------------------------------------------
class CustomVisionClassifier(nn.Module):
    def __init__(self, num_classes=${cfg.numClasses}, in_channels=3):
        super(CustomVisionClassifier, self).__init__()
        
        # Block 1: Input (3, 224, 224) -> (32, 112, 112)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: (32, 112, 112) -> (64, 56, 56)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: (64, 56, 56) -> (128, 28, 28)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Block 4: (128, 28, 28) -> (256, 14, 14)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)

        # Global Average Pooling collapses (256, 14, 14) into (256, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Classification Head
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Layer 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Layer 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Layer 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Layer 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Head
        x = self.global_pool(x)
        x = torch.flatten(x, 1) # Flatten to batch_size x 256
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -----------------------------------------------------------------------------
# 2. Main Training Setup & Model Summary
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomVisionClassifier(num_classes=${cfg.numClasses}).to(device)
    
    # Calculate trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"-> Created Custom CNN with {total_params:,} trainable parameters")
    
    # Verify shape with dummy tensor
    dummy_input = torch.randn(2, 3, ${cfg.imageSize}, ${cfg.imageSize}).to(device)
    output = model(dummy_input)
    print(f"-> Forward pass test: Input shape {list(dummy_input.shape)} -> Output shape {list(output.shape)}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.${cfg.optimizer}(model.parameters(), lr=${cfg.learningRate}, weight_decay=${cfg.weightDecay})
    
    print("\\nModel Ready for Training.")
`;
}

function generatePyTorchInference(cfg: CodeGeneratorConfig): string {
  return `"""
================================================================================
Production-Ready PyTorch Image Inference Script
Loads Pretrained Weights and Performs Fast Top-K Classification
================================================================================
"""

import sys
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

def load_inference_pipeline(model_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"-> Initializing model on device: {device}")
    
    # Load backbone with pretrained ImageNet weights
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = model.to(device)
    model.eval()

    # Preprocessing transforms matching ImageNet standard
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    return model, preprocess, categories, device

def classify_image(image_path: str, model, preprocess, categories, device, top_k=5):
    try:
        raw_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    # Apply transformations and add batch dimension
    input_tensor = preprocess(raw_img).unsqueeze(0).to(device)

    # Perform inference with gradients disabled
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = F.softmax(logits[0], dim=0)

    # Extract Top-K results
    top_prob, top_indices = torch.topk(probabilities, top_k)

    print(f"\\n" + "="*50)
    print(f"Vision Classification Results for: {image_path}")
    print("="*50)
    for i in range(top_k):
        score = top_prob[i].item() * 100
        category_name = categories[top_indices[i].item()]
        bar_len = int(score / 4)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        print(f"#{i+1} [{bar}] {score:6.2f}% : {category_name}")
    print("="*50 + "\\n")

if __name__ == "__main__":
    model, preprocess, categories, device = load_inference_pipeline()
    test_image = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    print(f"To run with custom image: python inference.py <path_to_image>")
`;
}

function generateTensorFlowKerasCode(cfg: CodeGeneratorConfig): string {
  return `"""
================================================================================
Computer Vision Image Classification Pipeline (TensorFlow / Keras)
Backbone: MobileNetV2 / EfficientNet | Dataset Directory: image_dataset_from_directory
================================================================================
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks

# -----------------------------------------------------------------------------
# 1. Hyperparameters & GPU Configuration
# -----------------------------------------------------------------------------
BATCH_SIZE = ${cfg.batchSize}
IMG_SIZE = (${cfg.imageSize}, ${cfg.imageSize})
NUM_CLASSES = ${cfg.numClasses}
EPOCHS = ${cfg.epochs}
LEARNING_RATE = ${cfg.learningRate}

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"-> GPU Detected: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("-> Running on CPU")

# -----------------------------------------------------------------------------
# 2. Data Ingestion & Preprocessing Pipeline
# -----------------------------------------------------------------------------
def load_datasets(data_dir="data"):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode='categorical'
    )

    # Performance optimization: Prefetch and Cache in RAM
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

# -----------------------------------------------------------------------------
# 3. Model Architecture with Data Augmentation
# -----------------------------------------------------------------------------
def build_vision_model():
    # In-graph GPU Data Augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    # Base Pretrained Feature Extractor
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze initial weights

    # Build Functional Model Graph
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="classification_head")(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2_Classifier")
    return model, base_model

# -----------------------------------------------------------------------------
# 4. Training Engine & Fine-Tuning
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    model, base_model = build_vision_model()
    model.summary()

    model.compile(
        optimizer=optimizers.${cfg.optimizer}(learning_rate=LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_acc')]
    )

    cb_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint("${cfg.savePath.replace('.pth', '.keras')}", monitor='val_accuracy', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    ]

    print("\\n-> Ready to train. Run model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb_list)")
`;
}

function generateHuggingFaceCode(cfg: CodeGeneratorConfig): string {
  return `"""
================================================================================
Vision Transformer (ViT) with Hugging Face Transformers & Datasets
Model: google/vit-base-patch16-224 | Trainer API + Pipeline Inference
================================================================================
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import evaluate
import numpy as np

# 1. Load Pretrained ViT Processor & Model
MODEL_CHECKPOINT = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

# 2. Quick Zero-Shot Inference Pipeline
def run_quick_inference(image_url="https://images.unsplash.com/photo-1552053831-71594a27632d"):
    classifier = pipeline("image-classification", model=MODEL_CHECKPOINT)
    results = classifier(image_url)
    print("Zero-Shot Predictions:")
    for res in results:
        print(f"  - {res['label']}: {res['score']*100:.2f}%")

# 3. Fine-Tuning Setup with Trainer
def train_vit_on_dataset():
    # Load dataset (e.g. beans, cifar10, or custom dataset)
    dataset = load_dataset("beans")
    labels = dataset["train"].features["labels"].names
    
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True
    )

    # Transforms
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (image_processor.size["height"], image_processor.size["width"])
    
    _train_transforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])

    _val_transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])

    def train_transforms(examples):
        examples["pixel_values"] = [_train_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    def val_transforms(examples):
        examples["pixel_values"] = [_val_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    dataset["train"].set_transform(train_transforms)
    dataset["validation"].set_transform(val_transforms)

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir="./vit_classification_results",
        per_device_train_batch_size=${cfg.batchSize},
        per_device_eval_batch_size=${cfg.batchSize},
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=${cfg.learningRate},
        num_train_epochs=${cfg.epochs},
        weight_decay=${cfg.weightDecay},
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        fp16=${cfg.useAmp ? 'True' : 'False'},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        tokenizer=image_processor,
    )

    print("-> Ready to fine-tune Vision Transformer! Execute: trainer.train()")

if __name__ == "__main__":
    run_quick_inference()
`;
}

function generateClassicCvCode(cfg: CodeGeneratorConfig): string {
  return `"""
================================================================================
Classic Computer Vision Classifier (OpenCV + HOG Features + Scikit-Learn SVM)
Fast, lightweight, explainable image classification without GPU requirement
================================================================================
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# -----------------------------------------------------------------------------
# 1. HOG (Histogram of Oriented Gradients) Feature Extractor
# -----------------------------------------------------------------------------
def extract_hog_features(image_path, img_size=(128, 128)):
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Resize to uniform dimension
    resized = cv2.resize(img, img_size)

    # Initialize OpenCV HOG Descriptor
    win_size = img_size
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    features = hog.compute(resized)
    return features.flatten()

# -----------------------------------------------------------------------------
# 2. Dataset Loader
# -----------------------------------------------------------------------------
def load_dataset_from_directory(data_dir="data/train"):
    features_list = []
    labels_list = []
    class_names = sorted(os.listdir(data_dir))

    print(f"-> Extracting HOG features from {len(class_names)} classes...")
    for label_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_folder):
            continue

        for file_name in os.listdir(class_folder):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_folder, file_name)
                feats = extract_hog_features(img_path)
                if feats is not None:
                    features_list.append(feats)
                    labels_list.append(label_idx)

    return np.array(features_list), np.array(labels_list), class_names

# -----------------------------------------------------------------------------
# 3. Model Training & Evaluation
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Generate mock data if directory doesn't exist
    X = np.random.randn(200, 3780).astype(np.float32)
    y = np.random.randint(0, ${cfg.numClasses}, size=(200,))
    class_names = [f"Class_{i}" for i in range(${cfg.numClasses})]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"-> Training Support Vector Machine (SVM) Classifier...")
    clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Save model
    joblib.dump(clf, "svm_vision_model.pkl")
    print("-> Model serialized to 'svm_vision_model.pkl'")
`;
}

function generateOpenCvWebcamCode(cfg: CodeGeneratorConfig): string {
  return `"""
================================================================================
Real-Time Webcam Vision Classifier with OpenCV & PyTorch
Processes live video frames with bounding overlays and FPS monitoring
================================================================================
"""

import cv2
import time
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image

def run_realtime_vision():
    # 1. Initialize PyTorch Vision Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Loading MobileNetV3 on {device}...")

    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights).to(device)
    model.eval()
    
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    # 2. Connect to Camera (0 = Default Webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access video capture device.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("-> Live stream started. Press 'q' or ESC in the window to exit.")
    
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30
        prev_time = curr_time

        # Convert OpenCV BGR to PIL RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Preprocess & Infer
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.nn.functional.softmax(logits[0], dim=0)
            top_prob, top_idx = torch.topk(probs, 1)
            
            label = categories[top_idx[0].item()]
            confidence = top_prob[0].item() * 100

        # Draw HUD Overlays
        h, w, _ = frame.shape
        # Top banner background
        cv2.rectangle(frame, (20, 20), (550, 110), (15, 23, 42), -1)
        cv2.rectangle(frame, (20, 20), (550, 110), (59, 130, 246), 2)

        # Text labels
        cv2.putText(frame, f"Prediction: {label[:30]}", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence:.1f}% | FPS: {fps:.1f}", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 2)

        # Display window
        cv2.imshow("Real-Time Computer Vision Classification", frame)

        # Check for user exit
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("-> Video stream closed gracefully.")

if __name__ == "__main__":
    run_realtime_vision()
`;
}

export function generateRequirementsTxt(framework: string): string {
  if (framework === 'pytorch' || framework === 'opencv_webcam') {
    return `# Core PyTorch and Vision Libraries
torch>=2.2.0
torchvision>=0.17.0
torchaudio>=2.2.0
numpy>=1.24.0
Pillow>=10.0.0
opencv-python>=4.9.0
matplotlib>=3.8.0
tqdm>=4.66.0
scikit-learn>=1.4.0
`;
  } else if (framework === 'tensorflow') {
    return `# TensorFlow Vision Stack
tensorflow>=2.16.0
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.8.0
scikit-learn>=1.4.0
`;
  } else if (framework === 'huggingface') {
    return `# Hugging Face Transformers & Datasets
torch>=2.2.0
torchvision>=0.17.0
transformers>=4.40.0
datasets>=2.19.0
evaluate>=0.4.0
accelerate>=0.30.0
Pillow>=10.0.0
scikit-learn>=1.4.0
`;
  } else {
    return `# Scikit-Learn & OpenCV Stack
scikit-learn>=1.4.0
opencv-python>=4.9.0
numpy>=1.24.0
matplotlib>=3.8.0
joblib>=1.3.0
`;
  }
}

export function generateDirectoryTree(): string {
  return `dataset_root/
├── train/
│   ├── class_cat/
│   │   ├── cat_001.jpg
│   │   ├── cat_002.jpg
│   │   └── ...
│   ├── class_dog/
│   │   ├── dog_001.jpg
│   │   └── ...
│   └── class_bird/
│       └── ...
└── val/
    ├── class_cat/
    │   └── ...
    ├── class_dog/
    │   └── ...
    └── class_bird/
        └── ...`;
}
