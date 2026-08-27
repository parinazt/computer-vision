import argparse
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

def run_webcam_inference(checkpoint_path: str = None, class_labels: list = None):
    """
    Real-time vision classifier running inference over OpenCV webcam feed with FPS telemetry.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")

    # Load default ImageNet weights or custom fine-tuned weights
    if checkpoint_path:
        print(f"Loading custom weights from {checkpoint_path}...")
        model = models.resnet50()
        if class_labels:
            model.fc = torch.nn.Linear(model.fc.in_features, len(class_labels))
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Using standard ImageNet pretrained ResNet-50 weights...")
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Standard ImageNet top classes will be loaded if none provided
        if not class_labels:
            weights = models.ResNet50_Weights.DEFAULT
            class_labels = weights.meta["categories"]

    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera device 0.")
        return

    print("Camera feed initialized! Press 'q' or ESC in the window to exit.")

    fps = 0.0
    frame_count = 0
    start_time = time.time()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

            # Model Forward Pass
            logits = model(input_tensor)
            probabilities = torch.nn.functional.softmax(logits[0], dim=0)

            # Top-3 predictions
            top3_prob, top3_catid = torch.topk(probabilities, 3)

            # FPS calculation
            frame_count += 1
            if frame_count >= 10:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # Render overlay box
            h, w, _ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (15, 15), (420, 165), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.rectangle(frame, (15, 15), (420, 165), (255, 255, 255), 2)

            cv2.putText(frame, f"INFERENCE FPS: {fps:.1f}", (30, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            for i in range(3):
                class_id = top3_catid[i].item()
                label = class_labels[class_id] if class_labels and class_id < len(class_labels) else f"Class {class_id}"
                prob = top3_prob[i].item() * 100
                color = (0, 255, 0) if i == 0 else (200, 200, 200)
                text = f"#{i+1} {label[:22]}: {prob:.1f}%"
                cv2.putText(frame, text, (30, 75 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            cv2.imshow('Live Classifier Stream', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-Time Webcam Classification")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to .pth checkpoint')
    args = parser.parse_args()

    run_webcam_inference(checkpoint_path=args.checkpoint)
