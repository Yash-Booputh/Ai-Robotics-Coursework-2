#!/usr/bin/env python3
"""
Quick test to verify YOLO model loads and works with camera
"""

import cv2
from ultralytics import YOLO

print("="*60)
print("YOLO + Camera Test")
print("="*60)

# Load model
print("\n1. Loading YOLO model...")
model = YOLO('models/best.pt')
print(f"✅ Model loaded!")
print(f"   Classes: {model.names}")

# Test camera
print("\n2. Testing camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame = cap.read()
if ret:
    print(f"✅ Camera working! Frame shape: {frame.shape}")
else:
    print("❌ Camera failed!")
    exit(1)

# Test detection
print("\n3. Testing detection on camera frame...")
results = model(frame, conf=0.5, imgsz=320, verbose=False)

if results and len(results) > 0:
    result = results[0]
    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = result.names[cls]
            print(f"✅ Detected: {name} (confidence: {conf:.2f})")
    else:
        print("ℹ️  No objects detected in frame (this is OK if slots are empty)")
else:
    print("ℹ️  No detections")

cap.release()

print("\n" + "="*60)
print("✅ All tests passed! YOLO is ready to use.")
print("="*60)
