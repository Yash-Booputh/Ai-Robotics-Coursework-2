#!/usr/bin/env python3
"""
Convert YOLOv8/YOLOv5 PyTorch model (.pt) to ONNX format
"""

from ultralytics import YOLO

# Load the PyTorch model
model = YOLO('best.pt')

# Export to ONNX format
model.export(format='onnx', imgsz=640)

print("✓ Model converted to best.onnx successfully!")
