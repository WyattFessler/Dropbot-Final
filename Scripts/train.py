import os
from ultralytics import YOLO
from pathlib import Path
import yaml
import sys
import numpy as np

HERE = Path(__file__).resolve().parent 
PROJECT_ROOT = HERE.parent           
DATA_YAML = PROJECT_ROOT / "data.yaml"

model = YOLO("yolov8n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    batch=16,
    workers=4,
    project=str(PROJECT_ROOT / "runs"),
    name="yolov8_training",
    lr0=0.01,
    optimizer="Adam",
    mosaic=1.0,
    mixup=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4
)

# Evaluate the trained model on validation set (from data.yaml)
metrics = model.val()

# You can also specify options, e.g.:
metrics = model.val(
    data=str(DATA_YAML),   # your dataset YAML
    imgsz=640,             # image size
    batch=16
)

print(metrics)  # dict with mAP50, mAP50-95, precision, recall, etc.

# Run inference on a folder of images
results = model.predict(source="datasets/images/test", imgsz=640, conf=0.25)