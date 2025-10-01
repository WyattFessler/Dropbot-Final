import os
from ultralytics import YOLO
from pathlib import Path
import yaml
import sys
import numpy as np

HERE = Path(__file__).resolve().parent 
PROJECT_ROOT = HERE.parent           
DATA_YAML = PROJECT_ROOT / "data.yaml"
# Get file paths (root path and data.yaml)

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
#training parameters for model



metrics = model.val()
#evaluate the trained model with validation

print(metrics) 

#make preditions on images in test folder
results = model.predict(source="datasets/images/test", imgsz=640, conf=0.25)

# DISCLAIMER: ChatGPT was used to assist the developer of this code. Namely, 
# it was used to explain logic and syntax errors, and suggest strategies to fix,
# and used to explain certain functions/model parameters. LLMs were NOT used to 
# generate code, except as examples in order to learn
