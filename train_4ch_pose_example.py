#!/usr/bin/env python3
"""
Sample training script for 4-channel YOLOv8 Pose model.
"""

from ultralytics import YOLO

def train_4ch_pose_model():
    """Train 4-channel pose model."""
    # Load the 4-channel pose model
    model = YOLO('yolov8n-4ch-pose.yaml')
    
    # Train the model
    results = model.train(
        data='coco8-pose.yaml',  # pose dataset
        epochs=100,
        imgsz=640,
        batch=8,  # Reduced batch size due to increased memory usage
        device=0,  # Use GPU if available
        project='runs/pose',
        name='yolov8n-4ch-pose',
        save=True,
        verbose=True
    )
    
    # Validate the model
    results = model.val()
    
    # Export the model
    model.export(format='onnx')
    
    return results

if __name__ == '__main__':
    train_4ch_pose_model()
