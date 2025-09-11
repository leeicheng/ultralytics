#!/usr/bin/env python3
"""
YOLOv8 Training Script with Weights & Biases Integration
Author: Claude
Description: Train YOLOv8m model on custom dataset with wandb logging
"""

import os
import argparse
from pathlib import Path
import wandb
from ultralytics import YOLO


def setup_wandb(project_name="yolov8-training", run_name=None):
    """Initialize Weights & Biases logging"""
    wandb.init(
        project=project_name,
        name=run_name,
        save_code=True,
        tags=["yolov8pose", "detection", "training"]
    )
    return wandb.config


def train_yolo(
    model_path="yolov8m-pose.pt",
    data_path="F:/NYCU/ultralytics/datasets/data.yaml",
    epochs=100,
    batch_size=8,
    image_size=640,
    device="cuda",
    project="yolov8-pose-training",
    name="junction-detection-with-original",
    patience=50,
    learning_rate=0.01,
    use_wandb=True
):
    """
    Train YOLOv8 model with specified parameters
    
    Args:
        model_path: Path to pretrained model or model architecture
        data_path: Path to dataset YAML file
        epochs: Number of training epochs
        batch_size: Batch size for training
        image_size: Input image size
        device: Device to use for training
        project: Project name for saving results
        name: Run name for this experiment
        patience: Early stopping patience
        learning_rate: Learning rate
        use_wandb: Whether to use wandb logging
    """
    
    # Setup wandb if requested
    if use_wandb:
        config = setup_wandb(project_name=project, run_name=name)
        # Update config with training parameters
        wandb.config.update({
            "model": model_path,
            "data": data_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "image_size": image_size,
            "device": device,
            "learning_rate": learning_rate,
            "patience": patience
        })
    
    # Load YOLO model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Train the model
    print("Starting training...")
    results = model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        project=project,
        name=name,
        patience=patience,
        lr0=learning_rate,
        save=True,
        plots=True,
        val=True,
        verbose=True,
        exist_ok=True,
        # Optimization settings
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=3,
        warmup_momentum=0.8,
        weight_decay=0.0005,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )
    
    # Log final results to wandb
    if use_wandb:
        # Log best metrics
        if hasattr(results, 'results_dict'):
            wandb.log(results.results_dict)
        
        # Log model artifacts
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        if best_model_path.exists():
            wandb.save(str(best_model_path))
            
        # Log training plots
        plots_dir = Path(results.save_dir)
        for plot_file in plots_dir.glob("*.png"):
            wandb.log({plot_file.stem: wandb.Image(str(plot_file))})
    
    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")
    
    if use_wandb:
        wandb.finish()
    
    return results


def main():
    """Main function to parse arguments and start training"""
    parser = argparse.ArgumentParser(description="Train YOLOv8 model with wandb logging")
    
    # Model and data arguments
    parser.add_argument("--model", type=str, default="yolov8m-pose.pt",
                       help="Path to model file (default: yolov8m-pose.pt)")
    parser.add_argument("--data", type=str, 
                       default="F:/NYCU/ultralytics/datasets/data.yaml",
                       help="Path to dataset YAML file")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    # Experiment settings
    parser.add_argument("--project", type=str, default="yolov8-pose-training",
                       help="Project name")
    parser.add_argument("--name", type=str, default="junction-detection-with-original",
                       help="Experiment name")
    parser.add_argument("--patience", type=int, default=50,
                       help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate")
    
    # Wandb settings
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    
    args = parser.parse_args()
    
    # Start training
    train_yolo(
        model_path=args.model,
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        learning_rate=args.lr,
        use_wandb=not args.no_wandb
    )


if __name__ == "__main__":
    main()