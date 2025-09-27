"""
Example usage of Court Points Detection module.

This script demonstrates how to use the court points detection system
for training, prediction, and validation tasks.
"""

import torch
from pathlib import Path

# Import the court points detection components
from . import (
    CourtPoints,
    CourtPointsModel,
    CourtPointsTrainer,
    CourtPointsPredictor,
    CourtPointsValidator,
    TASK_MAP,
    TASK_MAP_LIST,
    get_task_info,
    create_model
)


def example_task_map_usage():
    """Demonstrate how to use TASK_MAP for component access."""
    print("=== TASK_MAP Usage Example ===")
    
    # Access components via TASK_MAP (dictionary format)
    model_class = TASK_MAP['courtpoints']['model']
    trainer_class = TASK_MAP['courtpoints']['trainer']
    predictor_class = TASK_MAP['courtpoints']['predictor']
    validator_class = TASK_MAP['courtpoints']['validator']
    
    print(f"Model class: {model_class}")
    print(f"Trainer class: {trainer_class}")
    print(f"Predictor class: {predictor_class}")
    print(f"Validator class: {validator_class}")
    
    # Alternative: Use TASK_MAP_LIST (list format for backward compatibility)
    components = TASK_MAP_LIST['courtpoints']
    model_class_alt, trainer_class_alt, predictor_class_alt, validator_class_alt = components
    
    print(f"\nAlternative access:")
    print(f"Model class: {model_class_alt}")
    print(f"Trainer class: {trainer_class_alt}")
    print(f"Predictor class: {predictor_class_alt}")
    print(f"Validator class: {validator_class_alt}")


def example_model_creation():
    """Demonstrate model creation with custom PointDetect head."""
    print("\n=== Model Creation Example ===")
    
    # Create model using the factory function
    model = create_model(model_size='s', num_classes=3)
    print(f"Created model: {model}")
    
    # Direct model creation
    model_direct = CourtPointsModel(nc=3, verbose=True)
    print(f"Direct model creation: {model_direct}")
    
    # Access class names
    class_info = model_direct.get_class_names()
    print(f"Class names: {class_info}")


def example_high_level_interface():
    """Demonstrate high-level CourtPoints interface."""
    print("\n=== High-Level Interface Example ===")
    
    # Initialize court points detection system
    cp = CourtPoints(device='cpu')
    print(f"Initialized: {cp}")
    
    # Get task information
    task_info = get_task_info()
    print(f"Task info: {task_info}")
    
    # Get class information
    class_info = cp.get_class_info()
    print(f"Class info: {class_info}")


def example_training_workflow():
    """Demonstrate training workflow."""
    print("\n=== Training Workflow Example ===")
    
    # Get trainer from TASK_MAP
    trainer_class = TASK_MAP['courtpoints']['trainer']
    trainer = trainer_class()
    
    print(f"Trainer initialized: {trainer}")
    print(f"Loss names: {trainer.loss_names}")
    
    # Example training configuration
    training_config = {
        'data': 'court_points_dataset.yaml',
        'epochs': 100,
        'batch_size': 16,
        'img_size': 640,
        'lr': 0.001
    }
    
    print(f"Training config example: {training_config}")


def example_prediction_workflow():
    """Demonstrate prediction workflow."""
    print("\n=== Prediction Workflow Example ===")
    
    # Get predictor from TASK_MAP
    predictor_class = TASK_MAP['courtpoints']['predictor']
    
    # Initialize with custom settings
    predictor_args = {
        'conf': 0.5,
        'iou': 0.3,
        'use_nms': True,
        'custom_preprocess': False
    }
    
    # Note: In real usage, you'd pass a loaded model
    # predictor = predictor_class(model=loaded_model, **predictor_args)
    
    print(f"Predictor class: {predictor_class}")
    print(f"Prediction args example: {predictor_args}")


def example_validation_workflow():
    """Demonstrate validation workflow."""
    print("\n=== Validation Workflow Example ===")
    
    # Get validator from TASK_MAP
    validator_class = TASK_MAP['courtpoints']['validator']
    validator = validator_class()
    
    print(f"Validator initialized: {validator}")
    
    # Example validation configuration
    validation_config = {
        'data': 'court_points_val.yaml',
        'batch_size': 32,
        'save_dir': './validation_results'
    }
    
    print(f"Validation config example: {validation_config}")


def main():
    """Run all examples."""
    print("Court Points Detection - Usage Examples")
    print("=" * 50)
    
    # Run all example functions
    example_task_map_usage()
    example_model_creation()
    example_high_level_interface()
    example_training_workflow()
    example_prediction_workflow()
    example_validation_workflow()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")


if __name__ == '__main__':
    main()