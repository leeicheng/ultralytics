"""
Court Points Detection Module for Ultralytics YOLO.

This module provides specialized detection capabilities for court line intersections,
specifically designed to detect and classify T-junctions, Cross intersections, and L-corners
in sports court imagery.

Key Components:
    - CourtPointsModel: Custom detection model with PointDetect head
    - CourtPointsTrainer: Training pipeline for court point detection
    - CourtPointsPredictor: Inference engine with visualization capabilities  
    - CourtPointsValidator: Validation and evaluation metrics
    - PointDetect: Custom detection head optimized for point-like objects

Example Usage:
    ```python
    from courtpoints import CourtPoints, TASK_MAP
    
    # Initialize court points detection
    cp = CourtPoints()
    
    # Load model and predict
    cp.load_model('court_points_model.pt')
    results = cp.predict('court_image.jpg')
    
    # Access task components directly
    model = TASK_MAP['courtpoints']['model']()
    ```

Classes:
    CourtPoints: Main interface class
    CourtPointsModel: Detection model
    CourtPointsTrainer: Training pipeline
    CourtPointsPredictor: Inference engine
    CourtPointsValidator: Validation metrics
    PointDetect: Custom detection head
"""

from .tasks import (
    CourtPoints,
    CourtPointsModel,
    CourtPointsTrainer, 
    CourtPointsPredictor,
    CourtPointsValidator,
    TASK_MAP,
    TASK_MAP_LIST
)

from .head import PointDetect
from .courtpoints import CourtPointsModel
from .predictor import CourtPointsPredictor, CourtPointPrediction
from .trainer import CourtPointsTrainer
from .validator import CourtPointsValidator
from .datasets_loader import DatasetsLoader, PreprocessLayer

# Register PointDetect in the global namespace where YOLO can find it
def _register_point_detect():
    """Register PointDetect class in appropriate namespaces for YOLO discovery."""
    try:
        # Import the tasks module where parsing happens
        import ultralytics.nn.tasks
        
        # Register PointDetect in the tasks module globals
        ultralytics.nn.tasks.__dict__['PointDetect'] = PointDetect
        
        # Also register in the main nn module for completeness
        import ultralytics.nn
        ultralytics.nn.__dict__['PointDetect'] = PointDetect
        
        # And in the main ultralytics module
        import ultralytics
        ultralytics.__dict__['PointDetect'] = PointDetect
        
    except ImportError as e:
        import warnings
        warnings.warn(f"Could not register PointDetect: {e}")

# Call registration function
_register_point_detect()

# Version information
__version__ = "1.0.0"
__author__ = "Court Points Detection Team"

# Export all main components
__all__ = [
    # Main interface
    'CourtPoints',
    
    # Core components
    'CourtPointsModel',
    'CourtPointsTrainer',
    'CourtPointsPredictor', 
    'CourtPointsValidator',
    
    # Detection head
    'PointDetect',
    
    # Data structures
    'CourtPointPrediction',
    'DatasetsLoader',
    'PreprocessLayer',
    
    # Task mappings
    'TASK_MAP',
    'TASK_MAP_LIST',
    
    # Version info
    '__version__',
    '__author__'
]

# Module metadata
_TASK_INFO = {
    'name': 'courtpoints',
    'description': 'Court line intersection detection and classification',
    'classes': ['T-junction', 'Cross', 'L-corner'],
    'num_classes': 3,
    'input_format': 'RGB images',
    'output_format': 'Bounding boxes with point coordinates and class predictions'
}

def get_task_info():
    """
    Get information about the court points detection task.
    
    Returns:
        dict: Task metadata and configuration information.
    """
    return _TASK_INFO.copy()

def list_available_models():
    """
    List available pre-trained models for court points detection.
    
    Returns:
        list: List of available model configurations.
    """
    return [
        'court-points-n.yaml',  # Nano model
        'court-points-s.yaml',  # Small model  
        'court-points-m.yaml',  # Medium model
        'court-points-l.yaml',  # Large model
    ]

def create_model(model_size='s', num_classes=3, **kwargs):
    """
    Create a court points detection model with specified configuration.
    
    Args:
        model_size (str): Model size ('n', 's', 'm', 'l').
        num_classes (int): Number of detection classes (default: 3).
        **kwargs: Additional model arguments.
        
    Returns:
        CourtPointsModel: Initialized model instance.
    """
    config_path = f'court-points-{model_size}.yaml'
    return CourtPointsModel(cfg=config_path, nc=num_classes, **kwargs)