"""
Court Points Detection Task Module.

This module defines the task mapping and main interface for court line intersection detection.
Supports detection and classification of T-junction, Cross, and L-corner intersections.
"""

from .courtpoints import CourtPointsModel
from .predictor import CourtPointsPredictor
from .trainer import CourtPointsTrainer
from .validator import CourtPointsValidator

# Task mapping for court points detection
TASK_MAP = {
    'courtpoints': {
        'model': CourtPointsModel,
        'trainer': CourtPointsTrainer,
        'predictor': CourtPointsPredictor,
        'validator': CourtPointsValidator
    }
}

# Alternative list format for backward compatibility
TASK_MAP_LIST = {
    'courtpoints': [
        CourtPointsModel,
        CourtPointsTrainer,
        CourtPointsPredictor,
        CourtPointsValidator
    ]
}


class CourtPoints:
    """
    Main interface class for Court Points Detection.
    
    This class provides a high-level interface for working with court line intersection detection,
    including model loading, training, prediction, and validation workflows.
    
    Attributes:
        model (CourtPointsModel): The detection model.
        task_type (str): Task identifier ('courtpoints').
        class_names (dict): Mapping of class IDs to class names.
    """
    
    def __init__(self, model_path=None, device='auto'):
        """
        Initialize CourtPoints detection system.
        
        Args:
            model_path (str, optional): Path to pre-trained model weights.
            device (str): Device to run inference on ('auto', 'cpu', 'cuda', etc.).
        """
        self.task_type = 'courtpoints'
        self.class_names = {
            0: "T-junction",
            1: "Cross",
            2: "L-corner"
        }
        self.model = None
        self.device = device
        
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path, device=None):
        """
        Load a pre-trained court points detection model.
        
        Args:
            model_path (str): Path to model weights.
            device (str, optional): Device to load model on.
        """
        if device is None:
            device = self.device
            
        self.model = CourtPointsModel()
        # Add model loading logic here
        print(f"Loading CourtPoints model from {model_path} on device {device}")

    def train(self, data_config, model_config=None, **kwargs):
        """
        Train a court points detection model.
        
        Args:
            data_config (str | dict): Dataset configuration.
            model_config (str | dict, optional): Model configuration.
            **kwargs: Additional training arguments.
        """
        trainer = CourtPointsTrainer()
        # Add training logic here
        print(f"Training CourtPoints model with data config: {data_config}")

    def predict(self, source, **kwargs):
        """
        Run prediction on images or video.
        
        Args:
            source (str | list): Input source (image path, video path, or list of images).
            **kwargs: Additional prediction arguments.
            
        Returns:
            list: Detection results.
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first using load_model()")
            
        predictor = CourtPointsPredictor(model=self.model, **kwargs)
        results = predictor.predict(source)
        return results

    def validate(self, data_config, model_path=None, **kwargs):
        """
        Validate model performance on test dataset.
        
        Args:
            data_config (str | dict): Validation dataset configuration.
            model_path (str, optional): Path to model weights.
            **kwargs: Additional validation arguments.
            
        Returns:
            dict: Validation metrics.
        """
        validator = CourtPointsValidator()
        # Add validation logic here
        print(f"Validating CourtPoints model with data config: {data_config}")

    def get_supported_tasks(self):
        """
        Get list of supported task types.
        
        Returns:
            list: List of supported tasks.
        """
        return list(TASK_MAP.keys())

    def get_class_info(self):
        """
        Get information about detection classes.
        
        Returns:
            dict: Class information including names and descriptions.
        """
        return {
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'descriptions': {
                0: "T-junction: Intersection where one line ends at the middle of another line",
                1: "Cross: Intersection where two lines cross at their middle points",
                2: "L-corner: Intersection where two lines meet at their endpoints"
            }
        }

    def __repr__(self):
        return f"CourtPoints(task_type='{self.task_type}', num_classes={len(self.class_names)})"


# Export main components
__all__ = [
    'CourtPoints',
    'CourtPointsModel', 
    'CourtPointsTrainer',
    'CourtPointsPredictor', 
    'CourtPointsValidator',
    'TASK_MAP',
    'TASK_MAP_LIST'
]




