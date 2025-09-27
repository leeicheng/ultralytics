from ultralytics.nn import DetectionModel
from ultralytics.utils import LOGGER
from .head import PointDetect


class CourtPointsModel(DetectionModel):
    """
    Court Points Detection Model for detecting line intersections.
    
    This model is specifically designed to detect and classify court line intersections
    into three categories: T-junction, Cross, and L-corner.
    
    Attributes:
        head_type (str): Type of detection head to use.
        point_detect_head (PointDetect): Custom point detection head.
    """

    def __init__(self, cfg='ultralytics/cfg/models/v8/court-points-n.yaml', ch=3, nc=None, verbose=True):
        """
        Initialize the CourtPointsModel.
        Args:
            cfg (str | dict): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes (default: 3 for T, Cross, L-corner).
            verbose (bool): Print additional information during initialization.
        """
        # Set default number of classes for court points
        if nc is None:
            nc = 3  # T-junction, Cross, L-corner
        self.nc = nc

        # Load the model configuration dictionary
        from ultralytics.nn.tasks import yaml_model_load
        cfg_dict = yaml_model_load(cfg) if isinstance(cfg, str) else cfg

        # Temporarily replace PointDetect with Detect to build the model
        head_config = None
        for i, layer in enumerate(cfg_dict["head"]):
            if layer[2] == "PointDetect":
                LOGGER.info(f"Temporarily replacing PointDetect with Detect for model creation.")
                head_config = layer
                layer[2] = "Detect"
                break
        
        # Initialize the parent model with the modified config
        super().__init__(cfg=cfg_dict, ch=ch, nc=nc, verbose=verbose)

        # Now, replace the created Detect head with the actual PointDetect head
        if head_config:
            # The last module is the detection head
            detect_head = self.model[-1]
            # Get the input channels from the 'conv' attribute of the first layer of each sequential module
            in_channels = [m[0].conv.in_channels for m in detect_head.cv2]
            
            # Create and substitute the PointDetect head
            point_detect_head = PointDetect(nc=self.nc, ch=in_channels)
            self.model[-1] = point_detect_head
            LOGGER.info(f"Successfully replaced Detect head with PointDetect head.")
        
        if verbose:
            LOGGER.info(f"CourtPointsModel initialized with {nc} classes (T-junction, Cross, L-corner)")

    def get_class_names(self):
        """
        Get the class names for court point detection.
        
        Returns:
            dict: Dictionary mapping class IDs to class names.
        """
        return {
            0: "T-junction",
            1: "Cross", 
            2: "L-corner"
        }

    def forward(self, x, *args, **kwargs):
        """
        Forward pass through the CourtPointsModel.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Model predictions.
        """
        return super().forward(x, *args, **kwargs)

    def predict_points(self, x, conf_threshold=0.5):
        """
        Predict court points with confidence filtering.
        
        Args:
            x (torch.Tensor): Input tensor.
            conf_threshold (float): Confidence threshold for detections.
            
        Returns:
            list: List of detected points with coordinates and classes.
        """
        predictions = self.forward(x)
        
        # This is a simplified prediction method
        # You would implement the full decoding logic based on your head's output format
        detected_points = []
        
        # Process predictions through the head's decode method
        if hasattr(self.model[-1], 'decode_predictions'):
            decoded = self.model[-1].decode_predictions(predictions)
            
            # Filter by confidence and convert to point format
            for layer_idx, (reg_pred, cls_pred) in enumerate(decoded):
                # Extract high-confidence detections
                conf_mask = cls_pred.max(dim=1)[0] > conf_threshold
                
                if conf_mask.any():
                    # Get coordinates and classes for high-confidence detections
                    coords = self.model[-1].get_point_coordinates(
                        reg_pred[conf_mask], 
                        stride=self.model[-1].stride[layer_idx]
                    )
                    classes = cls_pred[conf_mask].argmax(dim=1)
                    confidences = cls_pred[conf_mask].max(dim=1)[0]
                    
                    # Convert to point format
                    for coord, cls_id, conf in zip(coords, classes, confidences):
                        detected_points.append({
                            'x': float(coord[0]),
                            'y': float(coord[1]), 
                            'class_id': int(cls_id),
                            'class_name': self.get_class_names()[int(cls_id)],
                            'confidence': float(conf)
                        })
        
        return detected_points