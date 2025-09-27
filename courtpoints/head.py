import math
import torch
import torch.nn as nn
import numpy as np

from ultralytics.nn.modules import Conv, DFL


class PointDetect(nn.Module):
    """
    YOLO Point Detection head for court line intersection detection.
    
    This head is specifically designed for detecting court line intersections (T, Cross, L-corner)
    with enhanced feature extraction and multi-group processing capabilities.
    
    Attributes:
        nc (int): Number of classes (3 for T, Cross, L-corner).
        nl (int): Number of detection layers.
        num_groups (int): Number of feature groups (default: 10).
        reg_max (int): DFL channels for regression.
        feat_no (int): Number of feature channels per anchor.
        dxdy_no (int): Number of coordinate offset channels.
        no (int): Total number of outputs per anchor.
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=3, ch=()):  # detection layer
        """
        Initialize the PointDetect head.
        
        Args:
            nc (int): Number of classes (default: 3 for T, Cross, L-corner).
            ch (tuple): Channel dimensions from backbone feature maps.
        """
        super().__init__()
        self.nc = nc  # number of classes (3: T-junction, Cross, L-corner)
        self.nl = len(ch)  # number of detection layers
        self.num_groups = 1  # feature groups for enhanced point detection
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.feat_no = 2  # feature number per point
        self.dxdy_no = 2  # coordinate offset channels
        self.no = nc + self.reg_max * self.feat_no  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        # Calculate channel dimensions
        c2 = max((16, ch[0] // self.feat_no, self.reg_max * self.feat_no))
        c3 = max(ch[0], min(self.nc, 100))
        
        # Regression head (for point coordinates and features)
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), 
                Conv(c2, c2, 3),
                nn.Conv2d(c2, self.feat_no * self.reg_max * self.num_groups, 1)
            ) for x in ch
        )
        
        # Classification head (for point type classification)
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), 
                Conv(c3, c3, 3), 
                nn.Conv2d(c3, self.nc * self.num_groups, 1)
            ) for x in ch
        )
        
        # Additional feature head (for enhanced point detection)
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), 
                Conv(c3, c3, 3), 
                nn.Conv2d(c3, self.nc * 2 * self.num_groups, 1)
            ) for x in ch
        )
        
        # Distribution Focal Loss for regression
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """
        Forward pass through the PointDetect head.
        
        Args:
            x (list): List of feature maps from different scales.
            
        Returns:
            list: Processed feature maps for point detection.
        """
        # Process each detection layer
        for i in range(self.nl):
            # Combine regression and classification features
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            
        if self.training:  # Training path
            return x
        
        # Inference path - could add additional post-processing here
        return x

    def bias_init(self):
        """
        Initialize PointDetect biases for better training convergence.
        
        WARNING: requires stride availability.
        """
        m = self  # self.model[-1]  # PointDetect() module
        
        # Initialize biases for each detection layer
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            # Initialize regression head bias
            a[-1].bias.data[:] = 1.0  # point regression
            
            # Initialize classification head bias
            # Use log(5 / nc / (640 / stride)^2) for better initialization
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

    def decode_predictions(self, predictions):
        """
        Decode raw predictions into interpretable point detections.
        
        Args:
            predictions (torch.Tensor): Raw model predictions.
            
        Returns:
            tuple: Decoded coordinates, confidence scores, and class predictions.
        """
        # This method would be used to convert raw network outputs
        # into point coordinates and class probabilities
        # Implementation depends on your specific output format
        
        batch_size = predictions[0].shape[0]
        decoded_results = []
        
        for i, pred in enumerate(predictions):
            # Split prediction into regression and classification parts
            reg_pred = pred[:, :self.feat_no * self.reg_max * self.num_groups]
            cls_pred = pred[:, self.feat_no * self.reg_max * self.num_groups:]
            
            # Apply DFL to regression predictions if needed
            if self.reg_max > 1:
                # Reshape and apply DFL
                reg_pred = reg_pred.view(batch_size, self.num_groups, self.feat_no, self.reg_max, -1)
                reg_pred = self.dfl(reg_pred)
            
            # Apply sigmoid to classification predictions
            cls_pred = torch.sigmoid(cls_pred)
            
            decoded_results.append((reg_pred, cls_pred))
        
        return decoded_results

    def get_point_coordinates(self, reg_pred, stride, anchor_grid=None):
        """
        Convert regression predictions to actual point coordinates.
        
        Args:
            reg_pred (torch.Tensor): Regression predictions.
            stride (int): Current layer stride.
            anchor_grid (torch.Tensor, optional): Anchor grid for coordinate calculation.
            
        Returns:
            torch.Tensor: Point coordinates in original image space.
        """
        # Convert relative predictions to absolute coordinates
        # This is a simplified version - you may need to adjust based on your coordinate system
        
        if anchor_grid is None:
            # Create default grid
            grid_h, grid_w = reg_pred.shape[-2:]
            anchor_grid = torch.meshgrid(
                torch.arange(grid_h), 
                torch.arange(grid_w), 
                indexing='ij'
            )
            anchor_grid = torch.stack(anchor_grid, dim=0).float()
        
        # Scale to original image coordinates
        coords = (reg_pred + anchor_grid) * stride
        
        return coords

    def __repr__(self):
        return f"PointDetect(nc={self.nc}, nl={self.nl}, num_groups={self.num_groups}, reg_max={self.reg_max})"