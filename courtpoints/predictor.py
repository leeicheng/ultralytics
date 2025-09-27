
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops, LOGGER
import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import cv2
import os


@dataclass
class CourtPointPrediction:
    """Data structure for a single court point prediction."""
    x: float
    y: float
    conf: float
    class_id: int
    class_name: str


class CourtPointsPredictor(BasePredictor):
    """
    A predictor class for detecting court line intersection points (T, Cross, L-corner).
    
    This predictor specializes in detecting and classifying three types of court line intersections:
    - T-junction (class 0): Intersection where one line ends at the middle of another line
    - Cross (class 1): Intersection where two lines cross at their middle points  
    - L-corner (class 2): Intersection where two lines meet at their endpoints
    
    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The court points detection model.
        
    Methods:
        postprocess: Process raw model predictions into court point detection results.
        construct_results: Build Results objects from processed predictions.
        construct_result: Create a single Result object from a prediction.
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """
        Post-process predictions for court point detection.
        
        Processes raw model outputs into court point detections with confidence scores
        and class predictions. Supports both NMS and non-NMS modes.
        
        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs: Additional keyword arguments.
            
        Returns:
            (list): List of Results objects containing court point detections.
        """
        # Configuration parameters
        use_nms = getattr(self.args, 'use_nms', True)
        conf_threshold = getattr(self.args, 'conf', 0.5)
        iou_threshold = getattr(self.args, 'iou', 0.3)  # Lower IoU for point detections
        
        if use_nms:
            # Apply standard YOLO NMS for bounding box detections
            preds = ops.non_max_suppression(
                preds,
                conf_threshold,
                iou_threshold,
                self.args.classes,
                self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=len(self.model.names) if hasattr(self.model, 'names') else 3,
            )
        else:
            # Custom processing for court points (similar to TrackNet approach)
            preds = self._process_court_points_output(preds, conf_threshold)
        
        # Convert torch tensor images to numpy if needed
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
            
        # Construct results objects
        results = self.construct_results(preds, img, orig_imgs, **kwargs)
        
        return results

    def _process_court_points_output(self, preds, conf_threshold=0.5):
        """
        Process raw model output for court point detection without standard NMS.
        
        This method is inspired by TrackNet's approach for processing point detections.
        
        Args:
            preds (torch.Tensor): Raw model predictions.
            conf_threshold (float): Confidence threshold for detections.
            
        Returns:
            (list): List of processed predictions for each image in the batch.
        """
        # Assuming model output format similar to YOLO detection
        # This is a simplified version - you may need to adjust based on your model's actual output
        batch_results = []
        
        for batch_idx in range(len(preds) if isinstance(preds, list) else 1):
            pred = preds[batch_idx] if isinstance(preds, list) else preds[0]
            
            # Filter predictions by confidence threshold
            if len(pred) > 0:
                conf_mask = pred[:, 4] >= conf_threshold
                filtered_pred = pred[conf_mask]
                
                # Convert to court point predictions
                if len(filtered_pred) > 0:
                    # Extract coordinates, confidence, and class
                    # Assuming format: [x_center, y_center, width, height, confidence, class_id, ...]
                    results = []
                    for detection in filtered_pred:
                        x, y, w, h = detection[:4].cpu().numpy()
                        conf = detection[4].cpu().item()
                        class_id = int(detection[5].cpu().item()) if len(detection) > 5 else 0
                        
                        # Create court point prediction
                        point_pred = CourtPointPrediction(
                            x=float(x),
                            y=float(y),
                            conf=float(conf),
                            class_id=class_id,
                            class_name=self._get_class_name(class_id)
                        )
                        results.append(point_pred)
                    
                    batch_results.append(filtered_pred)
                else:
                    # No detections above threshold
                    batch_results.append(torch.zeros((0, 6), device=pred.device))
            else:
                # No detections
                batch_results.append(torch.zeros((0, 6), device=pred.device if torch.is_tensor(pred) else 'cpu'))
        
        return batch_results

    def _get_class_name(self, class_id):
        """Get class name from class ID."""
        class_names = {0: 'T-junction', 1: 'Cross', 2: 'L-corner'}
        return class_names.get(class_id, f'class_{class_id}')

    def construct_results(self, preds, img, orig_imgs, **kwargs):
        """
        Construct a list of Results objects from court point predictions.
        
        Args:
            preds (List[torch.Tensor]): List of predicted court points for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.
            
        Returns:
            (List[Results]): List of Results objects containing court point information.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct a single Results object from court point predictions for one image.
        
        For court points, we treat each detection as a small bounding box around the
        intersection point, but the key information is the center coordinate and class.
        
        Args:
            pred (torch.Tensor): Predicted court points with shape (N, 6) where N is 
                               the number of detections and each detection is [x, y, w, h, conf, class_id].
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.
            
        Returns:
            (Results): Results object containing court point detections with scaled coordinates.
        """
        if len(pred) == 0:
            # No detections found, return empty results
            return Results(orig_img, path=img_path, names=self.model.names, boxes=pred)
        
        # Scale bounding boxes from model input size to original image size
        # pred[:, :4] contains [x_center, y_center, width, height] in model coordinates
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        
        # For court points, the bounding box represents the detection area around the intersection
        # The center of the box is the predicted intersection point location
        # pred format: [x_center, y_center, width, height, confidence, class_id]
        
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])

    def preprocess(self, im):
        """
        Preprocess input images for court point detection.
        
        Supports both standard YOLO preprocessing and custom court-specific preprocessing
        including grayscale conversion and median subtraction.
        
        Args:
            im: Input image(s) to preprocess.
            
        Returns:
            torch.Tensor: Preprocessed image tensor ready for model inference.
        """
        # Check if custom preprocessing is enabled
        use_custom_preprocess = getattr(self.args, 'custom_preprocess', False)
        
        if use_custom_preprocess:
            return self._custom_preprocess(im)
        else:
            # Standard YOLO preprocessing
            im = super().preprocess(im)
            
            # Apply court-specific preprocessing (grayscale conversion) if model supports it
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'preprocess_layer'):
                im = self.model.model.preprocess_layer(im)
                
        return im

    def _custom_preprocess(self, im):
        """
        Custom preprocessing for court point detection inspired by TrackNet.
        
        Applies grayscale conversion, median subtraction, and normalization.
        
        Args:
            im: Input image(s) to preprocess.
            
        Returns:
            torch.Tensor: Custom preprocessed image tensor.
        """
        # Convert to tensor if needed
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = torch.from_numpy(im)
            im = im.permute(2, 0, 1).contiguous()  # HWC -> CHW
        
        # Move to device
        im = im.to(self.device, dtype=torch.float32, non_blocking=True)
        
        # Apply median subtraction (inspired by TrackNet)
        median = im.median(dim=0).values  # shape: (H, W)
        im = im - median.unsqueeze(0)
        
        # Clamp and normalize
        im = torch.clamp(im, 0, 255) / 255.0
        
        # Add batch dimension if needed
        if im.dim() == 3:
            im = im.unsqueeze(0)
        
        # Convert to half precision if model supports it
        if hasattr(self.model, 'fp16') and self.model.fp16:
            im = im.half()
            
        return im

    def cpu_preprocess(self, im):
        """
        CPU version of preprocessing for court point detection.
        
        Useful for multi-threading scenarios where GPU preprocessing might cause conflicts.
        
        Args:
            im: Input image(s) to preprocess.
            
        Returns:
            torch.Tensor: CPU preprocessed image tensor.
        """
        if not isinstance(im, torch.Tensor):
            im = torch.from_numpy(im).permute(2, 0, 1).contiguous().float()
        else:
            im = im.float()

        # Apply median subtraction on CPU
        median = im.median(dim=0).values  # (H, W)
        im = (im - median.unsqueeze(0)).clamp(0, 255) / 255.0

        return im

    def save_court_points_visualization(self, results, save_dir=None):
        """
        Save court points detection results with visualization.
        
        Creates annotated images showing detected court points with class labels and confidence scores.
        
        Args:
            results: Detection results from the predictor.
            save_dir (str, optional): Directory to save visualization images.
        """
        if save_dir is None:
            save_dir = getattr(self, 'save_dir', './runs/predict')
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(results):
            if hasattr(result, 'orig_img') and hasattr(result, 'boxes'):
                img = result.orig_img.copy()
                
                if result.boxes is not None and len(result.boxes) > 0:
                    # Draw bounding boxes and points
                    for box in result.boxes.data:
                        x, y, w, h, conf, class_id = box[:6].cpu().numpy()
                        
                        # Draw bounding box
                        x1, y1 = int(x - w/2), int(y - h/2)
                        x2, y2 = int(x + w/2), int(y + h/2)
                        
                        # Color based on class
                        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # T, Cross, L
                        color = colors[int(class_id)] if int(class_id) < len(colors) else (255, 255, 255)
                        
                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw center point
                        cv2.circle(img, (int(x), int(y)), radius=3, color=color, thickness=-1)
                        
                        # Add label
                        label = f"{self._get_class_name(int(class_id))}: {conf:.2f}"
                        cv2.putText(img, label, (int(x) + 5, int(y) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Save image
                save_path = save_dir / f'court_points_{i:04d}.jpg'
                cv2.imwrite(str(save_path), img)
                LOGGER.info(f"Saved court points visualization to {save_path}")

    def export_court_points_csv(self, results, save_path=None):
        """
        Export court points detection results to CSV format.
        
        Args:
            results: Detection results from the predictor.
            save_path (str, optional): Path to save CSV file.
        """
        if save_path is None:
            save_path = './court_points_results.csv'
        
        import csv
        
        with open(save_path, 'w', newline='') as csvfile:
            fieldnames = ['image_id', 'x', 'y', 'confidence', 'class_id', 'class_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for img_id, result in enumerate(results):
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes.data:
                        x, y, w, h, conf, class_id = box[:6].cpu().numpy()
                        
                        writer.writerow({
                            'image_id': img_id,
                            'x': float(x),
                            'y': float(y),
                            'confidence': float(conf),
                            'class_id': int(class_id),
                            'class_name': self._get_class_name(int(class_id))
                        })
        
        LOGGER.info(f"Exported court points results to {save_path}")

    def get_court_points_summary(self, results):
        """
        Generate a summary of court points detection results.
        
        Args:
            results: Detection results from the predictor.
            
        Returns:
            dict: Summary statistics of the detection results.
        """
        summary = {
            'total_images': len(results),
            'total_detections': 0,
            'detections_by_class': {'T-junction': 0, 'Cross': 0, 'L-corner': 0},
            'average_confidence': 0.0,
            'confidence_by_class': {'T-junction': [], 'Cross': [], 'L-corner': []}
        }
        
        all_confidences = []
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes.data:
                    conf = box[4].cpu().item()
                    class_id = int(box[5].cpu().item()) if len(box) > 5 else 0
                    class_name = self._get_class_name(class_id)
                    
                    summary['total_detections'] += 1
                    all_confidences.append(conf)
                    
                    if class_name in summary['detections_by_class']:
                        summary['detections_by_class'][class_name] += 1
                        summary['confidence_by_class'][class_name].append(conf)
        
        # Calculate average confidences
        if all_confidences:
            summary['average_confidence'] = np.mean(all_confidences)
            
            for class_name in summary['confidence_by_class']:
                if summary['confidence_by_class'][class_name]:
                    summary['confidence_by_class'][class_name] = np.mean(summary['confidence_by_class'][class_name])
                else:
                    summary['confidence_by_class'][class_name] = 0.0
        
        return summary