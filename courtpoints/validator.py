import torch
import numpy as np
import json
from pathlib import Path
from copy import deepcopy
import cv2

from .datasets_loader import DatasetsLoader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import DetMetrics, ConfusionMatrix
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils.torch_utils import de_parallel


class CourtPointsMetrics:
    """
    Court Points Detection specific metrics.
    
    Provides specialized evaluation metrics for court line intersection detection,
    including point-wise accuracy (recall) at various distance thresholds.
    """
    
    def __init__(self, save_dir=Path('.'), thresholds=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.distance_thresholds = thresholds or [5, 10, 15, 20, 25]  # Pixel distance thresholds
        self.stats = []  # List of (prediction_class, ground_truth_class, distance)

    def update(self, matched_stats):
        """Update metrics with new matched statistics."""
        self.stats.extend(matched_stats)
        
    def compute(self):
        """Compute final metrics based on all collected stats."""
        stats = np.array(self.stats) if self.stats else np.empty((0, 3))
        
        results = {}
        if stats.shape[0] == 0:
            for thresh in self.distance_thresholds:
                results[f'PA@{thresh}px'] = 0.0
            return results

        # Total number of ground truth points for each class
        # This is a simplification; a more robust way would be to count all GTs from the dataset
        # For now, we count unique GTs that had a prediction matched to them.
        # A proper implementation should get total GT count from the validator.
        
        for thresh in self.distance_thresholds:
            # A true positive is a correct class match AND distance is within threshold
            correct_class_mask = stats[:, 0] == stats[:, 1]
            within_dist_mask = stats[:, 2] <= thresh
            
            tp = np.sum(correct_class_mask & within_dist_mask)
            
            # Total predictions made
            total_preds = stats.shape[0]
            
            # For point accuracy (recall), we need total number of ground truths.
            # This is passed from the validator.
            total_gt = self.total_gt
            
            recall = tp / (total_gt + 1e-9)
            results[f'PA@{thresh}px'] = recall
            
        return results

    def set_total_gt(self, count):
        """Set the total number of ground truth points."""
        self.total_gt = count


class CourtPointsValidator(BaseValidator):
    """
    Validator for Court Points Detection.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize CourtPointsValidator."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        
        self.args.task = "detect" # Use detect task for base functionalities
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.court_metrics = CourtPointsMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.distance_threshold = getattr(args, 'distance_threshold', 25.0) if args else 25.0
        self.class_names = {0: "T-junction", 1: "Cross", 2: "L-corner"}

    def get_dataloader(self, dataset_path, batch_size=32):
        """Get dataloader for validation."""
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        dataset = build_yolo_dataset(self.args, dataset_path, batch_size, self.data, mode="val", stride=gs)
        return build_dataloader(dataset, batch_size, self.args.workers * 2, shuffle=False, rank=-1)

    def preprocess(self, batch):
        """Preprocess batch for validation."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255.0
        for k in ["batch_idx", "cls", "bboxes"]:
            if k in batch:
                batch[k] = batch[k].to(self.device)
        return batch

    def postprocess(self, preds):
        """Apply Non-Maximum Suppression."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=[],
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

    def init_metrics(self, model):
        """Initialize metrics for evaluation."""
        self.metrics.names = model.names
        self.confusion_matrix = ConfusionMatrix(nc=model.nc)
        self.seen = 0
        self.jdict = []
        self.stats = []
        self.court_metrics = CourtPointsMetrics(save_dir=self.save_dir)

    def update_metrics(self, preds, batch):
        """Update metrics with predictions and ground truth."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            idx = batch["batch_idx"] == si
            cls = batch["cls"][idx]
            bbox = batch["bboxes"][idx]
            nl = len(cls)
            
            if npr == 0:
                continue

            # Post-process and scale predictions
            predn = pred.clone()
            ops.scale_boxes(batch["img"][si].shape[1:], predn[:, :4], batch["ori_shape"][si])
            
            # Ground truths
            tbox = ops.xywh2xyxy(bbox)
            labelsn = torch.cat((cls, tbox), 1)

            # Process for standard mAP metrics
            correct_bboxes = self._process_batch(predn, labelsn, self.iouv)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))

            # Process for custom court points metrics
            matched_stats = self._process_batch_for_points(predn, labelsn)
            self.court_metrics.update(matched_stats)

    def _process_batch_for_points(self, detections, labels):
        """Match predictions to ground truth based on distance for point metrics."""
        if detections.shape[0] == 0 or labels.shape[0] == 0:
            return []

        gt_points = labels[:, 1:3] # Use center of GT box as the point
        pred_points = detections[:, :2] # Use center of pred box as the point

        distances = torch.cdist(pred_points, gt_points)
        
        # Find the closest GT for each prediction
        min_dist, gt_idx = distances.min(dim=1)

        matched_stats = []
        # Only consider predictions within a reasonable distance threshold
        for i, dist in enumerate(min_dist):
            if dist < self.distance_threshold:
                pred_cls = detections[i, 5].item()
                gt_cls = labels[gt_idx[i], 0].item()
                matched_stats.append([pred_cls, gt_cls, dist.item()])
        
        return matched_stats

    def get_stats(self):
        """Get validation statistics."""
        # Standard mAP stats
        stats = self.metrics.process(self.stats)
        self.nt = self.metrics.nt
        
        # Custom court points stats
        self.court_metrics.set_total_gt(self.nt.sum())
        court_stats = self.court_metrics.compute()
        stats.update(court_stats)
        return stats

    def print_results(self):
        """Print validation results."""
        super().print_results() # Print standard mAP results
        
        # Print custom court points results
        LOGGER.info("\nCourt Points Specific Metrics:")
        pf = '%22s' + '%11.3g'
        for metric, value in self.get_stats().items():
            if 'PA@' in metric:
                LOGGER.info(pf % (metric, value))
