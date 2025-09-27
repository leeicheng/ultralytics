import math
import torch
import torch.nn as nn
from copy import copy
import numpy as np

from ultralytics.utils.loss import DFLoss
from .courtpoints import CourtPointsModel
from .validator import CourtPointsValidator
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK, LOGGER
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils.tal import make_anchors


class CourtPointsLoss:
    """
    Loss function for Court Points Detection.
    """

    def __init__(self, model):
        self.device = next(model.parameters()).device
        self.model = model
        self.head = model.model[-1]
        self.nc = self.head.nc
        self.reg_max = self.head.reg_max

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dfl_loss = DFLoss(self.reg_max)

        self.assigner = CourtPointsAssigner(topk=10, nc=self.nc, alpha=0.5, beta=6.0)
        
        self.point_loss_weight = getattr(model.args, 'point_loss_weight', 2.0)
        self.class_loss_weight = getattr(model.args, 'class_loss_weight', 1.0)
        self.conf_loss_weight = getattr(model.args, 'conf_loss_weight', 1.0)

    def preprocess(self, targets, batch_size):
        """Prepares targets for loss computation, handling padding."""
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device), torch.zeros(batch_size, 0, 4, device=self.device)

        i = targets[:, 0]  # image index
        _, counts = i.unique(return_counts=True)
        n_max_gt = counts.max()
        
        padded_labels = torch.full((batch_size, n_max_gt, 1), self.nc, device=self.device, dtype=torch.long)
        padded_bboxes = torch.zeros(batch_size, n_max_gt, 4, device=self.device)

        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                padded_labels[j, :n] = targets[matches, 1:2].long()
                padded_bboxes[j, :n] = targets[matches, 2:]
        
        return padded_labels, padded_bboxes

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)  # point, class, conf
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.head.no, -1) for xi in feats], 2).split(
            (self.head.reg_max * self.head.feat_no, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        batch_size = pred_scores.shape[0]
        
        anchor_points, stride_tensor = make_anchors(feats, self.head.stride, 0.5)

        # Preprocess targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        gt_labels, gt_bboxes = self.preprocess(targets.to(self.device), batch_size)
        mask_gt = (gt_bboxes.sum(2, keepdim=True) > 0)

        pred_points = self.decode_points(anchor_points, pred_distri)

        target_labels, target_points, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_points.detach() * stride_tensor.unsqueeze(0),
            gt_labels, gt_bboxes, mask_gt)

        if fg_mask.sum():
            target_points_scaled = target_points[fg_mask] / stride_tensor[0]
            anchor_points_pos = anchor_points.unsqueeze(0).expand(batch_size, -1, -1)[fg_mask]
            target_offsets = target_points_scaled - anchor_points_pos
            pred_distri_pos = pred_distri[fg_mask]

            loss[0] = self.dfl_loss(pred_distri_pos.view(-1, self.reg_max), target_offsets.view(-1)).mean() * self.point_loss_weight
            loss[1] = self.bce(pred_scores[fg_mask], target_scores[fg_mask]).mean() * self.class_loss_weight

        loss[2] = self.bce(pred_scores, torch.zeros_like(pred_scores)).mean() * self.conf_loss_weight

        return loss.sum() * batch_size, loss.detach()

    def decode_points(self, anchor_points, pred_dist):
        b, a, c = pred_dist.shape
        pred_dist_reshaped = pred_dist.view(b, a, self.head.feat_no, self.reg_max)
        
        pred_dist_softmax = pred_dist_reshaped.softmax(dim=3)
        dfl_range = torch.arange(self.reg_max, device=self.device, dtype=torch.float).view(1, 1, 1, -1)
        pred_offsets = (pred_dist_softmax * dfl_range).sum(dim=3)

        decoded_points = pred_offsets + anchor_points.unsqueeze(0)
        return decoded_points

class CourtPointsAssigner:
    def __init__(self, topk=10, nc=1, alpha=0.5, beta=6.0):
        self.topk = topk
        self.nc = nc
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def __call__(self, pd_scores, pd_points, gt_labels, gt_bboxes, mask_gt):
        bs = pd_scores.size(0)
        n_max_gt = gt_bboxes.size(1)

        if n_max_gt == 0:
            device = gt_labels.device
            return (torch.zeros(bs, 0, dtype=torch.long, device=device),
                    torch.zeros(bs, 0, 2, dtype=torch.float, device=device),
                    torch.zeros(bs, 0, self.nc, dtype=torch.float, device=device),
                    torch.zeros(bs, 0, dtype=torch.bool, device=device),
                    None)

        assigned_labels = torch.full((bs, pd_points.shape[1]), self.nc, dtype=torch.long, device=pd_scores.device)
        assigned_points = torch.zeros((bs, pd_points.shape[1], 2), device=pd_scores.device)
        assigned_scores = torch.zeros((bs, pd_points.shape[1], self.nc), device=pd_scores.device)
        fg_mask = torch.zeros(bs, pd_points.shape[1], dtype=torch.bool, device=pd_scores.device)

        for b in range(bs):
            gt_mask_b = mask_gt[b].squeeze()
            gt_labels_b = gt_labels[b][gt_mask_b]
            gt_points_b = gt_bboxes[b][:, :2][gt_mask_b]
            
            if gt_labels_b.numel() == 0:
                continue

            pd_scores_b = pd_scores[b]
            pd_points_b = pd_points[b]

            distances = torch.cdist(gt_points_b, pd_points_b)
            distance_cost = 1.0 / (1.0 + distances)
            
            cls_preds = pd_scores_b[:, gt_labels_b.squeeze()]
            cls_cost = cls_preds.pow(self.alpha)
            
            cost_matrix = cls_cost.t() * distance_cost.pow(self.beta)

            topk_ious, topk_idx = torch.topk(cost_matrix, self.topk, dim=1)
            
            matched_gt_idx = torch.arange(gt_labels_b.shape[0], device=pd_scores.device).repeat_interleave(self.topk)
            matched_pd_idx = topk_idx.view(-1)
            
            fg_mask[b, matched_pd_idx] = True
            assigned_labels[b, matched_pd_idx] = gt_labels_b[matched_gt_idx].squeeze()
            assigned_points[b, matched_pd_idx] = gt_points_b[matched_gt_idx]
            assigned_scores[b, matched_pd_idx] = torch.nn.functional.one_hot(gt_labels_b[matched_gt_idx].squeeze().long(), self.nc).float()

        return assigned_labels, assigned_points, assigned_scores, fg_mask, None


class CourtPointsTrainer(DetectionTrainer):
    def __init__(self, cfg='ultralytics/cfg/models/v8/court-points-n.yaml', overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)
        self.loss_names = ('point_loss', 'class_loss', 'conf_loss')

    def init_criterion(self):
        return CourtPointsLoss(self.model)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CourtPointsModel(
            cfg=cfg, 
            nc=self.data["nc"], 
            ch=self.data.get("channels", 3), 
            verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def get_validator(self):
        self.loss_names = ('point_loss', 'class_loss', 'conf_loss')
        return CourtPointsValidator(
            dataloader=self.test_loader, 
            save_dir=self.save_dir, 
            args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images and labels for training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)
        return batch

    def plot_training_samples(self, batch, ni):
        pass

    def plot_training_labels(self):
        pass

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.names = {0: "T-junction", 1: "Cross", 2: "L-corner"}
        if RANK in {-1, 0}:
            LOGGER.info(f"Model set with {self.model.nc} court point classes: {list(self.model.names.values())}")

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch", "GPU_mem", *self.loss_names, "Instances", "Size")
