# 新文件: ultralytics/utils/enhanced_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .geometry_constraints import HomographyConstraint, SymmetryConstraint

class EnhancedKeypointLoss(nn.Module):
    """
    增強型關鍵點損失函數 - 方案一+五的核心實現
    
    組合損失:
    1. 基礎關鍵點損失 (去除面積依賴)
    2. 幾何一致性損失 (單應性約束)
    3. 對稱性損失
    4. 距離比例損失
    5. 角度約束損失
    """
    
    def __init__(self, sigmas=None, device='cpu', 
                 w_base=1.0, w_homo=0.5, w_sym=0.3, w_ratio=0.2, w_angle=0.2):
        super().__init__()
        self.device = device
        
        # 損失權重
        self.w_base = w_base      # 基礎關鍵點損失權重
        self.w_homo = w_homo      # 單應性約束權重
        self.w_sym = w_sym        # 對稱性約束權重  
        self.w_ratio = w_ratio    # 距離比例權重
        self.w_angle = w_angle    # 角度約束權重
        
        # 基礎損失組件
        if sigmas is not None:
            self.register_buffer('sigmas', sigmas)
        else:
            # 為羽球場交點使用均勻的sigma值
            self.register_buffer('sigmas', torch.ones(3, device=device) / 3)
        
        # 幾何約束模組
        self.homography_constraint = HomographyConstraint(device)
        self.symmetry_constraint = SymmetryConstraint(device)
        
    def forward(self, pred_kpts, gt_kpts, kpt_mask, 
                pred_classes=None, gt_classes=None, **kwargs):
        """
        計算增強型關鍵點損失
        
        Args:
            pred_kpts: [N, num_kpts, 3] 預測關鍵點 (x, y, conf)
            gt_kpts: [N, num_kpts, 3] 真實關鍵點 (x, y, vis)
            kpt_mask: [N, num_kpts] 關鍵點遮罩
            pred_classes: [N, num_kpts] 預測類別 (可選)
            gt_classes: [N, num_kpts] 真實類別 (可選)
        """
        # 1. 基礎關鍵點損失 (移除面積依賴)
        base_loss = self._compute_base_keypoint_loss(pred_kpts, gt_kpts, kpt_mask)
        
        # 2. 幾何一致性損失
        homo_loss = torch.tensor(0.0, device=self.device)
        if pred_classes is not None and gt_classes is not None:
            pred_xy = pred_kpts[..., :2]
            homo_loss = self.homography_constraint(pred_xy, gt_classes)
        
        # 3. 對稱性損失
        sym_loss = torch.tensor(0.0, device=self.device)
        if pred_classes is not None and gt_classes is not None:
            pred_xy = pred_kpts[..., :2]
            sym_loss = self.symmetry_constraint(pred_xy, gt_classes)
        
        # 4. 距離比例損失
        ratio_loss = self._compute_distance_ratio_loss(pred_kpts, gt_kpts, kpt_mask)
        
        # 5. 角度約束損失
        angle_loss = self._compute_angle_constraint_loss(pred_kpts, gt_kpts, kpt_mask)
        
        # 組合總損失
        total_loss = (self.w_base * base_loss + 
                     self.w_homo * homo_loss +
                     self.w_sym * sym_loss +
                     self.w_ratio * ratio_loss +
                     self.w_angle * angle_loss)
        
        return total_loss
    
    def _compute_base_keypoint_loss(self, pred_kpts, gt_kpts, kpt_mask):
        """計算基礎關鍵點損失 (去除面積依賴)"""
        # 計算歐氏距離
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + \
            (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        
        # 關鍵點損失因子
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        
        # 去除面積依賴的歸一化
        e = d / ((2 * self.sigmas).pow(2) * 2)  # 移除面積項
        
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()
    
    def _compute_distance_ratio_loss(self, pred_kpts, gt_kpts, kpt_mask):
        """計算距離比例損失"""
        # 簡化實現 - 確保相對距離保持一致
        batch_size = pred_kpts.shape[0]
        total_loss = 0.0
        valid_samples = 0
        
        for b in range(batch_size):
            valid_mask = kpt_mask[b] > 0
            if valid_mask.sum() < 2:
                continue
            
            pred_valid = pred_kpts[b][valid_mask, :2]
            gt_valid = gt_kpts[b][valid_mask, :2]
            
            # 計算距離比例一致性
            if len(pred_valid) >= 2:
                pred_dist = torch.pdist(pred_valid)
                gt_dist = torch.pdist(gt_valid)
                
                # 比例損失
                if len(pred_dist) > 0 and len(gt_dist) > 0:
                    ratio_loss = F.smooth_l1_loss(
                        pred_dist / (pred_dist.mean() + 1e-6),
                        gt_dist / (gt_dist.mean() + 1e-6)
                    )
                    total_loss += ratio_loss
                    valid_samples += 1
        
        return total_loss / max(valid_samples, 1)
    
    def _compute_angle_constraint_loss(self, pred_kpts, gt_kpts, kpt_mask):
        """計算角度約束損失"""
        # 簡化實現 - 檢查關鍵直角關係
        batch_size = pred_kpts.shape[0]
        total_loss = 0.0
        valid_angles = 0
        
        for b in range(batch_size):
            valid_mask = kpt_mask[b] > 0
            if valid_mask.sum() < 3:
                continue
                
            pred_valid = pred_kpts[b][valid_mask, :2]
            gt_valid = gt_kpts[b][valid_mask, :2]
            
            # 對於每三個點組合，檢查角度約束
            n_points = len(pred_valid)
            for i in range(n_points):
                for j in range(n_points):
                    for k in range(n_points):
                        if i != j and j != k and i != k:
                            # 計算角度
                            pred_angle = self._compute_angle_between_points(
                                pred_valid[i], pred_valid[j], pred_valid[k]
                            )
                            gt_angle = self._compute_angle_between_points(
                                gt_valid[i], gt_valid[j], gt_valid[k]
                            )
                            
                            angle_diff = (pred_angle - gt_angle).pow(2)
                            total_loss += angle_diff
                            valid_angles += 1
        
        return total_loss / max(valid_angles, 1)
    
    def _compute_angle_between_points(self, p1, p2, p3):
        """計算三點間的角度 (以p2為頂點)"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-6)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        
        return torch.acos(cos_angle)