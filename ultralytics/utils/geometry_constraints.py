# 新文件: ultralytics/utils/geometry_constraints.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class BadmintonCourtGeometry:
    """羽球場標準尺寸和幾何約束"""
    
    # 羽球場標準尺寸 (米)
    COURT_LENGTH = 13.4
    COURT_WIDTH = 6.1
    SERVICE_LINE_DISTANCE = 1.98  # 從網線到發球線  
    DOUBLES_SIDELINE_DISTANCE = 0.46  # 單雙打邊線距離
    
    # 羽球場完整線條結構定義 (正規化座標，以場地中心為原點)
    # 基於標準羽球場規格，包含所有可能的交點
    
    @classmethod
    def generate_standard_court_grid(cls):
        """
        根據標準羽球場規格生成 6x5 = 30個交點矩陣
        
        矩陣佈局 (從上到下，從左到右):
        00:L-corner    01:T-junction  02:T-junction  03:T-junction  04:L-corner
        10:T-junction  11:Cross       12:Cross       13:Cross       14:T-junction  
        20:T-junction  21:T-junction  22:T-junction  23:T-junction  24:T-junction
        30:T-junction  31:T-junction  32:T-junction  33:T-junction  34:T-junction
        40:T-junction  41:Cross       42:Cross       43:Cross       44:T-junction
        50:L-corner  51:T-junction  52:T-junction  53:T-junction  54:L-corner
        """
        
        # 定義6行5列的網格
        rows = 6  # Y方向
        cols = 5  # X方向
        
        # 根據羽球場標準尺寸計算正規化座標
        # X座標: 從左邊線到右邊線 (-0.5 到 0.5)
        x_positions = [-0.5, -0.25, 0.0, 0.25, 0.5]  # 5個X位置
        
        # Y座標: 從底線到頂線 (-0.5 到 0.5) 
        y_positions = [0.5, 0.295, 0.0, -0.295, -0.5, -0.6]  # 6個Y位置
        # 注意：添加了-0.6來完成6行
        
        intersections = {}
        
        # 預定義每個位置的交點類型
        intersection_types = [
            [0, 1, 1, 1, 0],  # 第0行: L-corner, T-junction, T-junction, T-junction, L-corner
            [1, 2, 2, 2, 1],  # 第1行: T-junction, Cross, Cross, Cross, T-junction
            [1, 1, 1, 1, 1],  # 第2行: T-junction, T-junction, T-junction, T-junction, T-junction
            [1, 1, 1, 1, 1],  # 第3行: T-junction, T-junction, T-junction, T-junction, T-junction
            [1, 2, 2, 2, 1],  # 第4行: T-junction, Cross, Cross, Cross, T-junction
            [0, 1, 1, 1, 0],  # 第5行: T-junction, T-junction, T-junction, T-junction, T-junction
        ]
        
        intersection_id = 0
        for row in range(rows):
            for col in range(cols):
                x = x_positions[col]
                y = y_positions[row]
                intersection_type = intersection_types[row][col]
                
                intersections[f'point_{row}{col}'] = {
                    'coords': (x, y),
                    'type': intersection_type,
                    'id': intersection_id,
                    'grid_pos': (row, col)
                }
                intersection_id += 1
        
        return intersections
    
    @classmethod
    def generate_all_intersections(cls):
        """使用標準網格生成交點"""
        return cls.generate_standard_court_grid()
    
    @classmethod
    def get_expected_distances(cls):
        """獲取所有交點間的標準距離矩陣"""
        intersections = cls.generate_all_intersections()
        points = [info['coords'] for info in intersections.values()]
        n_points = len(points)
        
        distances = torch.zeros(n_points, n_points)
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.sqrt((points[i][0] - points[j][0])**2 + 
                              (points[i][1] - points[j][1])**2)
                distances[i, j] = distances[j, i] = dist
        
        return distances, intersections
    

class HomographyConstraint(nn.Module):
    """單應性幾何約束模組"""
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.court_geometry = BadmintonCourtGeometry()
        
        # 預計算標準距離矩陣和交點信息
        expected_distances, intersections_info = self.court_geometry.get_expected_distances()
        self.register_buffer('expected_distances', expected_distances)
        
        # 存儲交點信息用於分析
        self.intersections_info = intersections_info
        
        print(f"Initialized with {len(intersections_info)} standard intersection points")
        
    def forward(self, predicted_keypoints: torch.Tensor, 
                keypoint_classes: torch.Tensor,
                confidence_threshold: float = 0.5) -> torch.Tensor:
        """
        計算單應性幾何約束損失
        
        Args:
            predicted_keypoints: [B, N, 2] 預測的關鍵點座標
            keypoint_classes: [B, N] 關鍵點類別
            confidence_threshold: 置信度閾值
            
        Returns:
            homography_loss: 幾何一致性損失
        """
        batch_size = predicted_keypoints.shape[0]
        total_loss = 0.0
        valid_samples = 0
        
        for b in range(batch_size):
            # 提取該batch的有效關鍵點
            valid_mask = keypoint_classes[b] >= 0  # 假設-1表示無效點
            if valid_mask.sum() < 4:  # 至少需要4個點計算單應性
                continue
                
            valid_points = predicted_keypoints[b][valid_mask]
            valid_classes = keypoint_classes[b][valid_mask]
            
            # 計算幾何一致性損失
            geo_loss = self._compute_geometric_consistency(valid_points, valid_classes)
            total_loss += geo_loss
            valid_samples += 1
            
        return total_loss / max(valid_samples, 1)
    
    def _compute_geometric_consistency(self, points: torch.Tensor, 
                                     classes: torch.Tensor) -> torch.Tensor:
        """計算點集的幾何一致性"""
        n_points = points.shape[0]
        if n_points < 4:
            return torch.tensor(0.0, device=self.device)
        
        # 計算預測點間的距離矩陣
        pred_distances = self._compute_distance_matrix(points)
        
        # 根據類別獲取對應的期望距離
        expected_dist = self._get_expected_distances_for_classes(classes)
        
        # 計算距離比例一致性
        ratio_loss = self._compute_ratio_consistency(pred_distances, expected_dist)
        
        # 計算角度約束
        # angle_loss = self._compute_angle_constraints(points, classes)
        
        # return ratio_loss + 0.5 * angle_loss
        return ratio_loss

    def _compute_distance_matrix(self, points: torch.Tensor) -> torch.Tensor:
        """計算點集的距離矩陣"""
        n_points = points.shape[0]
        distances = torch.zeros(n_points, n_points, device=self.device)
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = torch.norm(points[i] - points[j])
                distances[i, j] = distances[j, i] = dist
                
        return distances
    
    def _get_expected_distances_for_classes(self, classes: torch.Tensor) -> torch.Tensor:
        """根據類別獲取期望距離"""
        # 這裡需要根據實際的類別映射來實現
        # 暫時返回預設值
        n_points = len(classes)
        return torch.ones(n_points, n_points, device=self.device)
    
    def _compute_ratio_consistency(self, pred_dist: torch.Tensor, 
                                 expected_dist: torch.Tensor) -> torch.Tensor:
        """計算距離比例一致性"""
        # 避免除零
        mask = expected_dist > 1e-6
        if not mask.any():
            return torch.tensor(0.0, device=self.device)
        
        ratios_pred = pred_dist[mask] / (pred_dist[mask].mean() + 1e-6)
        ratios_expected = expected_dist[mask] / (expected_dist[mask].mean() + 1e-6)
        
        return F.smooth_l1_loss(ratios_pred, ratios_expected)
    
    def _compute_angle_constraints(self, points: torch.Tensor, 
                                 classes: torch.Tensor) -> torch.Tensor:
        """計算角度約束 (90度關係)"""
        if points.shape[0] < 3:
            return torch.tensor(0.0, device=self.device)
        
        # 尋找形成直角的點組合
        angle_loss = 0.0
        angle_count = 0
        
        for i in range(len(points)):
            for j in range(len(points)):
                for k in range(len(points)):
                    if i != j and j != k and i != k:
                        # 檢查是否應該形成直角
                        if self._should_form_right_angle(classes[i], classes[j], classes[k]):
                            angle = self._compute_angle(points[i], points[j], points[k])
                            target_angle = torch.tensor(np.pi/2, device=self.device)
                            angle_loss += (angle - target_angle).pow(2)
                            angle_count += 1
        
        return angle_loss / max(angle_count, 1)
    
    def _should_form_right_angle(self, class_i: int, class_j: int, class_k: int) -> bool:
        """判斷三個類別的點是否應該形成直角"""
        # 根據羽球場幾何結構判斷
        # 這裡需要根據具體的類別定義來實現
        return True  # 簡化版本
    
    def _compute_angle(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        """計算三點形成的角度 (以p2為頂點)"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-6)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        
        return torch.acos(cos_angle)

class SymmetryConstraint(nn.Module):
    """對稱性約束模組 - 利用羽球場左右對稱特性"""
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
    def forward(self, predicted_keypoints: torch.Tensor,
                keypoint_classes: torch.Tensor,
                image_center_x: float = 0.5) -> torch.Tensor:
        """
        計算對稱性約束損失
        
        Args:
            predicted_keypoints: [B, N, 2] 預測關鍵點
            keypoint_classes: [B, N] 關鍵點類別
            image_center_x: 影像中心線x座標 (正規化)
        """
        batch_size = predicted_keypoints.shape[0]
        total_loss = 0.0
        valid_pairs = 0
        
        for b in range(batch_size):
            valid_mask = keypoint_classes[b] >= 0
            if valid_mask.sum() < 2:
                continue
                
            valid_points = predicted_keypoints[b][valid_mask]
            valid_classes = keypoint_classes[b][valid_mask]
            
            # 找到對稱點對
            symmetric_pairs = self._find_symmetric_pairs(valid_points, valid_classes, image_center_x)
            
            for left_point, right_point in symmetric_pairs:
                # 檢查對稱性
                expected_right_x = 2 * image_center_x - left_point[0]
                symmetry_error = (right_point[0] - expected_right_x).pow(2)
                
                # y座標應該相同
                y_error = (right_point[1] - left_point[1]).pow(2)
                
                total_loss += symmetry_error + y_error
                valid_pairs += 1
        
        return total_loss / max(valid_pairs, 1)
    
    def _find_symmetric_pairs(self, points: torch.Tensor, classes: torch.Tensor, 
                            center_x: float) -> list:
        """找到對稱的點對"""
        pairs = []
        n_points = len(points)
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                if self._are_symmetric_classes(classes[i], classes[j]):
                    left_idx, right_idx = (i, j) if points[i, 0] < center_x else (j, i)
                    pairs.append((points[left_idx], points[right_idx]))
        
        return pairs
    
    def _are_symmetric_classes(self, class1: int, class2: int) -> bool:
        """判斷兩個類別是否為對稱類別"""
        # 根據羽球場的對稱結構定義
        symmetric_pairs = [(0, 0), (1, 1), (2, 2)]  # 簡化版本
        return (class1.item(), class2.item()) in symmetric_pairs
