# 新文件: validate_enhanced.py

import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
import matplotlib.pyplot as plt

class EnhancedValidator:
    """增強版驗證器"""
    
    def __init__(self, model_path, data_path):
        self.model = YOLO(model_path)
        self.data_path = data_path
        
    def validate_with_geometry_analysis(self):
        """執行幾何分析驗證"""
        results = self.model.val(data=self.data_path)
        
        # 分析幾何一致性
        geometry_metrics = self._analyze_geometry_consistency()
        
        # 分析對稱性
        symmetry_metrics = self._analyze_symmetry()
        
        # 生成報告
        self._generate_validation_report(results, geometry_metrics, symmetry_metrics)
        
        return results
    
    def _analyze_geometry_consistency(self):
        """分析幾何一致性"""
        # 實現幾何一致性分析邏輯
        return {
            'avg_ratio_error': 0.0,
            'angle_accuracy': 0.0,
            'distance_consistency': 0.0
        }
    
    def _analyze_symmetry(self):
        """分析對稱性"""
        # 實現對稱性分析邏輯
        return {
            'symmetry_error': 0.0,
            'symmetric_pairs_detected': 0
        }
    
    def _generate_validation_report(self, results, geometry_metrics, symmetry_metrics):
        """生成驗證報告"""
        print("=== Enhanced Validation Report ===")
        print(f"mAP@0.5: {results.box.map50}")
        print(f"mAP@0.5:0.95: {results.box.map}")
        print(f"Geometry consistency: {geometry_metrics}")
        print(f"Symmetry metrics: {symmetry_metrics}")

def compare_with_baseline():
    """與基線模型比較"""
    print("Comparing enhanced model with baseline...")
    
    # 載入基線模型
    baseline_model = YOLO('runs/pose/badminton_kpts9/weights/best.pt')
    baseline_results = baseline_model.val()
    
    # 載入增強模型
    enhanced_model = YOLO('runs/pose/enhanced_badminton/weights/best.pt')  
    enhanced_results = enhanced_model.val()
    
    # 比較結果
    print(f"Baseline mAP@0.5: {baseline_results.box.map50:.4f}")
    print(f"Enhanced mAP@0.5: {enhanced_results.box.map50:.4f}")
    print(f"Improvement: {enhanced_results.box.map50 - baseline_results.box.map50:.4f}")

def test_geometry_model():
    """測試幾何模型是否正確生成交點"""
    from ultralytics.utils.geometry_constraints import BadmintonCourtGeometry
    
    geometry = BadmintonCourtGeometry()
    intersections = geometry.generate_all_intersections()
    
    print(f"Generated {len(intersections)} intersection points:")
    
    # 按類型統計
    type_counts = {0: 0, 1: 0, 2: 0}  # L角, T字, 十字
    type_names = {0: 'L角', 1: 'T字', 2: '十字'}
    
    for name, info in intersections.items():
        intersection_type = info['type']
        coords = info['coords']
        type_counts[intersection_type] += 1
        print(f"{name}: {coords} -> {type_names[intersection_type]}")
    
    print(f"\n統計:")
    for type_id, count in type_counts.items():
        print(f"{type_names[type_id]}: {count}個")
    
    print(f"總計: {sum(type_counts.values())}個交點")
    
    # 檢查是否接近實際標註的數量（應該約30-50個）
    total_intersections = sum(type_counts.values())
    if 25 <= total_intersections <= 50:
        print("✅ 交點數量合理")
    else:
        print(f"⚠️  交點數量可能有誤: {total_intersections}")

if __name__ == '__main__':
    # 首先測試幾何模型
    test_geometry_model()
    
    # 然後進行驗證
    validator = EnhancedValidator(
        'runs/pose/enhanced_badminton/weights/best.pt',
        'badminton_kpts.yaml'
    )
    validator.validate_with_geometry_analysis()
    compare_with_baseline()
