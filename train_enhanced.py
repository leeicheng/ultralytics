# 修改或創建新的訓練腳本: train_enhanced.py

import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import yaml

def load_enhanced_config(config_path):
    """載入增強版配置"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_enhanced_training(model_path, config):
    """設置增強版訓練"""
    model = YOLO(model_path)
    
    # 設置增強版損失權重
    if hasattr(model.model, 'model') and len(model.model.model) > 0:
        # 獲取最後一層（通常是損失函數）
        last_layer = model.model.model[-1]
        if hasattr(last_layer, 'loss'):
            loss_fn = last_layer.loss
            if hasattr(loss_fn, 'keypoint_loss'):
                # 更新權重
                enhanced_config = config.get('enhanced_loss', {})
                loss_fn.keypoint_loss.w_base = enhanced_config.get('w_base', 1.0)
                loss_fn.keypoint_loss.w_homo = enhanced_config.get('w_homo', 0.5)
                loss_fn.keypoint_loss.w_sym = enhanced_config.get('w_sym', 0.3)
                loss_fn.keypoint_loss.w_ratio = enhanced_config.get('w_ratio', 0.2)
                loss_fn.keypoint_loss.w_angle = enhanced_config.get('w_angle', 0.2)
                
                LOGGER.info(f"Enhanced loss weights updated: {enhanced_config}")
    return model

def main():
    # 載入配置
    config = load_enhanced_config('ultralytics/cfg/enhanced_badminton.yaml')
    
    # 設置模型
    model = setup_enhanced_training(config['model'], config)
    
    # 提取訓練參數
    train_args = {k: v for k, v in config.items() 
                  if k not in ['enhanced_loss', 'geometry_constraints']}
    
    # 開始訓練
    results = model.train(**train_args)
    
    # 訓練完成後的分析
    LOGGER.info("Enhanced training completed!")
    LOGGER.info(f"Best mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    
    return results

if __name__ == '__main__':
    main()
