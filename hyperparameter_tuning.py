# 新文件: hyperparameter_tuning.py

import optuna
from ultralytics import YOLO
import tempfile
import os

def objective(trial):
    """Optuna優化目標函數"""
    
    # 採樣超參數
    w_homo = trial.suggest_float('w_homo', 0.1, 1.0)
    w_sym = trial.suggest_float('w_sym', 0.1, 0.8) 
    w_ratio = trial.suggest_float('w_ratio', 0.05, 0.5)
    w_angle = trial.suggest_float('w_angle', 0.05, 0.5)
    
    lr0 = trial.suggest_float('lr0', 0.001, 0.1, log=True)
    pose_weight = trial.suggest_float('pose_weight', 8.0, 20.0)
    
    # 創建臨時配置
    config = {
        'model': 'yolov8n-pose.pt',
        'data': 'badminton_kpts.yaml',
        'epochs': 50,  # 減少epochs以加速調優
        'batch': 16,
        'imgsz': 640,
        'lr0': lr0,
        'pose': pose_weight,
        'patience': 10,
        'enhanced_loss': {
            'w_homo': w_homo,
            'w_sym': w_sym, 
            'w_ratio': w_ratio,
            'w_angle': w_angle
        }
    }
    
    # 訓練模型
    model = YOLO('yolov8n-pose.pt')
    results = model.train(**config)
    
    # 返回驗證mAP作為優化目標
    return results.results_dict.get('metrics/mAP50(B)', 0.0)

def run_hyperparameter_optimization():
    """執行超參數優化"""
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    print("Best parameters:")
    print(study.best_params)
    print(f"Best mAP@0.5: {study.best_value:.4f}")
    
    return study.best_params

if __name__ == '__main__':
    best_params = run_hyperparameter_optimization()
