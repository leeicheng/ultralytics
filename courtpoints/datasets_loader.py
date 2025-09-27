"""
Court Points Dataset Loader

This module provides data loading and preprocessing functionality for court line intersection detection.
Supports various data formats including YOLO format, keypoint annotations, and custom court point formats.
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import yaml
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    LOGGER.warning("Albumentations not available. Using basic transforms.")
import random

from ultralytics.utils import LOGGER
from ultralytics.data.utils import img2label_paths


class CourtPointsDataset(Dataset):
    """
    Dataset class for Court Points Detection.
    
    Supports loading images and annotations in various formats:
    - YOLO format (class, x_center, y_center, width, height)
    - Keypoint format (class, x, y, visibility)
    - Custom court points format
    
    Attributes:
        img_path (str): Path to images directory.
        mode (str): Dataset mode ('train', 'val', 'test').
        img_size (int): Target image size.
        augment (bool): Whether to apply data augmentation.
        class_names (dict): Mapping of class IDs to names.
    """
    
    def __init__(self, 
                 img_path: str, 
                 mode: str = 'train',
                 img_size: int = 640,
                 augment: bool = True,
                 cache_images: bool = False,
                 format_type: str = 'yolo'):
        """
        Initialize CourtPointsDataset.
        
        Args:
            img_path (str): Path to images directory.
            mode (str): Dataset mode ('train', 'val', 'test').
            img_size (int): Target image size for resizing.
            augment (bool): Whether to apply data augmentation.
            cache_images (bool): Whether to cache images in memory.
            format_type (str): Annotation format ('yolo', 'keypoint', 'court_points').
        """
        self.img_path = Path(img_path)
        self.mode = mode
        self.img_size = img_size
        self.augment = augment and mode == 'train'
        self.cache_images = cache_images
        self.format_type = format_type
        
        # Class names for court points
        self.class_names = {0: "T-junction", 1: "Cross", 2: "L-corner"}
        self.nc = len(self.class_names)
        
        # Load image and label paths
        self.img_files = self._load_image_paths()
        self.label_files = img2label_paths(self.img_files)
        
        # Verify files exist
        self.img_files, self.label_files = self._verify_files()
        
        # Initialize transforms
        self.transforms = self._get_transforms()
        
        # Cache for images if enabled
        self.imgs = [None] * len(self.img_files) if cache_images else None
        
        LOGGER.info(f"CourtPointsDataset: {len(self.img_files)} images, {len(self.label_files)} labels in {mode} mode")
    
    def _load_image_paths(self) -> List[str]:
        """Load all image file paths."""
        image_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
        img_files = []
        
        for ext in image_formats:
            img_files.extend(glob.glob(str(self.img_path / f"*.{ext}")))
            img_files.extend(glob.glob(str(self.img_path / f"*.{ext.upper()}")))
        
        return sorted(img_files)
    
    def _verify_files(self) -> Tuple[List[str], List[str]]:
        """Verify that image and label files exist."""
        valid_imgs, valid_labels = [], []
        
        for img_file, label_file in zip(self.img_files, self.label_files):
            if os.path.exists(img_file):
                if os.path.exists(label_file) or self.mode == 'test':
                    valid_imgs.append(img_file)
                    valid_labels.append(label_file)
                else:
                    LOGGER.warning(f"Label file not found: {label_file}")
        
        return valid_imgs, valid_labels
    
    def _get_transforms(self):
        """Get data augmentation transforms."""
        if ALBUMENTATIONS_AVAILABLE:
            if self.augment:
                return A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.RandomRotate90(p=0.1),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                    A.Blur(blur_limit=3, p=0.1),
                    A.Resize(self.img_size, self.img_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            else:
                return A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            # Fallback to basic PyTorch transforms
            return None
    
    def _load_image(self, index: int) -> np.ndarray:
        """Load image from file or cache."""
        if self.imgs is not None and self.imgs[index] is not None:
            return self.imgs[index]
        
        img_path = self.img_files[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.cache_images:
            self.imgs[index] = img
        
        return img
    
    def _load_labels(self, index: int) -> Dict:
        """Load labels from annotation file."""
        label_path = self.label_files[index]
        
        if not os.path.exists(label_path):
            return {"bboxes": [], "class_labels": []}
        
        labels = {"bboxes": [], "class_labels": []}
        
        try:
            with open(label_path, 'r') as f:
                lines = f.read().strip().split('\n')
                
                for line in lines:
                    if line:
                        parts = line.split()
                        
                        if self.format_type == 'yolo':
                            # YOLO format: class x_center y_center width height
                            cls_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            labels["bboxes"].append([x_center, y_center, width, height])
                            labels["class_labels"].append(cls_id)
                            
                        elif self.format_type == 'keypoint':
                            # Keypoint format: class x y visibility
                            cls_id = int(parts[0])
                            x, y = map(float, parts[1:3])
                            visibility = int(parts[3]) if len(parts) > 3 else 2
                            
                            # Convert keypoint to small bounding box
                            bbox_size = 0.02  # 2% of image size
                            labels["bboxes"].append([x, y, bbox_size, bbox_size])
                            labels["class_labels"].append(cls_id)
                            
        except Exception as e:
            LOGGER.warning(f"Error loading labels from {label_path}: {e}")
        
        return labels
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.img_files)
    
    def __getitem__(self, index: int) -> Dict:
        """Get dataset item."""
        # Load image
        img = self._load_image(index)
        h, w = img.shape[:2]
        
        # Load labels
        labels = self._load_labels(index)
        
        # Apply transforms
        if self.transforms:
            try:
                transformed = self.transforms(
                    image=img,
                    bboxes=labels["bboxes"],
                    class_labels=labels["class_labels"]
                )
                img = transformed['image']
                bboxes = transformed['bboxes']
                class_labels = transformed['class_labels']
            except Exception as e:
                LOGGER.warning(f"Transform error for {self.img_files[index]}: {e}")
                # Fallback: just resize and normalize
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                bboxes = labels["bboxes"]
                class_labels = labels["class_labels"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            bboxes = labels["bboxes"]
            class_labels = labels["class_labels"]
        
        # Prepare batch format
        batch_idx = torch.tensor([index] * len(class_labels), dtype=torch.long)
        cls = torch.tensor(class_labels, dtype=torch.float32).unsqueeze(1) if class_labels else torch.zeros((0, 1))
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4))
        
        return {
            'img': img,
            'batch_idx': batch_idx,
            'cls': cls,
            'bboxes': bboxes_tensor,
            'ori_shape': (h, w),
            'img_path': self.img_files[index],
            'im_file': self.img_files[index]
        }


class DatasetsLoader:
    """
    Court Points Dataset Loader with enhanced functionality.
    
    Provides a high-level interface for loading court points datasets with
    various configurations and preprocessing options.
    """
    
    def __init__(self, 
                 path: str,
                 mode: str = 'train',
                 img_size: int = 640,
                 batch_size: int = 16,
                 shuffle: bool = None,
                 augment: bool = True,
                 cache: bool = False,
                 format_type: str = 'yolo',
                 preprocess_config: Dict = None):
        """
        Initialize DatasetsLoader.
        
        Args:
            path (str): Path to dataset directory.
            mode (str): Dataset mode ('train', 'val', 'test').
            img_size (int): Target image size.
            batch_size (int): Batch size for DataLoader.
            shuffle (bool): Whether to shuffle data. Defaults to True for train, False for others.
            augment (bool): Whether to apply data augmentation.
            cache (bool): Whether to cache images in memory.
            format_type (str): Annotation format type.
            preprocess_config (dict): Additional preprocessing configuration.
        """
        self.path = path
        self.mode = mode
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle if shuffle is not None else (mode == 'train')
        self.augment = augment
        self.cache = cache
        self.format_type = format_type
        self.preprocess_config = preprocess_config or {}
        
        # Create dataset
        self.dataset = CourtPointsDataset(
            img_path=path,
            mode=mode,
            img_size=img_size,
            augment=augment,
            cache_images=cache,
            format_type=format_type
        )
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=self.shuffle,
            num_workers=4,
            pin_memory=True,
            collate_fn=self.collate_fn,
            drop_last=mode == 'train'
        )
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for court points data."""
        img = torch.stack([item['img'] for item in batch])
        
        # Handle variable number of annotations per image
        batch_idx, cls, bboxes = [], [], []
        
        for i, item in enumerate(batch):
            n = len(item['cls'])
            if n > 0:
                batch_idx.append(torch.full((n,), i, dtype=torch.long))
                cls.append(item['cls'])
                bboxes.append(item['bboxes'])
        
        batch_idx = torch.cat(batch_idx) if batch_idx else torch.zeros((0,), dtype=torch.long)
        cls = torch.cat(cls) if cls else torch.zeros((0, 1))
        bboxes = torch.cat(bboxes) if bboxes else torch.zeros((0, 4))
        
        return {
            'img': img,
            'batch_idx': batch_idx,
            'cls': cls,
            'bboxes': bboxes,
            'ori_shape': torch.tensor([item['ori_shape'] for item in batch]),
            'im_file': [item['im_file'] for item in batch]
        }
    
    def __len__(self):
        """Return dataset length."""
        return len(self.dataset)
    
    def __iter__(self):
        """Return dataloader iterator."""
        return iter(self.dataloader)
    
    def set_preprocessing(self, config: Dict):
        """Set preprocessing configuration."""
        self.preprocess_config.update(config)
        
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        class_counts = torch.zeros(self.dataset.nc)
        
        for item in self.dataset:
            if len(item['cls']) > 0:
                for cls_id in item['cls'].flatten():
                    class_counts[int(cls_id)] += 1
        
        # Inverse frequency weighting
        total = class_counts.sum()
        weights = total / (class_counts + 1e-6)  # Add small epsilon to avoid division by zero
        weights = weights / weights.sum() * len(weights)  # Normalize
        
        return weights
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'num_images': len(self.dataset),
            'num_classes': self.dataset.nc,
            'class_names': self.dataset.class_names,
            'img_size': self.img_size,
            'mode': self.mode,
            'format_type': self.format_type
        }
        
        # Calculate annotation statistics
        total_annotations = 0
        class_counts = torch.zeros(self.dataset.nc)
        
        for item in self.dataset:
            n_annotations = len(item['cls'])
            total_annotations += n_annotations
            
            if n_annotations > 0:
                for cls_id in item['cls'].flatten():
                    class_counts[int(cls_id)] += 1
        
        stats.update({
            'total_annotations': total_annotations,
            'avg_annotations_per_image': total_annotations / len(self.dataset),
            'class_distribution': {
                self.dataset.class_names[i]: int(count) 
                for i, count in enumerate(class_counts)
            }
        })
        
        return stats


class PreprocessLayer(nn.Module):
    """
    Preprocessing layer for court points detection.
    
    Provides various preprocessing options including grayscale conversion,
    median subtraction, and normalization.
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 grayscale: bool = False,
                 median_subtraction: bool = False,
                 normalize: bool = True):
        """
        Initialize PreprocessLayer.
        
        Args:
            in_channels (int): Number of input channels.
            grayscale (bool): Whether to convert to grayscale.
            median_subtraction (bool): Whether to apply median subtraction.
            normalize (bool): Whether to normalize values.
        """
        super().__init__()
        self.grayscale = grayscale
        self.median_subtraction = median_subtraction
        self.normalize = normalize
        
        if grayscale:
            # 灰階轉換：固定權重，不訓練
            self.to_grayscale = nn.Conv2d(3, 1, kernel_size=1, bias=False)
            self.to_grayscale.weight.data = torch.tensor([[[[0.2989]], [[0.5870]], [[0.1140]]]])
            self.to_grayscale.weight.requires_grad = False
            
            # 如果要保持3通道輸出
            self.expand_channels = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            self.expand_channels.weight.data = torch.ones(3, 1, 1, 1)
            self.expand_channels.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through preprocessing layer.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Preprocessed tensor.
        """
        B, C, H, W = x.shape
        
        # Apply grayscale conversion if enabled
        if self.grayscale and C == 3:
            gray = self.to_grayscale(x)  # (B, 1, H, W)
            x = self.expand_channels(gray)  # (B, 3, H, W)
        
        # Apply median subtraction if enabled
        if self.median_subtraction:
            if x.dim() == 4:  # Batch dimension
                median = x.median(dim=2, keepdim=True).values.median(dim=3, keepdim=True).values
                x = x - median
        
        # Clamp values to valid range
        x = torch.clamp(x, 0, 255) if not self.normalize else torch.clamp(x, 0, 1)
        
        # Normalize if enabled
        if self.normalize and x.max() > 1:
            x = x / 255.0
            
        return x


# Utility functions for dataset management
def create_dataset_yaml(data_path: str, 
                       train_path: str, 
                       val_path: str,
                       test_path: str = None,
                       output_path: str = "court_points.yaml"):
    """Create dataset YAML configuration file."""
    
    config = {
        'path': data_path,
        'train': train_path,
        'val': val_path,
        'nc': 3,
        'names': {
            0: 'T-junction',
            1: 'Cross', 
            2: 'L-corner'
        }
    }
    
    if test_path:
        config['test'] = test_path
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    LOGGER.info(f"Dataset configuration saved to {output_path}")
    return output_path


def verify_dataset(data_path: str) -> Dict:
    """Verify dataset integrity and provide statistics."""
    
    def count_files(path, extensions):
        count = 0
        for ext in extensions:
            count += len(glob.glob(os.path.join(path, f"*.{ext}")))
        return count
    
    image_exts = ['jpg', 'jpeg', 'png', 'bmp']
    label_exts = ['txt']
    
    stats = {}
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_path, split)
        
        if os.path.exists(split_path):
            img_count = count_files(os.path.join(split_path, 'images'), image_exts)
            label_count = count_files(os.path.join(split_path, 'labels'), label_exts)
            
            stats[split] = {
                'images': img_count,
                'labels': label_count,
                'missing_labels': max(0, img_count - label_count)
            }
        else:
            stats[split] = {'images': 0, 'labels': 0, 'missing_labels': 0}
    
    return stats