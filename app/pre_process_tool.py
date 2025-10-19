# -*- coding: utf-8 -*-
"""
PyQt6 app for "color-based pixel-by-pixel filtering" from the paper
"Accurate Tennis Court Line Detection on Amateur Recorded Matches".

功能：
- 選擇影像或影片檔案（JPG/PNG/MP4 等）。
- 顯示原始畫面、court mask、line mask、overlay。
- 可調參數：num_samples、bin_size、lab_tol、window_size、neighbor_thresh。
- 影片支援：用滑桿切換幀並處理當前幀。

依賴：
- PyQt6, opencv-python, numpy

執行：
  python app.py

作者：ChatGPT（改作示範）
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QSlider, QCheckBox, QMessageBox
)

# ----------------------------
# 影像處理核心函式
# ----------------------------

def quantize_color(img_bgr: np.ndarray, bin_size: int = 16) -> np.ndarray:
    """對 BGR 進行顏色量化，用於近似找出「最常見色」。"""
    q = (img_bgr // bin_size) * bin_size
    return q


def dominant_court_color(img_bgr: np.ndarray, num_samples: int = 1000, bin_size: int = 16, rng_seed: int = 42) -> np.ndarray:
    """以隨機抽樣 + 量化眾數估計球場主色（BGR）。"""
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty image for dominant color estimation.")

    rng = np.random.default_rng(rng_seed)
    ys = rng.integers(0, h, size=num_samples)
    xs = rng.integers(0, w, size=num_samples)

    samples = img_bgr[ys, xs]  # (N, 3) BGR
    samples_q = quantize_color(samples, bin_size=bin_size)

    keys = samples_q.reshape(-1, 3)
    keys_t = [tuple(px.tolist()) for px in keys]
    unique, counts = np.unique(keys_t, axis=0, return_counts=True)
    dom_idx = int(np.argmax(counts))
    dominant_bgr = np.array(unique[dom_idx], dtype=np.uint8)
    return dominant_bgr


def bgr2lab(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)


def lab_distance(lab_img: np.ndarray, lab_color: np.ndarray) -> np.ndarray:
    """計算每個像素與單一 LAB 顏色之歐氏距離。"""
    diff = lab_img.astype(np.int16) - lab_color.reshape(1, 1, 3).astype(np.int16)
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return dist


def court_line_filter(
    img_bgr: np.ndarray,
    num_samples: int = 1000,
    bin_size: int = 16,
    lab_tol: float = 12.0,
    window_size: int = 7,
    neighbor_thresh: int = 4,
    morphology: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    產生 (dominant_bgr, court_mask, line_mask, overlay)
    court_mask, line_mask 為 8-bit 單通道影像（0/255）。
    overlay 為 BGR。
    """
    dominant_bgr = dominant_court_color(img_bgr, num_samples=num_samples, bin_size=bin_size)

    lab_img = bgr2lab(img_bgr)
    dom_lab = cv2.cvtColor(dominant_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).reshape(3,)
    dist = lab_distance(lab_img, dom_lab)
    court_mask = (dist <= lab_tol).astype(np.uint8)

    k = np.ones((window_size, window_size), np.uint8)
    neighbor_count = cv2.filter2D(court_mask, ddepth=cv2.CV_16S, kernel=k, borderType=cv2.BORDER_REPLICATE)

    line_mask = ((neighbor_count >= neighbor_thresh) & (court_mask == 0)).astype(np.uint8) * 255

    if morphology:
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    overlay = img_bgr.copy()
    overlay[line_mask > 0] = (0, 255, 255)  # 標成黃

    return dominant_bgr, court_mask * 255, line_mask, overlay


# ----------------------------
# PyQt6 UI
# ----------------------------

def cvimg_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    """BGR (OpenCV) -> QPixmap"""
    if img_bgr.ndim == 2:
        qimg = QImage(img_bgr.data, img_bgr.shape[1], img_bgr.shape[0], img_bgr.strides[0], QImage.Format.Format_Grayscale8)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], img_rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class ImagePanel(QLabel):
    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setText(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(320, 180))
        self.setStyleSheet("QLabel { background: #222; color: #DDD; border: 1px solid #444; }")
        self.setScaledContents(True)

    def set_image(self, img_bgr: Optional[np.ndarray], title: str):
        if img_bgr is None:
            self.setText(title)
            return
        pix = cvimg_to_qpixmap(img_bgr)
        self.setPixmap(pix)
        self.setToolTip(title)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tennis Court Pixel-by-Pixel Filtering — PyQt6")
        self.resize(1500, 900)

        # 狀態
        self.current_path: Optional[Path] = None
        self.is_video: bool = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.current_frame_idx: int = 0
        self.current_frame_bgr: Optional[np.ndarray] = None

        # 介面
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 工具列/按鈕
        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("選擇影像/影片…")
        self.btn_open.clicked.connect(self.on_open)
        self.btn_process = QPushButton("處理當前畫面")
        self.btn_process.clicked.connect(self.on_process)
        self.btn_process.setEnabled(False)
        btn_row.addWidget(self.btn_open)
        btn_row.addWidget(self.btn_process)
        layout.addLayout(btn_row)

        # 影片滑桿
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setEnabled(False)
        self.video_slider.valueChanged.connect(self.on_seek)
        layout.addWidget(self.video_slider)

        # 參數面板
        param_box = QGroupBox("參數 (可即時調整)")
        form = QFormLayout(param_box)

        self.sp_num_samples = QSpinBox(); self.sp_num_samples.setRange(100, 100000); self.sp_num_samples.setValue(1000)
        self.sp_bin_size   = QSpinBox(); self.sp_bin_size.setRange(1, 64); self.sp_bin_size.setValue(16)
        self.sp_lab_tol    = QDoubleSpinBox(); self.sp_lab_tol.setDecimals(2); self.sp_lab_tol.setRange(0.0, 100.0); self.sp_lab_tol.setValue(12.0)
        self.sp_win_size   = QSpinBox(); self.sp_win_size.setRange(3, 31); self.sp_win_size.setSingleStep(2); self.sp_win_size.setValue(7)
        self.sp_neighbor   = QSpinBox(); self.sp_neighbor.setRange(1, 225); self.sp_neighbor.setValue(4)
        self.cb_morph      = QCheckBox("啟用形態學清理 (open/close)"); self.cb_morph.setChecked(True)

        form.addRow("num_samples", self.sp_num_samples)
        form.addRow("bin_size", self.sp_bin_size)
        form.addRow("lab_tol", self.sp_lab_tol)
        form.addRow("window_size (odd)", self.sp_win_size)
        form.addRow("neighbor_thresh", self.sp_neighbor)
        form.addRow(self.cb_morph)
        layout.addWidget(param_box)

        # 顯示面板 2x2
        grid = QHBoxLayout()
        left_col = QVBoxLayout(); right_col = QVBoxLayout()
        grid.addLayout(left_col, 1); grid.addLayout(right_col, 1)

        self.panel_orig   = ImagePanel("原始影像 / 當前幀")
        self.panel_cmask  = ImagePanel("Court Mask")
        self.panel_lmask  = ImagePanel("Line Mask")
        self.panel_overlay= ImagePanel("Overlay")

        left_col.addWidget(self.panel_orig)
        left_col.addWidget(self.panel_cmask)
        right_col.addWidget(self.panel_lmask)
        right_col.addWidget(self.panel_overlay)
        layout.addLayout(grid, 1)

        # 快捷選單
        self._build_menu()

    def _build_menu(self):
        menu = self.menuBar()
        filem = menu.addMenu("檔案")
        act_open = QAction("開啟…", self); act_open.triggered.connect(self.on_open)
        act_quit = QAction("離開", self); act_quit.triggered.connect(self.close)
        filem.addAction(act_open)
        filem.addSeparator(); filem.addAction(act_quit)

    # ----------------------------
    # 檔案/影片操作
    # ----------------------------
    def on_open(self):
        filters = "媒體檔案 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.mp4 *.mov *.avi *.mkv);;影像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;影片 (*.mp4 *.mov *.avi *.mkv);;所有檔案 (*)"
        path_str, _ = QFileDialog.getOpenFileName(self, "選擇影像或影片", "", filters)
        if not path_str:
            return
        path = Path(path_str)
        self.load_path(path)

    def load_path(self, path: Path):
        self.cleanup_video()
        self.current_path = path
        suffix = path.suffix.lower()

        if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
            self.is_video = True
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                QMessageBox.critical(self, "錯誤", f"無法開啟影片：{path}")
                self.cleanup_video()
                return
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.total_frames <= 0:
                self.total_frames = 0
            self.video_slider.setEnabled(True)
            self.video_slider.setMinimum(0)
            self.video_slider.setMaximum(max(0, self.total_frames - 1))
            self.current_frame_idx = 0
            self.read_frame(self.current_frame_idx)
        else:
            self.is_video = False
            img = cv2.imread(str(path))
            if img is None:
                QMessageBox.critical(self, "錯誤", f"無法讀取影像：{path}")
                return
            self.current_frame_bgr = img
            self.refresh_panels()

        self.btn_process.setEnabled(True)
        self.statusBar().showMessage(str(path))

    def cleanup_video(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.video_slider.setEnabled(False)

    def read_frame(self, idx: int) -> bool:
        if self.cap is None:
            return False
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False
        self.current_frame_idx = idx
        self.current_frame_bgr = frame
        self.refresh_panels()
        return True

    def on_seek(self, value: int):
        if not self.is_video or self.cap is None:
            return
        self.read_frame(value)

    # ----------------------------
    # 顯示/處理
    # ----------------------------
    def refresh_panels(self,
                       cmask: Optional[np.ndarray] = None,
                       lmask: Optional[np.ndarray] = None,
                       overlay: Optional[np.ndarray] = None):
        # 原始
        self.panel_orig.set_image(self.current_frame_bgr, "原始影像 / 當前幀")
        # 其他
        self.panel_cmask.set_image(cmask if cmask is not None else None, "Court Mask")
        self.panel_lmask.set_image(lmask if lmask is not None else None, "Line Mask")
        self.panel_overlay.set_image(overlay if overlay is not None else None, "Overlay")

    def on_process(self):
        if self.current_frame_bgr is None:
            QMessageBox.information(self, "提示", "尚未載入影像或影片")
            return
        try:
            num_samples = self.sp_num_samples.value()
            bin_size = self.sp_bin_size.value()
            lab_tol = self.sp_lab_tol.value()
            window_size = self.sp_win_size.value()
            if window_size % 2 == 0:
                window_size += 1  # 確保為奇數
            neighbor_thresh = self.sp_neighbor.value()
            morph = self.cb_morph.isChecked()

            dom_bgr, cmask, lmask, overlay = court_line_filter(
                self.current_frame_bgr,
                num_samples=num_samples,
                bin_size=bin_size,
                lab_tol=float(lab_tol),
                window_size=window_size,
                neighbor_thresh=neighbor_thresh,
                morphology=morph,
            )
            # 更新顯示
            self.refresh_panels(cmask, lmask, overlay)
            self.statusBar().showMessage(f"Dominant court color (B,G,R) = {dom_bgr.tolist()}")
        except Exception as e:
            QMessageBox.critical(self, "處理錯誤", str(e))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
