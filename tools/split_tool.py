#!/usr/bin/env python3
"""video_frame_extractor.py – PyQt6 GUI tool to list videos and split them into frames

Features
========
* Recursively scans a chosen root folder for video files (mp4/avi/mov/mkv).
* Displays a table with **Video Name | Duration (HH:MM:SS) | Total Frames | FPS**.
* Lets you select any row and specify how many frames **per second** to extract.
* The **Extract** button now shows the *expected* number of output images: «拆分所選影片 (產生 N 張照片)».
* Extracted frames are saved under a sibling directory named `<video_name>_frames` next to the original video.

Run with: ``python video_frame_extractor.py``  (Python ≥ 3.9, PyQt6, OpenCV‑Python).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Tuple

import cv2
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QMessageBox,
    QProgressBar,
    QCheckBox,
    QGroupBox,
    QGridLayout,
)
from PyQt6.QtCore import Qt

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class VideoFrameExtractor(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Video‑to‑Frames Tool")
        self.resize(1000, 600)

        # ── Widgets ───────────────────────────────────────────────────────────
        self.btn_choose_root = QPushButton("選擇資料夾並掃描影片…")
        self.btn_extract = QPushButton("拆分所選影片")
        self.btn_extract.setEnabled(False)

        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 120)
        self.spin_fps.setValue(5)
        self.lbl_fps = QLabel("每秒擷取幾幀：")
        
        self.checkbox_grayscale = QCheckBox("輸出灰階影像")
        self.checkbox_grayscale.setChecked(False)
        
        self.checkbox_dataset_split = QCheckBox("分割資料集")
        self.checkbox_dataset_split.setChecked(False)
        
        # 資料集比例設定
        self.dataset_group = QGroupBox("資料集分割比例 (%)")
        self.dataset_group.setEnabled(False)  # 預設關閉
        
        self.spin_train = QSpinBox()
        self.spin_train.setRange(10, 90)
        self.spin_train.setValue(70)
        self.spin_train.setSuffix("%")
        
        self.spin_test = QSpinBox()
        self.spin_test.setRange(5, 50)
        self.spin_test.setValue(20)
        self.spin_test.setSuffix("%")
        
        self.spin_valid = QSpinBox()
        self.spin_valid.setRange(5, 50)
        self.spin_valid.setValue(10)
        self.spin_valid.setSuffix("%")
        
        # 資料集比例佈局
        dataset_layout = QGridLayout(self.dataset_group)
        dataset_layout.addWidget(QLabel("Train:"), 0, 0)
        dataset_layout.addWidget(self.spin_train, 0, 1)
        dataset_layout.addWidget(QLabel("Test:"), 0, 2)
        dataset_layout.addWidget(self.spin_test, 0, 3)
        dataset_layout.addWidget(QLabel("Valid:"), 0, 4)
        dataset_layout.addWidget(self.spin_valid, 0, 5)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["影片名稱", "時間", "總 Frames", "FPS"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        # ── Layout ────────────────────────────────────────────────────────────
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.btn_choose_root)
        top_bar.addStretch()
        top_bar.addWidget(self.lbl_fps)
        top_bar.addWidget(self.spin_fps)
        top_bar.addWidget(self.checkbox_grayscale)
        top_bar.addWidget(self.checkbox_dataset_split)
        top_bar.addWidget(self.btn_extract)

        central = QWidget()
        vbox = QVBoxLayout(central)
        vbox.addLayout(top_bar)
        vbox.addWidget(self.dataset_group)
        vbox.addWidget(self.table)
        vbox.addWidget(self.progress)
        self.setCentralWidget(central)

        # ── Signals ───────────────────────────────────────────────────────────
        self.btn_choose_root.clicked.connect(self._choose_and_scan)
        self.btn_extract.clicked.connect(self._extract_selected)
        self.table.itemSelectionChanged.connect(self._update_extract_button)
        self.spin_fps.valueChanged.connect(self._update_extract_button)
        self.checkbox_dataset_split.toggled.connect(self._toggle_dataset_split)
        
        # 比例變更時自動調整
        self.spin_train.valueChanged.connect(self._adjust_ratios)
        self.spin_test.valueChanged.connect(self._adjust_ratios)
        self.spin_valid.valueChanged.connect(self._adjust_ratios)

        # internal
        self._video_paths: list[Path] = []  # parallel to table rows
    
    def _toggle_dataset_split(self, checked: bool):
        """啟用/關閉資料集分割選項"""
        self.dataset_group.setEnabled(checked)
    
    def _adjust_ratios(self):
        """確保三個比例總和為100%"""
        sender = self.sender()
        if sender == self.spin_train:
            remaining = 100 - self.spin_train.value()
            # 按現有比例分配剩餘的給test和valid
            current_test_valid = self.spin_test.value() + self.spin_valid.value()
            if current_test_valid > 0:
                test_ratio = self.spin_test.value() / current_test_valid
                self.spin_test.setValue(int(remaining * test_ratio))
                self.spin_valid.setValue(remaining - self.spin_test.value())
            else:
                self.spin_test.setValue(remaining // 2)
                self.spin_valid.setValue(remaining - self.spin_test.value())
        elif sender == self.spin_test:
            remaining = 100 - self.spin_test.value()
            # 按現有比例分配剩餘的給train和valid
            current_train_valid = self.spin_train.value() + self.spin_valid.value()
            if current_train_valid > 0:
                train_ratio = self.spin_train.value() / current_train_valid
                self.spin_train.setValue(int(remaining * train_ratio))
                self.spin_valid.setValue(remaining - self.spin_train.value())
            else:
                self.spin_train.setValue(int(remaining * 0.8))
                self.spin_valid.setValue(remaining - self.spin_train.value())
        elif sender == self.spin_valid:
            remaining = 100 - self.spin_valid.value()
            # 按現有比例分配剩餘的給train和test
            current_train_test = self.spin_train.value() + self.spin_test.value()
            if current_train_test > 0:
                train_ratio = self.spin_train.value() / current_train_test
                self.spin_train.setValue(int(remaining * train_ratio))
                self.spin_test.setValue(remaining - self.spin_train.value())
            else:
                self.spin_train.setValue(int(remaining * 0.8))
                self.spin_test.setValue(remaining - self.spin_train.value())

    # ───────────────────────────────────────────────────────────────────────
    # Scanning utilities
    # ───────────────────────────────────────────────────────────────────────

    def _choose_and_scan(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "選擇包含影片的根資料夾")
        if folder:
            self._scan_videos(Path(folder))

    def _scan_videos(self, root: Path) -> None:
        self.table.setRowCount(0)
        self._video_paths.clear()
        for path in sorted(self._iter_videos(root)):
            info = self._probe_video(path)
            if info is None:
                continue
            duration, frames, fps = info
            self._append_row(path, hhmmss(duration), frames, f"{fps:.2f}")
        self.table.resizeColumnsToContents()
        self._update_extract_button()

    @staticmethod
    def _iter_videos(root: Path) -> Iterable[Path]:
        for p in root.rglob("*"):
            if p.suffix.lower() in VIDEO_EXTS and p.is_file():
                yield p

    @staticmethod
    def _probe_video(path: Path) -> Tuple[float, int, float] | None:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps else 0.0
        cap.release()
        return duration, frame_count, fps

    def _append_row(self, path: Path, dur: str, frames: int, fps: str) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for col, text in enumerate([path.name, dur, str(frames), fps]):
            item = QTableWidgetItem(text)
            if col == 0:
                item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.table.setItem(row, col, item)
        self._video_paths.append(path)

    # ───────────────────────────────────────────────────────────────────────
    # UI Helpers
    # ───────────────────────────────────────────────────────────────────────

    def _update_extract_button(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            self.btn_extract.setEnabled(False)
            self.btn_extract.setText("拆分所選影片")
            return

        row = sel[0].row()
        total_frames = int(self.table.item(row, 2).text())
        vid_fps = float(self.table.item(row, 3).text())
        target_fps = self.spin_fps.value()
        interval = max(1, int(round(vid_fps / target_fps)))
        expected = total_frames // interval
        self.btn_extract.setEnabled(True)
        self.btn_extract.setText(f"拆分所選影片 (產生 {expected} 張照片)")

    # ───────────────────────────────────────────────────────────────────────
    # Extraction
    # ───────────────────────────────────────────────────────────────────────

    def _extract_selected(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        row = sel[0].row()
        video_path = self._video_paths[row]
        target_fps = self.spin_fps.value()
        grayscale = self.checkbox_grayscale.isChecked()
        dataset_split = self.checkbox_dataset_split.isChecked()
        
        if dataset_split:
            train_ratio = self.spin_train.value() / 100.0
            test_ratio = self.spin_test.value() / 100.0
            valid_ratio = self.spin_valid.value() / 100.0
            self._extract_frames_with_split(video_path, target_fps, grayscale, 
                                          train_ratio, test_ratio, valid_ratio)
        else:
            self._extract_frames(video_path, target_fps, grayscale)
        self._update_extract_button()  # refresh label with new selection / counts

    def _extract_frames(self, video_path: Path, target_fps: int, grayscale: bool = False):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            QMessageBox.warning(self, "錯誤", f"無法開啟影片：{video_path}")
            return
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        if vid_fps == 0:
            QMessageBox.warning(self, "錯誤", "讀取不到 FPS。")
            cap.release()
            return
        interval = max(1, int(round(vid_fps / target_fps)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 根據灰階選項決定輸出目錄名稱
        suffix = "_frames_gray" if grayscale else "_frames"
        out_dir = video_path.parent / f"{video_path.stem}{suffix}"
        out_dir.mkdir(exist_ok=True)

        self.progress.setMaximum(total_frames)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        saved = 0
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                # 處理灰階轉換
                if grayscale:
                    # 轉換為灰階
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # 為了保持一致性，使用_gray前綴
                    out_name = out_dir / f"{video_path.stem}_gray_{saved:06d}.jpg"
                else:
                    out_name = out_dir / f"{video_path.stem}_{saved:06d}.jpg"
                
                cv2.imwrite(str(out_name), frame)
                saved += 1
            idx += 1
            if idx % 5 == 0:
                self.progress.setValue(idx)
                QApplication.processEvents()

        cap.release()
        self.progress.setVisible(False)
        
        # 根據灰階選項顯示不同的完成訊息
        mode_text = "灰階" if grayscale else "彩色"
        QMessageBox.information(self, "完成", f"已擷取 {saved} 幀（{mode_text}）至 {out_dir}。")

    def _extract_frames_with_split(self, video_path: Path, target_fps: int, 
                                 grayscale: bool, train_ratio: float, 
                                 test_ratio: float, valid_ratio: float):
        """擷取影格並分割到train/test/valid資料夾"""
        import random
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            QMessageBox.warning(self, "錯誤", f"無法開啟影片：{video_path}")
            return
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        if vid_fps == 0:
            QMessageBox.warning(self, "錯誤", "讀取不到 FPS。")
            cap.release()
            return
        interval = max(1, int(round(vid_fps / target_fps)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 建立資料集結構
        suffix = "_dataset_gray" if grayscale else "_dataset"
        base_dir = video_path.parent / f"{video_path.stem}{suffix}"
        
        train_dir = base_dir / "train"
        test_dir = base_dir / "test"
        valid_dir = base_dir / "valid"
        
        for dir_path in [train_dir, test_dir, valid_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.progress.setMaximum(total_frames)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        
        # 收集所有要擷取的影格
        frames_to_extract = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                if grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames_to_extract.append(frame.copy())
            idx += 1
            if idx % 5 == 0:
                self.progress.setValue(idx)
                QApplication.processEvents()
        
        cap.release()
        
        # 隨機分配影格到不同資料集
        total_extracted = len(frames_to_extract)
        train_count = int(total_extracted * train_ratio)
        test_count = int(total_extracted * test_ratio)
        valid_count = total_extracted - train_count - test_count
        
        # 建立索引列表並隨機打亂
        indices = list(range(total_extracted))
        random.shuffle(indices)
        
        # 分配索引
        train_indices = set(indices[:train_count])
        test_indices = set(indices[train_count:train_count + test_count])
        valid_indices = set(indices[train_count + test_count:])
        
        # 儲存影格到對應資料夾
        train_saved = test_saved = valid_saved = 0
        
        for i, frame in enumerate(frames_to_extract):
            if i in train_indices:
                if grayscale:
                    out_name = train_dir / f"{video_path.stem}_gray_{train_saved:06d}.jpg"
                else:
                    out_name = train_dir / f"{video_path.stem}_{train_saved:06d}.jpg"
                cv2.imwrite(str(out_name), frame)
                train_saved += 1
            elif i in test_indices:
                if grayscale:
                    out_name = test_dir / f"{video_path.stem}_gray_{test_saved:06d}.jpg"
                else:
                    out_name = test_dir / f"{video_path.stem}_{test_saved:06d}.jpg"
                cv2.imwrite(str(out_name), frame)
                test_saved += 1
            elif i in valid_indices:
                if grayscale:
                    out_name = valid_dir / f"{video_path.stem}_gray_{valid_saved:06d}.jpg"
                else:
                    out_name = valid_dir / f"{video_path.stem}_{valid_saved:06d}.jpg"
                cv2.imwrite(str(out_name), frame)
                valid_saved += 1
        
        self.progress.setVisible(False)
        
        # 顯示分割結果
        mode_text = "灰階" if grayscale else "彩色"
        QMessageBox.information(
            self, 
            "完成", 
            f"已擷取並分割 {total_extracted} 幀（{mode_text}）至:\n"
            f"{base_dir}\n\n"
            f"Train: {train_saved} 幀 ({train_saved/total_extracted*100:.1f}%)\n"
            f"Test: {test_saved} 幀 ({test_saved/total_extracted*100:.1f}%)\n"
            f"Valid: {valid_saved} 幀 ({valid_saved/total_extracted*100:.1f}%)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    window = VideoFrameExtractor()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
