import json
import sys
import os
import shutil
import yaml
import random
import cv2
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QPixmap, QUndoStack, QPolygonF, QPen, QBrush, QColor, QTransform
from PyQt6.QtCore import QPointF
import constants
from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QListWidget, QListWidgetItem, QSplitter,
    QDockWidget, QToolBar, QMessageBox, QStatusBar, QDialog, QVBoxLayout,
    QHBoxLayout, QLabel, QSpinBox, QPushButton, QGroupBox, QGridLayout, QRadioButton, QLineEdit, QCheckBox
)
import scene
import viewer
import table
import commands

class DatasetSplitDialog(QDialog):
    """對話框用於選擇資料集分割比例和匯出選項"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("匯出標記資料")
        self.setModal(True)
        
        # 初始化控件
        # 匯出格式選擇 (目前只支援YOLO，但UI上顯示)
        self.radio_yolo = QRadioButton("YOLO")
        self.radio_yolo.setChecked(True) # 預設選中YOLO
        self.radio_coco = QRadioButton("COCO")
        self.radio_coco.setEnabled(False) # 暫時禁用
        self.radio_csv = QRadioButton("CSV")
        self.radio_csv.setEnabled(False) # 暫時禁用
        
        self.output_dir_label = QLabel("輸出目錄:")
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("瀏覽")
        self.output_dir_button.clicked.connect(self._select_output_directory)

        self.spin_train = QSpinBox()
        self.spin_train.setRange(0, 100)
        self.spin_train.setValue(70)
        self.spin_train.setSuffix("%")
        
        self.spin_test = QSpinBox()
        self.spin_test.setRange(0, 100)
        self.spin_test.setValue(20)
        self.spin_test.setSuffix("%")
        
        self.spin_valid = QSpinBox()
        self.spin_valid.setRange(0, 100)
        self.spin_valid.setValue(10)
        self.spin_valid.setSuffix("%")

        self.radio_random_split = QRadioButton("隨機分割")
        self.radio_sort_by_filename = QRadioButton("依檔名排序")
        self.radio_random_split.setChecked(True) # 預設隨機分割
        
        self.check_reviewed_only = QCheckBox("僅匯出已審核的標記")
        self.check_include_stats = QCheckBox("包含統計報告")
        self.check_include_quality = QCheckBox("包含品質報告")
        
        # 設置佈局
        layout = QVBoxLayout(self)
        
        # 匯出格式群組
        format_group = QGroupBox("匯出格式")
        format_layout = QHBoxLayout(format_group)
        format_layout.addWidget(self.radio_yolo)
        format_layout.addWidget(self.radio_coco)
        format_layout.addWidget(self.radio_csv)
        layout.addWidget(format_group)

        # 輸出目錄群組
        output_dir_group = QGroupBox("輸出設定")
        output_dir_layout = QHBoxLayout(output_dir_group)
        output_dir_layout.addWidget(self.output_dir_label)
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_button)
        layout.addWidget(output_dir_group)
        
        # 比例設定群組
        ratio_group = QGroupBox("資料分割比例 (%)")
        ratio_layout = QGridLayout(ratio_group)
        ratio_layout.addWidget(QLabel("訓練集:"), 0, 0)
        ratio_layout.addWidget(self.spin_train, 0, 1)
        ratio_layout.addWidget(QLabel("驗證集:"), 0, 2)
        ratio_layout.addWidget(self.spin_valid, 0, 3)
        ratio_layout.addWidget(QLabel("測試集:"), 0, 4)
        ratio_layout.addWidget(self.spin_test, 0, 5)
        
        split_method_layout = QHBoxLayout()
        split_method_layout.addWidget(self.radio_random_split)
        split_method_layout.addWidget(self.radio_sort_by_filename)
        ratio_layout.addLayout(split_method_layout, 1, 0, 1, 6)
        
        layout.addWidget(ratio_group)
        
        # 標記選項群組
        options_group = QGroupBox("標記選項")
        options_layout = QVBoxLayout(options_group)
        options_layout.addWidget(self.check_reviewed_only)
        options_layout.addWidget(self.check_include_stats)
        options_layout.addWidget(self.check_include_quality)
        layout.addWidget(options_group)

        # 按鈕
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("匯出")
        self.cancel_button = QPushButton("取消")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        # 連接信號
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.spin_train.valueChanged.connect(self._adjust_ratios)
        self.spin_test.valueChanged.connect(self._adjust_ratios)
        self.spin_valid.valueChanged.connect(self._adjust_ratios)

    def _select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇輸出目錄")
        if directory:
            self.output_dir_edit.setText(directory)
    
    def _adjust_ratios(self):
        """確保三個比例總和為100%"""
        sender = self.sender()
        if sender == self.spin_train:
            remaining = 100 - self.spin_train.value()
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
            current_train_test = self.spin_train.value() + self.spin_test.value()
            if current_train_test > 0:
                train_ratio = self.spin_train.value() / current_train_test
                self.spin_train.setValue(int(remaining * train_ratio))
                self.spin_test.setValue(remaining - self.spin_train.value())
            else:
                self.spin_train.setValue(int(remaining * 0.8))
                self.spin_test.setValue(remaining - self.spin_train.value())
    
    def get_export_options(self):
        """返回選擇的匯出選項"""
        return {
            "train_ratio": self.spin_train.value() / 100.0,
            "test_ratio": self.spin_test.value() / 100.0,
            "valid_ratio": self.spin_valid.value() / 100.0,
            "output_dir": self.output_dir_edit.text(),
            "split_method": "random" if self.radio_random_split.isChecked() else "sorted",
            "export_reviewed_only": self.check_reviewed_only.isChecked(),
            "include_stats_report": self.check_include_stats.isChecked(),
            "include_quality_report": self.check_include_quality.isChecked(),
            "export_format": "yolo" # 目前只支援yolo
        }

class MainWindow(QMainWindow):
    """Main application window."""
    def __init__(self, folder: Path = None):
        super().__init__()
        self.setWindowTitle("Badminton Court Annotator v0.2")
        self.resize(1200, 800)

        self.folder = folder
        self.images = []
        self.current_index = -1
        self.modified = False

        self.undo_stack = QUndoStack(self)
        self.undo_stack.indexChanged.connect(lambda: (self.notify_modified(), self.update_table()))

        self.scene = scene.AnnotationScene(self)
        self.viewer = viewer.ImageViewer(self.scene)
        splitter = QSplitter()
        self.setCentralWidget(splitter)

        self.list_widget = QListWidget()
        splitter.addWidget(self.list_widget)
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(1, 1)

        self.point_table = table.PointTable()
        dock = QDockWidget("Points", self)
        dock.setWidget(self.point_table)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # Statistics labels
        self.total_images_label = QLabel("Total Images: 0")
        self.current_image_label = QLabel("Current Image: 0/0")
        self.points_count_label = QLabel("Points: 0")
        self.t_junction_count_label = QLabel("T-junction: 0")
        self.cross_count_label = QLabel("Cross: 0")
        self.l_corner_count_label = QLabel("L-corner: 0")

        self.status.addPermanentWidget(self.total_images_label)
        self.status.addPermanentWidget(self.current_image_label)
        self.status.addPermanentWidget(self.points_count_label)
        self.status.addPermanentWidget(self.t_junction_count_label)
        self.status.addPermanentWidget(self.cross_count_label)
        self.status.addPermanentWidget(self.l_corner_count_label)

        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.setSingleShot(True)
        self._auto_save_timer.timeout.connect(self.save_project_if_needed)
        # Homography mode state
        self.homography_mode = False
        self.homography_src_points: list[QPointF] = []
        self.homography_point_items: list = []
        self.homo_overlay = None
        # Default point type for new annotations
        self.default_ptype = 0

        self._build_actions()
        if folder:
            self.load_folder(folder)

        self.list_widget.currentRowChanged.connect(self.show_image_by_index)
        self.viewer.scene().selectionChanged.connect(self.sync_table_selection)

    def _build_actions(self):
        open_act = QAction("&Open Folder", self, shortcut=QKeySequence.StandardKey.Open,
                           triggered=self.open_folder_dialog)
        save_act = QAction("&Save", self, shortcut=QKeySequence.StandardKey.Save,
                           triggered=self.save_project_if_needed)
        export_act = QAction("&Export Current CSV", self, triggered=self.export_current_csv)
        export_yolo_act = QAction("Export Current &YOLO", self, triggered=self.export_current_yolo)
        export_batch_act = QAction("Export &Batch YOLO", self, triggered=self.export_batch_yolo)
        undo_act = self.undo_stack.createUndoAction(self, "&Undo")
        undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        redo_act = self.undo_stack.createRedoAction(self, "&Redo")
        redo_act.setShortcut(QKeySequence.StandardKey.Redo)

        # Homography mode toggle
        self.homography_act = QAction("&Homography", self, checkable=True,
                                     shortcut=QKeySequence("H"),
                                     triggered=self.toggle_homography)
        # Grid toggle
        self.grid_act = QAction("&Grid", self, checkable=True,
                                shortcut=QKeySequence("G"),
                                triggered=self.toggle_grid)
        # top toolbar with file and undo/redo actions
        toolbar = QToolBar("Main")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        toolbar.addActions([open_act, save_act, export_act, export_yolo_act, export_batch_act, undo_act, redo_act])
        # magnifier zoom controls
        mag_in_act = QAction("Zoom+ (Magnifier)", self,
                             shortcut=QKeySequence("+"),
                             triggered=lambda: self.viewer.increase_magnifier_zoom())
        mag_out_act = QAction("Zoom- (Magnifier)", self,
                              shortcut=QKeySequence("-"),
                              triggered=lambda: self.viewer.decrease_magnifier_zoom())
        # prepare side toolbar actions (tools panel)
        # Default point type selector (radio actions)
        # secondary tools toolbar on right
        side_toolbar = QToolBar("Tools")
        side_toolbar.setOrientation(Qt.Orientation.Vertical)
        self.addToolBar(Qt.ToolBarArea.RightToolBarArea, side_toolbar)
        # Homography toggle
        side_toolbar.addAction(self.homography_act)
        # Grid toggle
        side_toolbar.addAction(self.grid_act)
        # Magnifier zoom controls
        side_toolbar.addAction(mag_in_act)
        side_toolbar.addAction(mag_out_act)
        # Default point type selector (radio actions)
        type_group = QActionGroup(self)
        for t, name in constants.TYPE_NAMES.items():
            act = QAction(f"Type {t}: {name}", self, checkable=True,
                          triggered=lambda _, tt=t: self.set_default_ptype(tt))
            type_group.addAction(act)
            side_toolbar.addAction(act)
            if t == self.default_ptype:
                act.setChecked(True)

    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            self.load_folder(Path(folder))

    def load_folder(self, folder: Path):
        supported = {".jpg", ".jpeg", ".png"}
        self.images = sorted([p for p in folder.iterdir() if p.suffix.lower() in supported])
        if not self.images:
            QMessageBox.warning(self, "No images", "Folder contains no supported images.")
            return
        self.folder = folder
        self.list_widget.clear()
        for p in self.images:
            self.list_widget.addItem(QListWidgetItem(p.name))
        self.current_index = 0
        self.list_widget.setCurrentRow(0)
        self.show_image_by_index(0)
        self.total_images_label.setText(f"Total Images: {len(self.images)}")

    def show_image_by_index(self, idx: int):
        if idx < 0 or idx >= len(self.images):
            return
        if self.modified:
            self.save_project_if_needed()
        self.current_index = idx
        img_path = self.images[idx]
        pix = QPixmap(str(img_path))
        self.scene.load_image(pix)
        self.load_project()
        self.status.showMessage(f"Loaded {img_path.name}")
        self.current_image_label.setText(f"Current Image: {self.current_index + 1}/{len(self.images)}")
        self.update_table()
        self.undo_stack.clear()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_PageUp):
            self.list_widget.setCurrentRow(max(0, self.current_index - 1))
        elif event.key() in (Qt.Key.Key_Down, Qt.Key.Key_PageDown):
            self.list_widget.setCurrentRow(min(len(self.images) - 1, self.current_index + 1))
        else:
            super().keyPressEvent(event)

    def project_file(self) -> Path:
        return self.images[self.current_index].with_suffix(constants.PROJECT_EXT)

    def notify_modified(self):
        self.modified = True
        self._auto_save_timer.start(constants.AUTO_SAVE_MS)
        self.update_table()

    def load_project(self):
        pj = self.project_file()
        if pj.exists():
            try:
                with open(pj, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.scene.from_dict(data["points"])
            except Exception as ex:
                self.status.showMessage(f"Failed to load project: {ex}")
        self.modified = False

    def save_project_if_needed(self):
        if not self.modified:
            return
        pj = self.project_file()
        try:
            data = {"points": self.scene.to_dict()}
            tmp = pj.with_suffix(pj.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(pj)
            self.status.showMessage(f"Saved {pj.name}")
            self.modified = False
        except Exception as ex:
            backup = pj.with_suffix(pj.suffix + ".backup")
            tmp.replace(backup)
            QMessageBox.critical(self, "Save failed",
                                 f"Could not save project. Backup at {backup}\n{ex}")

    def update_table(self):
        self.point_table.load_points(self.scene.points)
        # Update statistics
        self.points_count_label.setText(f"Points: {len(self.scene.points)}")
        t_junction_count = sum(1 for p in self.scene.points if p.ptype == 0)
        cross_count = sum(1 for p in self.scene.points if p.ptype == 1)
        l_corner_count = sum(1 for p in self.scene.points if p.ptype == 2)
        self.t_junction_count_label.setText(f"T-junction: {t_junction_count}")
        self.cross_count_label.setText(f"Cross: {cross_count}")
        self.l_corner_count_label.setText(f"L-corner: {l_corner_count}")

    def sync_table_selection(self):
        sel_ids = {p.pid for p in self.scene.selectedItems() if hasattr(p, "pid")}
        self.point_table.blockSignals(True)
        self.point_table.clearSelection()
        for row in range(self.point_table.rowCount()):
            pid = int(self.point_table.item(row, 0).text())
            if pid in sel_ids:
                self.point_table.selectRow(row)
        self.point_table.blockSignals(False)

    def export_current_csv(self):
        img = self.images[self.current_index]
        out = img.with_suffix(".csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            f.write("id,x,y,type\n")
            for p in self.scene.points:
                f.write(f"{p.pid},{p.pos().x():.1f},{p.pos().y():.1f},{p.ptype}\n")
        self.status.showMessage(f"Exported {out.name}")

    def export_current_yolo(self):
        """Export current image points in YOLO format"""
        if self.current_index < 0 or not self.scene.image_item:
            QMessageBox.warning(self, "Export Error", "No image loaded")
            return
        
        img = self.images[self.current_index]
        out = img.with_suffix(".txt")
        
        # Get image dimensions
        pixmap = self.scene.image_item.pixmap()
        img_width = pixmap.width()
        img_height = pixmap.height()
        
        with open(out, "w", encoding="utf-8") as f:
            for p in self.scene.points:
                # Convert to normalized coordinates [0, 1]
                x_norm = p.pos().x() / img_width
                y_norm = p.pos().y() / img_height
                
                # Ensure coordinates are within [0, 1] range
                x_norm = max(0, min(1, x_norm))
                y_norm = max(0, min(1, y_norm))
                
                # Write YOLO format: class_id x_center y_center width height kpt_x kpt_y kpt_visibility
                # width and height are fixed to 0.01 as per specification for point detection
                # kpt_x and kpt_y are the same as x_center and y_center
                # kpt_visibility is fixed to 2 (visible and labeled)
                f.write(f"{p.ptype} {x_norm:.6f} {y_norm:.6f} 0.010000 0.010000 {x_norm:.6f} {y_norm:.6f} 2\n")
        
        self.status.showMessage(f"Exported YOLO format: {out.name}")

    def export_batch_yolo(self):
        """Export all images with points in YOLO dataset format"""
        if not self.images:
            QMessageBox.warning(self, "Export Error", "No images loaded")
            return
        
        # Show dataset split dialog
        dialog = DatasetSplitDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        export_options = dialog.get_export_options()
        train_ratio = export_options["train_ratio"]
        test_ratio = export_options["test_ratio"]
        valid_ratio = export_options["valid_ratio"]
        output_dir = export_options["output_dir"]
        split_method = export_options["split_method"]
        export_reviewed_only = export_options["export_reviewed_only"]
        include_stats_report = export_options["include_stats_report"]
        include_quality_report = export_options["include_quality_report"]

        if not output_dir:
            QMessageBox.warning(self, "Export Error", "Output directory not selected.")
            return
        
        output_path = Path(output_dir)
        
        # Create YOLO dataset structure
        dataset_root = output_path / "badminton_dataset"
        images_dir = dataset_root / "images"
        labels_dir = dataset_root / "labels"
        
        # Create train/test/valid directories
        train_images_dir = images_dir / "train"
        test_images_dir = images_dir / "test"
        valid_images_dir = images_dir / "valid"
        train_labels_dir = labels_dir / "train"
        test_labels_dir = labels_dir / "test"
        valid_labels_dir = labels_dir / "valid"
        
        for dir_path in [train_images_dir, test_images_dir, valid_images_dir, 
                        train_labels_dir, test_labels_dir, valid_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Split data according to user-specified ratios and method
        total_images = len(self.images)
        indices = list(range(total_images))
        
        if split_method == "random":
            random.shuffle(indices)
        else: # "sorted" by filename
            # Images are already sorted by filename due to self.images = sorted([...])
            pass
        
        train_count = int(total_images * train_ratio)
        test_count = int(total_images * test_ratio)
        valid_count = total_images - train_count - test_count
        
        train_indices = set(indices[:train_count])
        test_indices = set(indices[train_count:train_count + test_count])
        valid_indices = set(indices[train_count + test_count:])
        
        exported_count = 0
        train_exported = test_exported = valid_exported = 0
        
        for idx, img_path in enumerate(self.images):
            # Determine which set this image belongs to
            if idx in train_indices:
                target_images_dir = train_images_dir
                target_labels_dir = train_labels_dir
                split_type = "train"
            elif idx in test_indices:
                target_images_dir = test_images_dir
                target_labels_dir = test_labels_dir
                split_type = "test"
            else:
                target_images_dir = valid_images_dir
                target_labels_dir = valid_labels_dir
                split_type = "valid"
            
            # Load and resize image to 640x640
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
                
            original_height, original_width = img.shape[:2]
            
            # Resize image to 640x640 (may distort aspect ratio for better training)
            resized_img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
            
            # Save resized image
            target_img_path = target_images_dir / img_path.name
            cv2.imwrite(str(target_img_path), resized_img)
            
            # Load project file for this image
            project_file = img_path.with_suffix(constants.PROJECT_EXT)
            points_data = []
            
            if project_file.exists():
                try:
                    with open(project_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        points_data = data.get("points", [])
                        # Filter by reviewed status if option is enabled
                        if export_reviewed_only:
                            # Assuming 'reviewed' status is part of point data, if not, this will need actual implementation
                            points_data = [p for p in points_data if p.get("reviewed", False)]
                except Exception as ex:
                    print(f"Warning: Failed to load project for {img_path.name}: {ex}")
            
            # Create YOLO label file
            label_file = target_labels_dir / (img_path.stem + ".txt")
            
            if points_data:
                with open(label_file, "w", encoding="utf-8") as f:
                    for point in points_data:
                        # Transform coordinates to match the resized 640x640 image
                        # Original coordinates are based on the original image size
                        original_x = point["x"]
                        original_y = point["y"]
                        
                        # Scale coordinates to 640x640
                        scaled_x = original_x * (640 / original_width)
                        scaled_y = original_y * (640 / original_height)
                        
                        # Convert to normalized coordinates [0, 1] for YOLO format
                        x_norm = scaled_x / 640
                        y_norm = scaled_y / 640
                        
                        # Ensure coordinates are within [0, 1] range
                        x_norm = max(0, min(1, x_norm))
                        y_norm = max(0, min(1, y_norm))
                        
                        # Write YOLO format: class_id x_center y_center width height kpt_x kpt_y kpt_visibility
                        # width and height are fixed to 0.01 as per specification for point detection
                        # kpt_x and kpt_y are the same as x_center and y_center
                        # kpt_visibility is fixed to 2 (visible and labeled)
                        f.write(f"{point['type']} {x_norm:.6f} {y_norm:.6f} 0.010000 0.010000 {x_norm:.6f} {y_norm:.6f} 2\n")
                
                exported_count += 1
                if split_type == "train":
                    train_exported += 1
                elif split_type == "test":
                    test_exported += 1
                else:
                    valid_exported += 1
            else:
                # Create empty label file for images without annotations
                label_file.touch()
        
        # Generate data.yaml configuration file
        self.generate_data_yaml(dataset_root, train_count, test_count, valid_count)

        # Placeholder for generating reports
        if include_stats_report:
            print("Generating statistics report...") # Placeholder
        if include_quality_report:
            print("Generating quality report...") # Placeholder
        
        QMessageBox.information(
            self, 
            "Export Complete", 
            f"Exported {exported_count}/{total_images} images with annotations to:\n{dataset_root}\n\n"
            f"Train: {train_count} images ({train_exported} with annotations)\n"
            f"Test: {test_count} images ({test_exported} with annotations)\n"
            f"Valid: {valid_count} images ({valid_exported} with annotations)"
        )
        self.status.showMessage(f"Batch export completed: {dataset_root}")

    def generate_data_yaml(self, dataset_root: Path, train_count: int, test_count: int, valid_count: int):
        """Generate data.yaml configuration file for YOLO training"""
        data_config = {
            'train': 'images/train',
            'test': 'images/test', 
            'val': 'images/valid',
            'nc': 3,  # number of classes
            'names': {
                0: 'L-corner',
                1: 'Cross', 
                2: 'T-junction'
            },
            'task': 'point_detect'
        }
        
        yaml_file = dataset_root / 'data.yaml'
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        # Also create a README file with dataset info
        total_images = train_count + test_count + valid_count
        readme_content = f"""# Badminton Court Point Detection Dataset

## Dataset Information
- Total images: {total_images}
- Training images: {train_count} ({train_count/total_images*100:.1f}%)
- Test images: {test_count} ({test_count/total_images*100:.1f}%)
- Validation images: {valid_count} ({valid_count/total_images*100:.1f}%)
- Image size: 640x640 pixels (resized from original)
- Classes: 3 (L-corner, Cross, T-junction)

## Directory Structure
```
badminton_dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── test/
│   └── valid/
└── labels/
    ├── train/
    ├── test/
    └── valid/
```

## Image Processing
- All images are resized to 640x640 pixels for optimal YOLO training
- Point coordinates are automatically scaled to match the resized images
- Original aspect ratios may be altered for consistent input size

## Class Definitions
- **Class 0 (L-corner)**: 直角點 - 兩條直線以90度角相交的端點
- **Class 1 (Cross)**: 十字交叉點 - 兩條直線垂直相交的交點  
- **Class 2 (T-junction)**: T字型交點 - 一條直線與另一條直線中點相交

## Label Format
YOLO format: `class_id x_center y_center`
- Coordinates are normalized to [0, 1] range relative to 640x640 image
- Each line represents one point annotation
"""
        
        readme_file = dataset_root / 'README.md'
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)

    def delete_selected(self):
        pts = [it for it in self.scene.selectedItems() if hasattr(it, "pid")]
        if pts:
            self.undo_stack.push(commands.DeletePointCommand(self.scene, pts))

    def change_type_selected(self, new_t: int):
        pts = [it for it in self.scene.selectedItems() if hasattr(it, "ptype")]
        for p in pts:
            if p.ptype != new_t:
                self.undo_stack.push(commands.ChangeTypeCommand(p, p.ptype, new_t))
    def set_default_ptype(self, t: int):
        """Set the default point type for new annotations."""
        self.default_ptype = t
        self.status.showMessage(f"Default point type set to {constants.TYPE_NAMES[t]}")
    def toggle_homography(self, checked: bool):
        """Enable or disable homography mode."""
        self.homography_mode = checked
        # clear existing overlay and homography markers
        if self.homo_overlay:
            self.scene.removeItem(self.homo_overlay)
            self.homo_overlay = None
        # clear initial homography click markers
        for item in self.homography_point_items:
            self.scene.removeItem(item)
        self.homography_point_items.clear()
        # reset source points
        self.homography_src_points = []
        if checked:
            self.status.showMessage(
                "Homography mode: click bottom-left, bottom-right, top-right, top-left"
            )
        else:
            self.status.showMessage("Exited homography mode")

    def toggle_grid(self, checked: bool):
        """Enable or disable grid overlay."""
        self.viewer.grid_enabled = checked
        self.viewer.viewport().update() # Trigger repaint
        self.status.showMessage(f"Grid overlay: {'Enabled' if checked else 'Disabled'}")

    def on_homography_click(self, pos: QPointF):
        """Record a homography reference point."""
        # draw a temporary marker for this homography click
        r = constants.POINT_RADIUS
        pen = QPen(QColor(0, 255, 255), 2)
        brush = QBrush(QColor(0, 255, 255, 100))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        ellipse = self.scene.addEllipse(
            pos.x() - r, pos.y() - r, 2 * r, 2 * r, pen, brush
        )
        self.homography_point_items.append(ellipse)
        # record the source point
        self.homography_src_points.append(pos)
        count = len(self.homography_src_points)
        self.status.showMessage(f"Selected homography point {count}/4")
        if count == 4:
            self.finish_homography()

    def finish_homography(self):
        """Compute and apply homography to generate standard court points."""
        # compute perspective transform from template to image
        src_poly = QPolygonF([QPointF(x, y) for x, y in constants.HOMOGRAPHY_TEMPLATE])
        dst_poly = QPolygonF(self.homography_src_points)
        # compute QTransform mapping from src quadrilateral to dst quadrilateral
        transform = QTransform()
        if not QTransform.quadToQuad(src_poly, dst_poly, transform):
            self.status.showMessage("Failed to compute homography")
            self.homography_mode = False
            self.homography_act.setChecked(False)
            return
        # overlay border polygon
        pen = QPen(QColor(255, 255, 0), 2)
        brush = QBrush()
        brush.setStyle(Qt.BrushStyle.NoBrush)
        self.homo_overlay = self.scene.addPolygon(dst_poly, pen, brush)
        # remove temporary click markers
        for item in self.homography_point_items:
            self.scene.removeItem(item)
        self.homography_point_items.clear()
        # generate template points with auto-assigned types
        self.undo_stack.beginMacro("Apply homography")
        for idx, (x, y) in enumerate(constants.TEMPLATE_POINTS):
            mapped = transform.map(QPointF(x, y))
            ptype = constants.TEMPLATE_TYPES[idx]
            self.undo_stack.push(
                commands.AddPointCommand(self.scene, mapped, ptype)
            )
        self.undo_stack.endMacro()
        self.notify_modified()
        self.update_table()
        # exit homography mode
        self.homography_mode = False
        self.homography_act.setChecked(False)
        self.status.showMessage("Homography applied: generated standard points")
