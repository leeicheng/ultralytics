import json
import yaml
import random
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QPixmap, QUndoStack, QPolygonF, QPen, QBrush, QColor, QTransform, QImage, QPainter
from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QListWidget, QListWidgetItem, QSplitter,
    QDockWidget, QToolBar, QMessageBox, QStatusBar, QDialog, QVBoxLayout,
    QHBoxLayout, QLabel, QSpinBox, QPushButton, QGroupBox, QGridLayout, QRadioButton, QLineEdit, QCheckBox, QSlider,
    QGraphicsScene, QGraphicsView, QProgressDialog
)
import albumentations as A

import viewer
import table
from tools.label_tool import commands, constants, scene

def qimage_to_cv(qimage: QImage) -> np.ndarray:
    """Convert QImage to OpenCV BGR format."""
    qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    ptr.setsize(height * width * 3)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    """Convert OpenCV image to QPixmap."""
    if len(cv_img.shape) == 3:
        h, w, ch = cv_img.shape
        if ch == 3:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else: # Grayscale
            bytes_per_line = w
            convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
    else: # Grayscale
        h, w = cv_img.shape
        bytes_per_line = w
        convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        
    return QPixmap.fromImage(convert_to_Qt_format)

class AugmentationPreviewDialog(QDialog):
    """A dialog to show side-by-side previews of augmented images."""
    def __init__(self, preview_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("增強效果預覽")
        self.setMinimumSize(1200, 400)

        main_layout = QHBoxLayout(self)

        for name, cv_img, points in preview_data:
            container = QGroupBox(name)
            container_layout = QVBoxLayout(container)

            scene = QGraphicsScene()
            view = QGraphicsView(scene)
            view.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)

            pixmap = cv_to_qpixmap(cv_img)
            scene.addPixmap(pixmap)
            view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

            for point in points:
                x, y, ptype, visibility = point['x'], point['y'], point['type'], point['visibility']
                color = constants.TYPE_COLORS[ptype]
                pen = QPen(color)
                if visibility == constants.VISIBILITY_OCCLUDED:
                    pen.setStyle(Qt.PenStyle.DashLine)
                brush = QBrush(color)
                radius = constants.POINT_RADIUS * 2
                scene.addEllipse(x - radius, y - radius, 2 * radius, 2 * radius, pen, brush)

            container_layout.addWidget(view)
            main_layout.addWidget(container)

class DatasetSplitDialog(QDialog):
    """對話框用於選擇資料集分割比例和匯出選項"""
    def __init__(self, main_window: 'MainWindow', parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("匯出標記資料")
        self.setModal(True)
        
        self.radio_yolo = QRadioButton("YOLO")
        self.radio_yolo.setChecked(True)
        self.output_dir_edit = QLineEdit()
        self.output_dir_button = QPushButton("瀏覽")
        self.spin_train = QSpinBox()
        self.spin_train.setRange(0, 100); self.spin_train.setValue(70)
        self.spin_test = QSpinBox()
        self.spin_test.setRange(0, 100); self.spin_test.setValue(20)
        self.spin_valid = QSpinBox()
        self.spin_valid.setRange(0, 100); self.spin_valid.setValue(10)
        self.radio_random_split = QRadioButton("隨機分割"); self.radio_random_split.setChecked(True)
        self.radio_sort_by_filename = QRadioButton("依檔名排序")
        
        layout = QVBoxLayout(self)
        
        output_dir_group = QGroupBox("輸出設定")
        output_dir_layout = QHBoxLayout(output_dir_group)
        output_dir_layout.addWidget(QLabel("輸出目錄:"))
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_button)
        layout.addWidget(output_dir_group)

        self.aug_group = QGroupBox("資料強化 (Data Augmentation)")
        self.aug_group.setCheckable(True)
        self.aug_group.setChecked(False)
        aug_layout = QGridLayout(self.aug_group)
        
        self.rot_label = QLabel("旋轉角度:")
        self.rot_slider = QSlider(Qt.Orientation.Horizontal)
        self.rot_slider.setRange(-30, 30); self.rot_slider.setValue(0)
        self.rot_value_label = QLabel("0°")
        
        self.dist_label = QLabel("桶狀/枕狀變形:")
        self.dist_slider = QSlider(Qt.Orientation.Horizontal)
        self.dist_slider.setRange(-50, 50); self.dist_slider.setValue(0)
        self.dist_value_label = QLabel("0.00")

        self.preview_button = QPushButton("預覽")

        aug_layout.addWidget(self.rot_label, 0, 0)
        aug_layout.addWidget(self.rot_slider, 0, 1)
        aug_layout.addWidget(self.rot_value_label, 0, 2)
        aug_layout.addWidget(self.dist_label, 1, 0)
        aug_layout.addWidget(self.dist_slider, 1, 1)
        aug_layout.addWidget(self.dist_value_label, 1, 2)
        aug_layout.addWidget(self.preview_button, 2, 0, 1, 3)
        
        layout.addWidget(self.aug_group)

        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("匯出")
        self.cancel_button = QPushButton("取消")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        self.output_dir_button.clicked.connect(self._select_output_directory)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.rot_slider.valueChanged.connect(self._update_aug_labels)
        self.dist_slider.valueChanged.connect(self._update_aug_labels)
        self.aug_group.toggled.connect(self._toggle_aug_controls)
        self.preview_button.clicked.connect(self.show_preview)

        self._toggle_aug_controls(False)
        self._update_aug_labels()

    def _toggle_aug_controls(self, checked):
        for widget in [self.rot_label, self.rot_slider, self.rot_value_label,
                       self.dist_label, self.dist_slider, self.dist_value_label, self.preview_button]:
            widget.setEnabled(checked)

    def _update_aug_labels(self):
        self.rot_value_label.setText(f"{self.rot_slider.value()}°")
        self.dist_value_label.setText(f"{self.dist_slider.value() / 100.0:.2f}")

    def show_preview(self):
        if not self.main_window:
            return
        
        rot = self.rot_slider.value()
        dist = self.dist_slider.value() / 100.0
        
        preview_data = self.main_window.generate_preview_data(rot, dist)
        if not preview_data:
            QMessageBox.warning(self, "預覽錯誤", "沒有可供預覽的圖片或標記。")
            return

        preview_dialog = AugmentationPreviewDialog(preview_data, self)
        preview_dialog.exec()

    def _select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇輸出目錄")
        if directory:
            self.output_dir_edit.setText(directory)
    
    def get_export_options(self):
        options = {
            "train_ratio": self.spin_train.value() / 100.0,
            "test_ratio": self.spin_test.value() / 100.0,
            "valid_ratio": self.spin_valid.value() / 100.0,
            "output_dir": self.output_dir_edit.text(),
            "split_method": "random" if self.radio_random_split.isChecked() else "sorted",
            "augmentation": {
                "enabled": self.aug_group.isChecked(),
                "rotation": self.rot_slider.value(),
                "distortion": self.dist_slider.value() / 100.0
            }
        }
        return options

class MainWindow(QMainWindow):
    """Main application window."""
    def __init__(self, folder: Path = None):
        super().__init__()
        self.setWindowTitle("Badminton Court Annotator v0.7")
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
        
        self.homography_mode = False
        self.homography_src_points: list[QPointF] = []
        self.homography_point_items: list = []
        self.homo_overlay = None
        
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
        apply_to_all_act = QAction("Apply to All", self,
                                   triggered=self.confirm_apply_to_all)
        apply_to_all_act.setToolTip("Apply the current annotations to all other images in the folder.")
        
        export_act = QAction("&Export Current CSV", self, triggered=self.export_current_csv)
        export_yolo_act = QAction("Export Current &YOLO", self, triggered=self.export_current_yolo)
        export_batch_act = QAction("Export &Batch YOLO", self, triggered=self.export_batch_yolo)
        undo_act = self.undo_stack.createUndoAction(self, "&Undo")
        undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        redo_act = self.undo_stack.createRedoAction(self, "&Redo")
        redo_act.setShortcut(QKeySequence.StandardKey.Redo)

        self.homography_act = QAction("&Homography", self, checkable=True,
                                     shortcut=QKeySequence("H"),
                                     triggered=self.toggle_homography)
        self.grid_act = QAction("&Grid", self, checkable=True,
                                shortcut=QKeySequence("G"),
                                triggered=self.toggle_grid)
        self.toggle_visibility_act = QAction("Toggle &Visibility", self, 
                                           shortcut=QKeySequence("V"), 
                                           triggered=self.toggle_visibility_selected)

        toolbar = QToolBar("Main")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        toolbar.addActions([open_act, save_act, apply_to_all_act, export_act, export_yolo_act, export_batch_act, undo_act, redo_act])
        
        mag_in_act = QAction("Zoom+ (Magnifier)", self,
                             shortcut=QKeySequence("+"),
                             triggered=lambda: self.viewer.increase_magnifier_zoom())
        mag_out_act = QAction("Zoom- (Magnifier)", self,
                              shortcut=QKeySequence("-"),
                              triggered=lambda: self.viewer.decrease_magnifier_zoom())
        
        side_toolbar = QToolBar("Tools")
        side_toolbar.setOrientation(Qt.Orientation.Vertical)
        self.addToolBar(Qt.ToolBarArea.RightToolBarArea, side_toolbar)
        side_toolbar.addAction(self.homography_act)
        side_toolbar.addAction(self.grid_act)
        side_toolbar.addAction(self.toggle_visibility_act)
        side_toolbar.addAction(mag_in_act)
        side_toolbar.addAction(mag_out_act)
        
        type_group = QActionGroup(self)
        for t, name in constants.TYPE_NAMES.items():
            act = QAction(f"Type {t}: {name}", self, checkable=True,
                          triggered=lambda _, tt=t: self.set_default_ptype(tt))
            type_group.addAction(act)
            side_toolbar.addAction(act)
            if t == getattr(self, 'default_ptype', 0):
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
        if self.images:
            self.list_widget.setCurrentRow(0)

    def show_image_by_index(self, idx: int):
        if 0 <= idx < len(self.images):
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
        self.scene.clear_points()
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

    def confirm_apply_to_all(self):
        if self.current_index < 0:
            QMessageBox.warning(self, "No Image Selected", "Please select an image first.")
            return

        reply = QMessageBox.question(self, "Confirm Action",
                                     "This will overwrite the annotations for all other images in this folder "
                                     "with the points from the current image.\n\n"
                                     "<b>This action cannot be undone.</b><br><br>"
                                     "Are you sure you want to proceed?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                     QMessageBox.StandardButton.Cancel)

        if reply == QMessageBox.StandardButton.Yes:
            self.apply_to_all()

    def apply_to_all(self):
        current_points = self.scene.to_dict()
        if not current_points:
            QMessageBox.warning(self, "No Annotations", "There are no annotations on the current image to apply.")
            return

        current_img_path = self.images[self.current_index]
        other_images = [img for img in self.images if img != current_img_path]
        
        progress = QProgressDialog("Applying annotations to all images...", "Cancel", 0, len(other_images), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        for i, img_path in enumerate(other_images):
            progress.setValue(i)
            if progress.wasCanceled():
                break

            pj = img_path.with_suffix(constants.PROJECT_EXT)
            try:
                data = {"points": current_points}
                tmp = pj.with_suffix(pj.suffix + ".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                tmp.replace(pj)
            except Exception as ex:
                QMessageBox.critical(self, "Apply Failed", f"Could not apply annotations to {img_path.name}.\n{ex}")
                progress.cancel()
                return
        
        progress.setValue(len(other_images))
        QMessageBox.information(self, "Success", f"Successfully applied annotations to {len(other_images)} other images.")


    def update_table(self):
        self.point_table.load_points(self.scene.points)
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
            f.write("id,x,y,type,visibility\n")
            for p in self.scene.points:
                f.write(f"{p.pid},{p.pos().x():.1f},{p.pos().y():.1f},{p.ptype},{p.visibility}\n")
        self.status.showMessage(f"Exported {out.name}")

    def export_current_yolo(self):
        if self.current_index < 0 or not self.scene.image_item:
            QMessageBox.warning(self, "Export Error", "No image loaded")
            return
        
        img = self.images[self.current_index]
        out = img.with_suffix(".txt")
        
        pixmap = self.scene.image_item.pixmap()
        img_width = pixmap.width()
        img_height = pixmap.height()
        
        with open(out, "w", encoding="utf-8") as f:
            for p in self.scene.points:
                x_norm = max(0, min(1, p.pos().x() / img_width))
                y_norm = max(0, min(1, p.pos().y() / img_height))
                f.write(f"{p.ptype} {x_norm:.6f} {y_norm:.6f} 0.010000 0.010000 {x_norm:.6f} {y_norm:.6f} {p.visibility}\n")
        
        self.status.showMessage(f"Exported YOLO format: {out.name}")

    def _apply_single_augmentation(self, img, points, rotation_angle, distortion_k):
        h, w = img.shape[:2]

        alb_keypoints = []
        class_ids = []
        track_ids = []
        visibilities = []
        for p in points:
            alb_keypoints.append((p['x'], p['y']))
            class_ids.append(p['type'])
            track_ids.append(p['id'])
            visibilities.append(p['visibility'])

        transforms_list = []

        if rotation_angle != 0:
            transforms_list.append(A.Rotate(limit=(rotation_angle, rotation_angle), interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, p=1))

        if distortion_k != 0:
            transforms_list.append(A.OpticalDistortion(
                distort_limit=distortion_k,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=1
            ))
        
        if not transforms_list:
            return img, points

        transform = A.Compose(transforms_list, keypoint_params=A.KeypointParams(format='xy', label_fields=['class_ids', 'track_ids', 'visibilities']))

        transformed = transform(image=img, keypoints=alb_keypoints, class_ids=class_ids, track_ids=track_ids, visibilities=visibilities)
        augmented_img = transformed['image']
        
        augmented_points = []
        for i, kp in enumerate(transformed['keypoints']):
            x, y = kp
            class_id = transformed['class_ids'][i]
            track_id = transformed['track_ids'][i]
            visibility = transformed['visibilities'][i]
            if 0 <= x < w and 0 <= y < h:
                augmented_points.append({'id': track_id, 'x': x, 'y': y, 'type': class_id, 'visibility': visibility})
        
        return augmented_img, augmented_points

    def generate_preview_data(self, rotation, distortion):
        if not self.scene.image_item:
            return []
        
        original_pixmap = self.scene.image_item.pixmap()
        cv_img = qimage_to_cv(original_pixmap.toImage())
        points = self.scene.to_dict()

        preview_data = [('Original', cv_img, points)]
        
        # Generate preview for individual augmentations
        if rotation != 0:
            rot_img, rot_points = self._apply_single_augmentation(cv_img, points, rotation, 0)
            preview_data.append((f'Rotation {rotation}°', rot_img, rot_points))

        if distortion > 0: # Barrel
            barrel_img, barrel_points = self._apply_single_augmentation(cv_img, points, 0, distortion)
            preview_data.append((f'Barrel {distortion:.2f}', barrel_img, barrel_points))
        elif distortion < 0: # Pincushion
            pincushion_img, pincushion_points = self._apply_single_augmentation(cv_img, points, 0, distortion)
            preview_data.append((f'Pincushion {distortion:.2f}', pincushion_img, pincushion_points))

        # Generate preview for combined augmentations
        if rotation != 0 and distortion > 0: # Barrel + Rotation
            barrel_rot_img, barrel_rot_points = self._apply_single_augmentation(cv_img, points, rotation, distortion)
            preview_data.append((f'Barrel {distortion:.2f} + Rot {rotation}°', barrel_rot_img, barrel_rot_points))
        
        if rotation != 0 and distortion < 0: # Pincushion + Rotation
            pincushion_rot_img, pincushion_rot_points = self._apply_single_augmentation(cv_img, points, rotation, distortion)
            preview_data.append((f'Pincushion {distortion:.2f} + Rot {rotation}°', pincushion_rot_img, pincushion_rot_points))

        return preview_data

    def export_batch_yolo(self):
        if not self.images:
            QMessageBox.warning(self, "Export Error", "No images loaded")
            return
        
        dialog = DatasetSplitDialog(self, self)
        if not dialog.exec():
            return
        
        export_options = dialog.get_export_options()
        output_dir = export_options["output_dir"]
        if not output_dir:
            QMessageBox.warning(self, "Export Error", "Output directory not selected.")
            return

        dataset_root = Path(output_dir) / "badminton_dataset"
        images_dir = dataset_root / "images"
        labels_dir = dataset_root / "labels"
        
        train_images_dir = images_dir / "train"; train_images_dir.mkdir(parents=True, exist_ok=True)
        test_images_dir = images_dir / "test"; test_images_dir.mkdir(parents=True, exist_ok=True)
        valid_images_dir = images_dir / "val"; valid_images_dir.mkdir(parents=True, exist_ok=True)
        train_labels_dir = labels_dir / "train"; train_labels_dir.mkdir(parents=True, exist_ok=True)
        test_labels_dir = labels_dir / "test"; test_labels_dir.mkdir(parents=True, exist_ok=True)
        valid_labels_dir = labels_dir / "val"; valid_labels_dir.mkdir(parents=True, exist_ok=True)

        total_images = len(self.images)
        indices = list(range(total_images))
        if export_options["split_method"] == "random":
            random.shuffle(indices)
        
        train_count = int(total_images * export_options["train_ratio"])
        test_count = int(total_images * export_options["test_ratio"])
        
        train_indices = set(indices[:train_count])
        test_indices = set(indices[train_count:train_count + test_count])

        progress_dialog = QProgressDialog("匯出資料中...", "取消", 0, total_images, self)
        progress_dialog.setWindowTitle("匯出進度")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(True)
        progress_dialog.show()

        for idx, img_path in enumerate(self.images):
            if progress_dialog.wasCanceled():
                QMessageBox.information(self, "匯出取消", "匯出操作已被使用者取消。")
                break
            
            progress_dialog.setValue(idx + 1)
            progress_dialog.setLabelText(f"處理圖片: {img_path.name} ({idx + 1}/{total_images})")
            
            if idx in train_indices:
                target_images_dir, target_labels_dir, split_type = train_images_dir, train_labels_dir, "train"
            elif idx in test_indices:
                target_images_dir, target_labels_dir, split_type = test_images_dir, test_labels_dir, "test"
            else:
                target_images_dir, target_labels_dir, split_type = valid_images_dir, valid_labels_dir, "val"
            
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            project_file = img_path.with_suffix(constants.PROJECT_EXT)
            points_data = []
            if project_file.exists():
                with open(project_file, "r") as f:
                    points_data = json.load(f).get("points", [])

            images_to_save = [('', img, points_data)] # Start with the original

            if export_options["augmentation"]["enabled"] and points_data and split_type == 'train':
                # Define augmentation parameters for this image
                random_rotation_angle = random.randint(-30, 30)
                random_barrel_k = random.uniform(0.05, 0.5)
                random_pincushion_k = random.uniform(-0.5, -0.05)

                # Pure Rotation
                if random_rotation_angle != 0:
                    rot_img, rot_points = self._apply_single_augmentation(img, points_data, random_rotation_angle, 0)
                    images_to_save.append((f'_rot{random_rotation_angle}', rot_img, rot_points))

                # Pure Barrel Distortion
                if random_barrel_k != 0:
                    barrel_img, barrel_points = self._apply_single_augmentation(img, points_data, 0, random_barrel_k)
                    images_to_save.append((f'_barrel{int(random_barrel_k*100)}', barrel_img, barrel_points))

                # Pure Pincushion Distortion
                if random_pincushion_k != 0:
                    pincushion_img, pincushion_points = self._apply_single_augmentation(img, points_data, 0, random_pincushion_k)
                    images_to_save.append((f'_pincushion{int(random_pincushion_k*100)}', pincushion_img, pincushion_points))

                # Barrel Distortion + Rotation
                if random_barrel_k != 0 and random_rotation_angle != 0:
                    barrel_rot_img, barrel_rot_points = self._apply_single_augmentation(img, points_data, random_rotation_angle, random_barrel_k)
                    images_to_save.append((f'_barrel{int(random_barrel_k*100)}_rot{random_rotation_angle}', barrel_rot_img, barrel_rot_points))

                # Pincushion Distortion + Rotation
                if random_pincushion_k != 0 and random_rotation_angle != 0:
                    pincushion_rot_img, pincushion_rot_points = self._apply_single_augmentation(img, points_data, random_rotation_angle, random_pincushion_k)
                    images_to_save.append((f'_pincushion{int(random_pincushion_k*100)}_rot{random_rotation_angle}', pincushion_rot_img, pincushion_rot_points))

            for suffix, image_to_save, points_to_save in images_to_save:
                resized_img = cv2.resize(image_to_save, (640, 640))
                
                new_img_name = img_path.stem + suffix + img_path.suffix
                new_label_name = img_path.stem + suffix + ".txt"
                
                cv2.imwrite(str(target_images_dir / new_img_name), resized_img)
                
                with open(target_labels_dir / new_label_name, "w") as f:
                    h, w = image_to_save.shape[:2]
                    for p in points_to_save:
                        x_norm = max(0, min(1, p['x'] / w))
                        y_norm = max(0, min(1, p['y'] / h))
                        f.write(f"{p['type']} {x_norm:.6f} {y_norm:.6f} 0.01 0.01 {x_norm:.6f} {y_norm:.6f} {p['visibility']}\n")

        progress_dialog.close()
        QMessageBox.information(self, "Export Complete", f"Batch export completed to:\n{dataset_root}")

    def delete_selected(self):
        pts = [it for it in self.scene.selectedItems() if hasattr(it, "pid")]
        if pts:
            self.undo_stack.push(commands.DeletePointCommand(self.scene, pts))

    def change_type_selected(self, new_t: int):
        pts = [it for it in self.scene.selectedItems() if hasattr(it, "ptype")]
        for p in pts:
            if p.ptype != new_t:
                self.undo_stack.push(commands.ChangeTypeCommand(p, p.ptype, new_t))

    def change_visibility_selected(self, new_v: int):
        pts = [it for it in self.scene.selectedItems() if hasattr(it, "visibility")]
        for p in pts:
            if p.visibility != new_v:
                self.undo_stack.push(commands.ChangeVisibilityCommand(p, p.visibility, new_v))

    def toggle_visibility_selected(self):
        pts = [it for it in self.scene.selectedItems() if hasattr(it, "visibility")]
        if not pts:
            return
        # Toggle based on the first selected point's state
        current_vis = pts[0].visibility
        new_vis = constants.VISIBILITY_OCCLUDED if current_vis == constants.VISIBILITY_VISIBLE else constants.VISIBILITY_VISIBLE
        self.change_visibility_selected(new_vis)

    def set_default_ptype(self, t: int):
        self.default_ptype = t
        self.status.showMessage(f"Default point type set to {constants.TYPE_NAMES[t]}")

    def toggle_homography(self, checked: bool):
        self.homography_mode = checked
        if self.homo_overlay:
            self.scene.removeItem(self.homo_overlay)
            self.homo_overlay = None
        for item in self.homography_point_items:
            self.scene.removeItem(item)
        self.homography_point_items.clear()
        self.homography_src_points = []
        if checked:
            self.status.showMessage("Homography mode: click bottom-left, bottom-right, top-right, top-left")
        else:
            self.status.showMessage("Exited homography mode")

    def toggle_grid(self, checked: bool):
        self.viewer.grid_enabled = checked
        self.viewer.viewport().update()
        self.status.showMessage(f"Grid overlay: {'Enabled' if checked else 'Disabled'}")

    def on_homography_click(self, pos: QPointF):
        r = constants.POINT_RADIUS
        pen = QPen(QColor(0, 255, 255), 2)
        brush = QBrush(QColor(0, 255, 255, 100))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        ellipse = self.scene.addEllipse(pos.x() - r, pos.y() - r, 2 * r, 2 * r, pen, brush)
        self.homography_point_items.append(ellipse)
        self.homography_src_points.append(pos)
        count = len(self.homography_src_points)
        self.status.showMessage(f"Selected homography point {count}/4")
        if count == 4:
            self.finish_homography()

    def finish_homography(self):
        src_poly = QPolygonF([QPointF(x, y) for x, y in constants.HOMOGRAPHY_TEMPLATE])
        dst_poly = QPolygonF(self.homography_src_points)
        transform = QTransform()
        if not QTransform.quadToQuad(src_poly, dst_poly, transform):
            self.status.showMessage("Failed to compute homography")
            self.homography_mode = False
            self.homography_act.setChecked(False)
            return
        pen = QPen(QColor(255, 255, 0), 2)
        brush = QBrush()
        brush.setStyle(Qt.BrushStyle.NoBrush)
        self.homo_overlay = self.scene.addPolygon(dst_poly, pen, brush)
        for item in self.homography_point_items:
            self.scene.removeItem(item)
        self.homography_point_items.clear()
        self.undo_stack.beginMacro("Apply homography")
        for idx, (x, y) in enumerate(constants.TEMPLATE_POINTS):
            mapped = transform.map(QPointF(x, y))
            ptype = constants.TEMPLATE_TYPES[idx]
            self.undo_stack.push(commands.AddPointCommand(self.scene, mapped, ptype))
        self.undo_stack.endMacro()
        self.notify_modified()
        self.update_table()
        self.homography_mode = False
        self.homography_act.setChecked(False)
        self.status.showMessage("Homography applied: generated standard points")
