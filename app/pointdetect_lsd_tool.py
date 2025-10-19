import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QCheckBox,
    QListWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QDockWidget,
    QGroupBox, QSpinBox, QListWidgetItem, QMessageBox, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QRectF
from ultralytics import YOLO
import math

class ZoomableView(QGraphicsView):
    """ A custom QGraphicsView that supports zooming and panning. """
    def __init__(self, scene):
        super().__init__(scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        old_pos = self.mapToScene(event.position().toPoint())
        if event.angleDelta().y() > 0:
            scale_factor = zoom_in_factor
        else:
            scale_factor = zoom_out_factor
        self.scale(scale_factor, scale_factor)
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PointDetect+LSD Tool")
        self.setGeometry(100, 100, 1200, 800)

        # --- State Variables ---
        self.model, self.model_path = None, ""
        self.media_path, self.frames = "", []
        self.current_frame_index = -1
        self.processed_image, self.base_image_for_highlight, self.pixmap_item = None, None, None
        self.detected_points_data, self.detected_lines, self.corner_points = [], [], []
        self.point_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()

    def _setup_ui(self):
        self.graphics_scene = QGraphicsScene()
        self.graphics_view = ZoomableView(self.graphics_scene)
        self.setCentralWidget(self.graphics_view)

        left_dock = QDockWidget("影像/影片", self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, left_dock)
        media_widget = QWidget()
        media_layout = QVBoxLayout(media_widget)
        self.load_media_button = QPushButton("讀取影像/影片")
        self.frame_list_widget = QListWidget()
        self.prev_frame_button, self.next_frame_button = QPushButton("上一張"), QPushButton("下一張")
        media_layout.addWidget(self.load_media_button)
        media_layout.addWidget(QLabel("幀列表:"))
        media_layout.addWidget(self.frame_list_widget)
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_frame_button)
        nav_layout.addWidget(self.next_frame_button)
        media_layout.addLayout(nav_layout)
        left_dock.setWidget(media_widget)

        right_dock = QDockWidget("操作與結果", self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, right_dock)
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        model_group = QGroupBox("模型選取")
        model_layout = QVBoxLayout(model_group)
        self.load_model_button = QPushButton("選擇 PointDetect 模型 (.pt)")
        self.model_path_label = QLabel("模型未載入")
        self.model_path_label.setWordWrap(True)
        model_layout.addWidget(self.load_model_button)
        model_layout.addWidget(self.model_path_label)

        ops_group = QGroupBox("操作")
        ops_layout = QVBoxLayout(ops_group)
        self.cb_pointdetect = QCheckBox("PointDetect (偵測場地交點)")
        self.cb_lsd = QCheckBox("Line Segment Detector (偵測線段)")
        self.cb_filter = QCheckBox("過濾 (顯示與交點最近的線段)")
        self.cb_outlier_detection = QCheckBox("分離異常線段 (依角度)")
        self.cb_find_nearest_endpoints = QCheckBox("尋找最近線段端點")

        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("搜尋半徑(px):"))
        self.radius_spinbox = QSpinBox()
        self.radius_spinbox.setRange(1, 500); self.radius_spinbox.setValue(50)
        params_layout.addWidget(self.radius_spinbox)
        params_layout.addWidget(QLabel("數量(x):"))
        self.nearest_endpoints_count_spinbox = QSpinBox()
        self.nearest_endpoints_count_spinbox.setRange(1, 20); self.nearest_endpoints_count_spinbox.setValue(3)
        params_layout.addWidget(self.nearest_endpoints_count_spinbox)

        cluster_layout = QHBoxLayout()
        cluster_layout.addWidget(QLabel("合併距離(px):"))
        self.merge_distance_spinbox = QSpinBox()
        self.merge_distance_spinbox.setRange(0, 100); self.merge_distance_spinbox.setValue(10)
        cluster_layout.addWidget(self.merge_distance_spinbox)

        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("交點大小:"))
        self.point_size_spinbox = QSpinBox()
        self.point_size_spinbox.setRange(1, 30); self.point_size_spinbox.setValue(5)
        style_layout.addWidget(self.point_size_spinbox)
        style_layout.addWidget(QLabel("線段粗細:"))
        self.line_thickness_spinbox = QSpinBox()
        self.line_thickness_spinbox.setRange(1, 20); self.line_thickness_spinbox.setValue(2)
        style_layout.addWidget(self.line_thickness_spinbox)

        self.execute_button = QPushButton("執行")
        ops_layout.addWidget(self.cb_pointdetect)
        ops_layout.addWidget(self.cb_lsd)
        ops_layout.addWidget(self.cb_filter)
        ops_layout.addWidget(self.cb_outlier_detection)
        ops_layout.addWidget(self.cb_find_nearest_endpoints)
        ops_layout.addLayout(params_layout)
        ops_layout.addLayout(cluster_layout)
        ops_layout.addLayout(style_layout)
        ops_layout.addWidget(self.execute_button)

        results_group = QGroupBox("結果")
        results_layout = QVBoxLayout(results_group)
        results_layout.addWidget(QLabel("交點列表:"))
        self.points_list_widget = QListWidget()
        results_layout.addWidget(self.points_list_widget)
        results_layout.addWidget(QLabel("線段列表:"))
        self.lines_list_widget = QListWidget()
        results_layout.addWidget(self.lines_list_widget)
        results_layout.addWidget(QLabel("角點列表:"))
        self.corner_points_list_widget = QListWidget()
        results_layout.addWidget(self.corner_points_list_widget)

        control_layout.addWidget(model_group)
        control_layout.addWidget(ops_group)
        control_layout.addWidget(results_group)
        control_layout.addStretch()
        right_dock.setWidget(control_widget)

    def _connect_signals(self):
        self.load_model_button.clicked.connect(self.load_model)
        self.load_media_button.clicked.connect(self.load_media)
        self.prev_frame_button.clicked.connect(self.show_prev_frame)
        self.next_frame_button.clicked.connect(self.show_next_frame)
        self.frame_list_widget.currentItemChanged.connect(self.on_frame_selected)
        self.execute_button.clicked.connect(self.run_processing)
        self.points_list_widget.itemSelectionChanged.connect(self.highlight_selection)
        self.lines_list_widget.itemSelectionChanged.connect(self.highlight_selection)
        self.corner_points_list_widget.itemSelectionChanged.connect(self.highlight_selection)

    def _update_ui_state(self):
        media_loaded = self.current_frame_index != -1
        model_loaded = self.model is not None
        self.prev_frame_button.setEnabled(media_loaded and self.current_frame_index > 0)
        self.next_frame_button.setEnabled(media_loaded and self.current_frame_index < len(self.frames) - 1)
        self.execute_button.setEnabled(media_loaded and model_loaded)
        base_enabled = self.cb_pointdetect.isChecked() and self.cb_lsd.isChecked()
        self.cb_filter.setEnabled(base_enabled)
        self.cb_outlier_detection.setEnabled(self.cb_lsd.isChecked())
        self.cb_find_nearest_endpoints.setEnabled(base_enabled)
        for cb in [self.cb_filter, self.cb_outlier_detection, self.cb_find_nearest_endpoints]:
            if not cb.isEnabled(): cb.setChecked(False)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇 PointDetect 模型", "", "PyTorch Models (*.pt)")
        if path:
            try:
                self.model = YOLO(path)
                self.model_path = path
                self.model_path_label.setText(f"已載入: ...{self.model_path[-40:]}")
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"無法載入模型: {e}")
                self.model, self.model_path = None, ""
                self.model_path_label.setText("模型載入失敗")
        self._update_ui_state()

    def load_media(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇影像或影片", "", "Media Files (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi)")
        if path:
            self.media_path, self.frames = path, []
            cap = cv2.VideoCapture(path)
            if not cap.isOpened(): QMessageBox.critical(self, "錯誤", "無法開啟影像或影片檔案"); return
            while True:
                ret, frame = cap.read()
                if not ret: break
                self.frames.append(frame)
            cap.release()
            self.frame_list_widget.clear()
            for i in range(len(self.frames)): self.frame_list_widget.addItem(f"frame{i+1:04d}")
            if self.frames:
                self.current_frame_index = 0
                self.frame_list_widget.setCurrentRow(0)
        self._update_ui_state()

    def display_frame(self, index, image_to_display=None):
        if image_to_display is None:
            if not (0 <= index < len(self.frames)): return
            image_to_display = self.frames[index]
        self.processed_image = image_to_display
        pixmap = QPixmap.fromImage(self.convert_cv_to_qt(image_to_display))
        self.graphics_scene.clear()
        self.pixmap_item = self.graphics_scene.addPixmap(pixmap)
        self.graphics_scene.setSceneRect(QRectF(pixmap.rect()))
        if self.current_frame_index == index: self.graphics_view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def on_frame_selected(self, current_item, previous_item):
        if current_item:
            self.current_frame_index = self.frame_list_widget.row(current_item)
            self.display_frame(self.current_frame_index)
            self.clear_results()
            self._update_ui_state()

    def show_prev_frame(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.frame_list_widget.setCurrentRow(self.current_frame_index)
    
    def show_next_frame(self):
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            self.frame_list_widget.setCurrentRow(self.current_frame_index)

    def clear_results(self):
        self.points_list_widget.clear(); self.lines_list_widget.clear(); self.corner_points_list_widget.clear()
        self.detected_points_data, self.detected_lines, self.corner_points = [], [], []
        self.base_image_for_highlight = None

    def run_processing(self):
        if self.current_frame_index == -1 or self.model is None: return
        self.clear_results()
        image = self.frames[self.current_frame_index].copy()
        
        if self.cb_pointdetect.isChecked():
            results = self.model(image, verbose=False)[0]
            if results.keypoints and hasattr(results, 'point_cls') and results.keypoints.xy.numel() > 0:
                points, classes, names = results.keypoints.xy, results.point_cls, results.names
                if len(points.shape) == 3: points = points.squeeze(1) if points.shape[1] == 1 else points[0]
                for i, pt in enumerate(points):
                    self.detected_points_data.append({"coords": pt.cpu().numpy().astype(int), "class_name": names.get(int(classes[i])), "class_id": int(classes[i])})
        
        if self.cb_lsd.isChecked():
            lsd = cv2.createLineSegmentDetector(0)
            lines = lsd.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))[0]
            if lines is not None: self.detected_lines = [line[0] for line in lines]

        lines_to_draw = self.detected_lines
        if self.cb_filter.isChecked() and self.detected_points_data and self.detected_lines:
            radius = self.radius_spinbox.value()
            lines_to_draw = [line for line in self.detected_lines if any(self.point_to_line_segment_dist(p['coords'], line[:2], line[2:]) < radius for p in self.detected_points_data)]

        normal_lines, abnormal_lines = lines_to_draw, []
        if self.cb_outlier_detection.isChecked() and self.detected_lines:
            all_angles = [(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi) % 180 for x1, y1, x2, y2 in self.detected_lines]
            if all_angles:
                hist, bin_edges = np.histogram(all_angles, bins=18, range=(0, 180))
                threshold = max(np.max(hist) * 0.25, 2)
                dominant_bins = np.where(hist >= threshold)[0]
                normal_lines, abnormal_lines = [], []
                for line in lines_to_draw:
                    angle = (np.arctan2(line[3] - line[1], line[2] - line[0]) * 180 / np.pi) % 180
                    is_normal = any(bin_edges[b] <= angle < bin_edges[b+1] for b in dominant_bins)
                    (normal_lines if is_normal else abnormal_lines).append(line)

        if self.cb_find_nearest_endpoints.isChecked() and self.detected_points_data and self.detected_lines:
            radius, num_to_find = self.radius_spinbox.value(), self.nearest_endpoints_count_spinbox.value()
            all_endpoints = {tuple(line[:2]) for line in self.detected_lines} | {tuple(line[2:]) for line in self.detected_lines}
            found_endpoints = set()
            for p_data in self.detected_points_data:
                p_coords = p_data['coords']
                endpoints_in_radius = [(ep, np.linalg.norm(p_coords - np.array(ep))) for ep in all_endpoints if np.linalg.norm(p_coords - np.array(ep)) <= radius]
                endpoints_in_radius.sort(key=lambda item: item[1])
                for ep, dist in endpoints_in_radius[:num_to_find]:
                    found_endpoints.add(ep)

            merge_dist = self.merge_distance_spinbox.value()
            if found_endpoints and merge_dist > 0:
                self.corner_points = self._merge_close_points(list(found_endpoints), merge_dist)
            else:
                self.corner_points = sorted(list(found_endpoints))

        drawn_image = self.frames[self.current_frame_index].copy()
        point_size, line_thickness = self.point_size_spinbox.value(), self.line_thickness_spinbox.value()

        for line in normal_lines: cv2.line(drawn_image, tuple(map(int, line[:2])), tuple(map(int, line[2:])), (0, 255, 0), line_thickness)
        for line in abnormal_lines: cv2.line(drawn_image, tuple(map(int, line[:2])), tuple(map(int, line[2:])), (255, 0, 255), line_thickness)
        for ep in self.corner_points: cv2.circle(drawn_image, tuple(map(int, ep)), point_size, (0, 255, 255), -1)

        if self.cb_pointdetect.isChecked():
            for p_data in self.detected_points_data:
                pt, c_id = p_data["coords"], p_data["class_id"]
                color = self.point_colors[c_id % len(self.point_colors)]
                cv2.circle(drawn_image, tuple(pt), point_size, color, -1)
                cv2.circle(drawn_image, tuple(pt), point_size + 1, (255, 255, 255), 1)

        self.base_image_for_highlight = drawn_image.copy()
        self.display_frame(self.current_frame_index, drawn_image)
        self._update_results_lists(lines_to_draw)

    def _merge_close_points(self, points, merge_distance):
        if not points or merge_distance <= 0:
            return sorted(points)
        points_np = [np.array(p) for p in points]
        n = len(points_np)
        visited = [False] * n
        merged_points = []
        for i in range(n):
            if visited[i]: continue
            cluster_indices = []
            queue = [i]
            visited[i] = True
            head = 0
            while head < len(queue):
                current_idx = queue[head]; head += 1
                cluster_indices.append(current_idx)
                for j in range(i + 1, n):
                    if not visited[j] and np.linalg.norm(points_np[current_idx] - points_np[j]) < merge_distance:
                        visited[j] = True
                        queue.append(j)
            cluster_points = [points_np[k] for k in cluster_indices]
            centroid = np.mean(cluster_points, axis=0)
            merged_points.append(tuple(centroid.astype(int)))
        return sorted(merged_points)

    def _update_results_lists(self, lines_to_display):
        self.points_list_widget.clear()
        for i, p_data in enumerate(self.detected_points_data):
            self.points_list_widget.addItem(f"交點 {i+1}: ({p_data['coords'][0]}, {p_data['coords'][1]}) - {p_data['class_name']}")
        self.lines_list_widget.clear()
        for i, line in enumerate(lines_to_display):
            self.lines_list_widget.addItem(f"線段 {i+1}: ({int(line[0])},{int(line[1])}) -> ({int(line[2])},{int(line[3])})")
        self.corner_points_list_widget.clear()
        for i, point in enumerate(self.corner_points):
            self.corner_points_list_widget.addItem(f"角點 {i+1}: ({point[0]}, {point[1]})")

    def highlight_selection(self):
        if self.base_image_for_highlight is None: return
        img = self.base_image_for_highlight.copy()
        pt_size, ln_thick = self.point_size_spinbox.value(), self.line_thickness_spinbox.value()
        
        for item in self.points_list_widget.selectedItems():
            idx = self.points_list_widget.row(item)
            if 0 <= idx < len(self.detected_points_data):
                pt = self.detected_points_data[idx]["coords"]
                cv2.circle(img, tuple(pt), pt_size + 4, (0, 255, 255), 3)
        
        for item in self.corner_points_list_widget.selectedItems():
            idx = self.corner_points_list_widget.row(item)
            if 0 <= idx < len(self.corner_points):
                pt = self.corner_points[idx]
                cv2.circle(img, tuple(map(int, pt)), pt_size + 4, (255, 165, 0), 3)

        if not self.cb_outlier_detection.isChecked():
            for item in self.lines_list_widget.selectedItems():
                try:
                    idx = self.lines_list_widget.row(item)
                    if 0 <= idx < len(self.detected_lines):
                        line = self.detected_lines[idx]
                        cv2.line(img, tuple(map(int, line[:2])), tuple(map(int, line[2:])), (255, 0, 255), ln_thick + 1)
                except (IndexError, ValueError): pass
        self.display_frame(self.current_frame_index, img)

    @staticmethod
    def point_to_line_segment_dist(p, a, b):
        p, a, b = np.array(p), np.array(a), np.array(b)
        norm_b_a = np.linalg.norm(b - a)
        if norm_b_a == 0: return np.linalg.norm(p-a)
        d = np.divide(b - a, norm_b_a)
        s, t = np.dot(a - p, d), np.dot(p - b, d)
        h = np.maximum.reduce([s, t, 0])
        return np.hypot(h, np.abs(np.cross(p - a, d)))

    @staticmethod
    def convert_cv_to_qt(cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)

    def resizeEvent(self, event):
        if self.pixmap_item: self.graphics_view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())