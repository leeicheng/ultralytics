import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QSlider, QRadioButton, QButtonGroup,
    QGridLayout, QGroupBox
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QTimer, QSize


class EdgeDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.cv_original_image = None
        self.video_capture = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.next_frame)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("PyQt6 Sobel & Canny 混合比對工具")
        self.setGeometry(100, 100, 1400, 800)

        # -----------------------------------------------
        # 主要版面配置
        # -----------------------------------------------
        main_layout = QVBoxLayout()

        # 1. 頂部控制列
        top_layout = QHBoxLayout()
        self.load_button = QPushButton("載入圖片 / 影像 (Load Image / Video)")
        self.load_button.setFont(QFont("Arial", 12))
        self.load_button.clicked.connect(self.load_file)
        top_layout.addWidget(self.load_button)
        main_layout.addLayout(top_layout)

        # 2. 影像顯示區
        display_layout = QHBoxLayout()
        self.original_label = self.create_image_label("原始影像 (Original)")
        self.processed_label = self.create_image_label("處理後影像 (Processed)")
        display_layout.addWidget(self.original_label)
        display_layout.addWidget(self.processed_label)
        main_layout.addLayout(display_layout, 1)  # 讓影像區域可伸縮

        # 3. 底部控制區
        controls_layout = QHBoxLayout()

        # 3.1 模式選擇
        mode_group = QGroupBox("模式選擇 (Mode Selection)")
        mode_vbox = QVBoxLayout()
        self.radio_sobel = QRadioButton("Sobel")
        self.radio_canny = QRadioButton("Canny")
        self.radio_mixed = QRadioButton("混合 (Mixed)")
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.radio_sobel)
        self.mode_button_group.addButton(self.radio_canny)
        self.mode_button_group.addButton(self.radio_mixed)

        self.radio_sobel.toggled.connect(self.on_mode_change)
        self.radio_canny.toggled.connect(self.on_mode_change)
        self.radio_mixed.toggled.connect(self.on_mode_change)

        mode_vbox.addWidget(self.radio_sobel)
        mode_vbox.addWidget(self.radio_canny)
        mode_vbox.addWidget(self.radio_mixed)
        mode_group.setLayout(mode_vbox)
        controls_layout.addWidget(mode_group)

        # 3.2 參數滑桿
        params_group = QGroupBox("參數調整 (Parameters)")
        params_grid = QGridLayout()

        # Canny
        self.canny1_label, self.canny1_slider, self.canny1_value_label = self.create_slider("Canny 閾值 1", 0, 255, 50)
        self.canny2_label, self.canny2_slider, self.canny2_value_label = self.create_slider("Canny 閾值 2", 0, 255, 150)

        # Sobel
        self.sobel_k_label, self.sobel_k_slider, self.sobel_k_value_label = self.create_slider("Sobel 核心 (KSize)", 1,
                                                                                               15,
                                                                                               1)  # 1-15 -> 3, 5, ... 31

        # Mixed
        self.mix_label, self.mix_slider, self.mix_value_label = self.create_slider("混合比例 (Canny %)", 0, 100, 50)

        params_grid.addWidget(self.canny1_label, 0, 0)
        params_grid.addWidget(self.canny1_slider, 0, 1)
        params_grid.addWidget(self.canny1_value_label, 0, 2)

        params_grid.addWidget(self.canny2_label, 1, 0)
        params_grid.addWidget(self.canny2_slider, 1, 1)
        params_grid.addWidget(self.canny2_value_label, 1, 2)

        params_grid.addWidget(self.sobel_k_label, 2, 0)
        params_grid.addWidget(self.sobel_k_slider, 2, 1)
        params_grid.addWidget(self.sobel_k_value_label, 2, 2)

        params_grid.addWidget(self.mix_label, 3, 0)
        params_grid.addWidget(self.mix_slider, 3, 1)
        params_grid.addWidget(self.mix_value_label, 3, 2)

        params_group.setLayout(params_grid)
        controls_layout.addWidget(params_group, 1)  # 讓滑桿區域可伸縮

        main_layout.addLayout(controls_layout)
        self.setLayout(main_layout)

        # 初始狀態
        self.radio_canny.setChecked(True)
        self.update_control_visibility()
        self.show()

    def create_image_label(self, text):
        """輔助函式：建立一個用於顯示影像的 QLabel"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFont(QFont("Arial", 14))
        label.setMinimumSize(640, 480)
        label.setStyleSheet("border: 1px solid #AAA; background-color: #EEE;")
        return label

    def create_slider(self, text, min_val, max_val, default_val):
        """輔助函式：建立一組 Labe + Slider + ValueLabel"""
        label = QLabel(text)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)

        value_label = QLabel(str(default_val))
        value_label.setMinimumWidth(30)

        slider.valueChanged.connect(lambda: self.update_display())
        slider.valueChanged.connect(lambda val: value_label.setText(str(val)))

        return label, slider, value_label

    def on_mode_change(self):
        """當 RadioButton 改變時，更新顯示並調整 UI"""
        self.update_control_visibility()
        self.update_display()

    def update_control_visibility(self):
        """根據選擇的模式，顯示/隱藏相關的滑桿"""
        is_sobel = self.radio_sobel.isChecked()
        is_canny = self.radio_canny.isChecked()
        is_mixed = self.radio_mixed.isChecked()

        # Canny 滑桿
        self.canny1_label.setVisible(is_canny or is_mixed)
        self.canny1_slider.setVisible(is_canny or is_mixed)
        self.canny1_value_label.setVisible(is_canny or is_mixed)
        self.canny2_label.setVisible(is_canny or is_mixed)
        self.canny2_slider.setVisible(is_canny or is_mixed)
        self.canny2_value_label.setVisible(is_canny or is_mixed)

        # Sobel 滑桿
        self.sobel_k_label.setVisible(is_sobel or is_mixed)
        self.sobel_k_slider.setVisible(is_sobel or is_mixed)
        self.sobel_k_value_label.setVisible(is_sobel or is_mixed)

        # Mix 滑桿
        self.mix_label.setVisible(is_mixed)
        self.mix_slider.setVisible(is_mixed)
        self.mix_value_label.setVisible(is_mixed)

    def load_file(self):
        """開啟檔案對話框並載入圖片或影片"""
        # 停止計時器
        self.video_timer.stop()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇檔案", "",
            "影像檔案 (*.png *.jpg *.jpeg *.bmp);;影片檔案 (*.mp4 *.avi *.mov)"
        )

        if not file_path:
            return

        # 檢查檔案類型
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            self.cv_original_image = cv2.imread(file_path)
            if self.cv_original_image is not None:
                self.display_image(self.cv_original_image, self.original_label)
                self.update_display()

        elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            self.video_capture = cv2.VideoCapture(file_path)
            if self.video_capture.isOpened():
                self.video_timer.start(30)  # 約 33 FPS
            else:
                self.original_label.setText("無法開啟影片檔案")

    def next_frame(self):
        """從影片讀取下一幀"""
        ret, frame = self.video_capture.read()
        if ret:
            self.cv_original_image = frame
            self.display_image(self.cv_original_image, self.original_label)
            self.update_display()
        else:
            # 影片結束
            self.video_timer.stop()
            self.video_capture.release()
            self.video_capture = None

    def update_display(self):
        """核心函式：根據滑桿和模式，處理並顯示影像"""
        if self.cv_original_image is None:
            return

        # 讀取滑桿數值
        canny_t1 = self.canny1_slider.value()
        canny_t2 = self.canny2_slider.value()
        # KSize 必須是奇數
        sobel_k = self.sobel_k_slider.value() * 2 + 1
        self.sobel_k_value_label.setText(str(sobel_k))  # 更新 KSize 顯示

        mix_alpha = self.mix_slider.value() / 100.0  # 0.0 to 1.0

        # 預處理
        gray = cv2.cvtColor(self.cv_original_image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Canny 建議先降噪

        # -----------------
        # 1. 計算 Canny
        # -----------------
        canny_edges = cv2.Canny(gray_blur, canny_t1, canny_t2)
        # 轉換為 3 通道 BGR 以便混合
        canny_display = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)

        # -----------------
        # 2. 計算 Sobel
        # -----------------
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_k)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_k)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
        # 轉換為 3 通道 BGR 以便混合
        sobel_display = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)

        # -----------------
        # 3. 根據模式選擇/混合
        # -----------------
        if self.radio_sobel.isChecked():
            final_image = sobel_display
        elif self.radio_canny.isChecked():
            final_image = canny_display
        elif self.radio_mixed.isChecked():
            # cv2.addWeighted(src1, alpha, src2, beta, gamma)
            # 這裡 alpha 是 Canny 的比例
            final_image = cv2.addWeighted(sobel_display, 1.0 - mix_alpha, canny_display, mix_alpha, 0)
        else:
            # 預設
            final_image = canny_display

        # 顯示處理後的影像
        self.display_image(final_image, self.processed_label)

    def display_image(self, cv_img, label: QLabel):
        """將 OpenCV 影像 (NumPy) 轉換並顯示在 QLabel 上"""
        try:
            qt_img = self.convert_cv_to_qt(cv_img)
            pixmap = QPixmap.fromImage(qt_img)
            # 縮放 pixmap 以符合 label 大小，並保持長寬比
            scaled_pixmap = pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error displaying image: {e}")
            label.setText("顯示錯誤")

    def convert_cv_to_qt(self, cv_img):
        """將 OpenCV 影像轉換為 QImage"""
        if cv_img is None:
            return QImage()

        if cv_img.ndim == 2:
            # 灰階影像
            h, w = cv_img.shape
            bytes_per_line = w
            return QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)

        elif cv_img.ndim == 3:
            # 彩色影像 (BGR to RGB)
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        return QImage()

    def closeEvent(self, event):
        """關閉視窗時釋放資源"""
        self.video_timer.stop()
        if self.video_capture:
            self.video_capture.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EdgeDetectorApp()
    sys.exit(app.exec())