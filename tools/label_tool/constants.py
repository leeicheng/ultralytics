from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

POINT_RADIUS = 5
SQUARE_SIDE = 8
TRIANGLE_SIDE = 8
TYPE_NAMES = {0: "T-junction", 1: "Cross", 2: "L-corner"}
TYPE_COLORS = {0: QColor(Qt.GlobalColor.red), 1: QColor(Qt.GlobalColor.blue), 2: QColor(Qt.GlobalColor.green)}
PROJECT_EXT = ".json"
AUTO_SAVE_MS = 1500

# Visibility states, following YOLO Pose format
VISIBILITY_VISIBLE = 2
VISIBILITY_OCCLUDED = 1
VISIBILITY_NAMES = {
    VISIBILITY_VISIBLE: "Visible",
    VISIBILITY_OCCLUDED: "Occluded"
}

# Template for homography: bottom-left, bottom-right, top-right, top-left (in meters)
HOMOGRAPHY_TEMPLATE = [
    (0.00, 0.00),   # bottom-left
    (6.10, 0.00),   # bottom-right
    (6.10, 13.40),  # top-right
    (0.00, 13.40),  # top-left
]

# Standard badminton court template points (30 points) in meters
TEMPLATE_POINTS = [
    (0.00, 13.40),  (0.46, 13.40),  (3.05, 13.40),  (5.64, 13.40),  (6.10, 13.40),
    (0.00, 12.64),  (0.46, 12.64),  (3.05, 12.64),  (5.64, 12.64),  (6.10, 12.64),
    (0.00,  8.68),  (0.46,  8.68),  (3.05,  8.68),  (5.64,  8.68),  (6.10,  8.68),
    (0.00,  4.72),  (0.46,  4.72),  (3.05,  4.72),  (5.64,  4.72),  (6.10,  4.72),
    (0.00,  0.76),  (0.46,  0.76),  (3.05,  0.76),  (5.64,  0.76),  (6.10,  0.76),
    (0.00,  0.00),  (0.46,  0.00),  (3.05,  0.00),  (5.64,  0.00),  (6.10,  0.00),
]
# Types for each template point: pattern [Corner, T, Cross, T, Corner] per row
TEMPLATE_TYPES = [
    # Far baseline (corners and T/C/R/C/T)
    2, 0, 0, 0, 2,
    # Far doubles long service (T/C/C/C/T)
    0, 1, 1, 1, 0,
    # Far short service (T/C/C/C/T)
    0, 1, 0, 1, 0,
    # Near short service (T/C/C/C/T)
    0, 1, 0, 1, 0,
    # Near doubles long service (T/C/C/C/T)
    0, 1, 1, 1, 0,
    # Near baseline (corners and T/C/R/C/T)
    2, 0, 0, 0, 2,
]