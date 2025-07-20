from __future__ import annotations

from PyQt6.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import QRectF, QPointF
from typing import List, Dict
from PyQt6.QtGui import QPixmap

import items

class AnnotationScene(QGraphicsScene):
    """Handles image display and point items."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_item: QGraphicsPixmapItem | None = None
        self.points: List[items.PointItem] = []
        self._next_pid = 1

    def load_image(self, pix: QPixmap):
        self.clear()
        self.image_item = self.addPixmap(pix)
        self.setSceneRect(QRectF(pix.rect()))
        self.points = []
        self._next_pid = 1

    def _create_point(self, pos: QPointF, ptype: int = 0) -> items.PointItem:
        p = items.PointItem(self._next_pid, pos, ptype)
        self._next_pid += 1
        self.points.append(p)
        self.addItem(p)
        return p

    def _delete_point(self, point: items.PointItem):
        if point in self.points:
            self.removeItem(point)
            self.points.remove(point)

    def _reinsert_point(self, point: items.PointItem):
        self.points.append(point)
        self.addItem(point)

    def to_dict(self) -> List[Dict]:
        return [
            {"id": p.pid, "x": p.pos().x(), "y": p.pos().y(), "type": p.ptype}
            for p in self.points
        ]

    def from_dict(self, data: List[Dict]):
        for d in data:
            p = self._create_point(QPointF(d["x"], d["y"]))
            p.pid = d["id"]
            p.ptype = d["type"]
            p.update_style()
        if self.points:
            self._next_pid = max(p.pid for p in self.points) + 1