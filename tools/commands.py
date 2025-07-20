from __future__ import annotations

from PyQt6.QtGui import QUndoCommand
from PyQt6.QtCore import QPointF
from typing import List

import scene
import items

class AddPointCommand(QUndoCommand):
    """Undoable command to add a point with specified type."""
    def __init__(self, scene_obj: scene.AnnotationScene, pos: QPointF, ptype: int = 0):
        super().__init__("Add point")
        self.scene = scene_obj
        self.pos = pos
        self.ptype = ptype
        self.point: items.PointItem | None = None

    def redo(self):
        if self.point is None:
            self.point = self.scene._create_point(self.pos, self.ptype)
        else:
            self.scene._reinsert_point(self.point)
        # auto-select the new point if not in homography mode
        mw = self.scene.parent()
        if not getattr(mw, 'homography_mode', False):
            # clear existing selections
            for p in self.scene.points:
                p.setSelected(False)
            # select this point
            if self.point:
                self.point.setSelected(True)

    def undo(self):
        self.scene._delete_point(self.point)

class DeletePointCommand(QUndoCommand):
    """Undoable command to delete points."""
    def __init__(self, scene_obj: scene.AnnotationScene, points: List[items.PointItem]):
        super().__init__("Delete point(s)")
        self.scene = scene_obj
        self.points = points

    def redo(self):
        for p in self.points:
            self.scene._delete_point(p)

    def undo(self):
        for p in self.points:
            self.scene._reinsert_point(p)

class MovePointCommand(QUndoCommand):
    """Undoable command to move a point."""
    def __init__(self, point: items.PointItem, old_pos: QPointF, new_pos: QPointF):
        super().__init__("Move point")
        self.point = point
        self.old = old_pos
        self.new = new_pos

    def redo(self):
        self.point.setPos(self.new)

    def undo(self):
        self.point.setPos(self.old)

class ChangeTypeCommand(QUndoCommand):
    """Undoable command to change point type."""
    def __init__(self, point: items.PointItem, old_t: int, new_t: int):
        super().__init__("Change type")
        self.point = point
        self.old = old_t
        self.new = new_t

    def redo(self):
        self.point.ptype = self.new
        self.point.update_style()

    def undo(self):
        self.point.ptype = self.old
        self.point.update_style()