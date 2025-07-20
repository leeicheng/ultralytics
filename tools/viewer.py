from PyQt6.QtWidgets import QGraphicsView
from PyQt6.QtGui import QPainter, QTransform, QCursor, QPen
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt

import scene
import commands

class MagnifierView(QGraphicsView):
    """View subclass that draws a 1px red center reticle."""
    def __init__(self, scene_obj, parent=None):
        super().__init__(scene_obj, parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setInteractive(False)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        painter.resetTransform()
        w = self.viewport().width()
        h = self.viewport().height()
        pen = QPen(Qt.GlobalColor.red)
        pen.setWidth(1)
        painter.setPen(pen)
        # draw crosshair: vertical and horizontal lines
        painter.drawLine(w // 2, 0, w // 2, h)
        painter.drawLine(0, h // 2, w, h // 2)

class ImageViewer(QGraphicsView):
    """Graphics view for image display and point interaction."""
    _MAX_ZOOM_IN = 30
    _MAX_ZOOM_OUT = -15

    def __init__(self, scene_obj: scene.AnnotationScene):
        super().__init__(scene_obj)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._zoom_steps = 0
        # enable mouse tracking for magnifier overlay
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        # magnifier configuration
        self.magnifier_zoom = 2.0
        self.magnifier_size = 200
        # create magnifier view as overlay
        self.magnifier_view = MagnifierView(self.scene(), self.viewport())
        self.magnifier_view.setStyleSheet("border:2px solid black; background:rgba(255,255,255,200)")
        self.magnifier_view.setFixedSize(self.magnifier_size, self.magnifier_size)
        self.magnifier_view.show()

        self.crosshair_enabled = True  # New: enable crosshair by default
        self.grid_enabled = False      # New: disable grid by default
        self.mouse_pos_scene = None    # New: store mouse position in scene coordinates

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        
        if self.grid_enabled:
            self._draw_grid(painter)
        
        if self.crosshair_enabled and self.mouse_pos_scene:
            self._draw_crosshair(painter)
        
        painter.end()

    def _draw_crosshair(self, painter):
        if not self.mouse_pos_scene:
            return

        # Map scene coordinates to viewport coordinates
        viewport_pos = self.mapFromScene(self.mouse_pos_scene)
        
        # Set pen for crosshair
        pen = QPen(QColor(255, 0, 0, 150)) # Semi-transparent red
        pen.setWidth(1)
        painter.setPen(pen)

        # Draw horizontal line
        painter.drawLine(0, viewport_pos.y(), self.viewport().width(), viewport_pos.y())
        # Draw vertical line
        painter.drawLine(viewport_pos.x(), 0, viewport_pos.x(), self.viewport().height())

    def _draw_grid(self, painter):
        # Set pen for grid
        pen = QPen(QColor(150, 150, 150, 100)) # Semi-transparent gray
        pen.setWidth(1)
        painter.setPen(pen)

        # Get visible scene rect
        visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()

        # Calculate grid spacing based on current zoom level
        # Adjust this value to control grid density
        grid_spacing = 50 / self.transform().m11() # 50 pixels in viewport space

        # Draw vertical lines
        x_start = int(visible_rect.left() / grid_spacing) * grid_spacing
        x_end = visible_rect.right()
        for x in range(int(x_start), int(x_end) + 1, int(grid_spacing)):
            p1 = self.mapFromScene(x, visible_rect.top())
            p2 = self.mapFromScene(x, visible_rect.bottom())
            painter.drawLine(p1, p2)

        # Draw horizontal lines
        y_start = int(visible_rect.top() / grid_spacing) * grid_spacing
        y_end = visible_rect.bottom()
        for y in range(int(y_start), int(y_end) + 1, int(grid_spacing)):
            p1 = self.mapFromScene(visible_rect.left(), y)
            p2 = self.mapFromScene(visible_rect.right(), y)
            painter.drawLine(p1, p2)

    def wheelEvent(self, event):
        # default wheel zoom on main view
        factor = 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
        new_steps = self._zoom_steps + (1 if factor > 1 else -1)
        if self._MAX_ZOOM_OUT <= new_steps <= self._MAX_ZOOM_IN:
            self._zoom_steps = new_steps
            self.scale(factor, factor)
        # update magnifier overlay at cursor
        try:
            pt = event.position().toPoint()
        except AttributeError:
            pt = self.mapFromGlobal(QCursor.pos())
        self._update_magnifier(pt)

    def mousePressEvent(self, event):
        # update magnifier before handling click
        self._update_magnifier(event.pos())
        # handle homography mode clicks first
        mw = self.scene().parent().window()
        if (event.button() == Qt.MouseButton.LeftButton
                and getattr(mw, 'homography_mode', False)):
            scene_pos = self.mapToScene(event.pos())
            mw.on_homography_click(scene_pos)
            return
        # normal point addition
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            item = self.scene().itemAt(scene_pos, self.transform())
            if item is None or item == self.scene().image_item:
                stack = mw.undo_stack
                # use default point type from main window
                ptype = getattr(mw, 'default_ptype', 0)
                stack.push(commands.AddPointCommand(self.scene(), scene_pos, ptype))
                mw.notify_modified()
                return
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        # switch default add-point type with keys 1/2/3 when no selection
        if event.key() in (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3):
            mw = self.scene().parent().window()
            ptype = event.key() - Qt.Key.Key_1
            if not self.scene().selectedItems():
                mw.set_default_ptype(ptype)
            else:
                mw.change_type_selected(ptype)
            return
        if event.key() == Qt.Key.Key_R and not self.scene().selectedItems(): # R for Reset Zoom
            self.resetTransform()
            self._zoom_steps = 0
        elif event.key() == Qt.Key.Key_F and not self.scene().selectedItems(): # F for Fit in View
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_steps = 0
        else:
            super().keyPressEvent(event)
        # refresh magnifier position on key actions
        self._update_magnifier(self.mapFromGlobal(QCursor.pos()))
    
    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.mouse_pos_scene = self.mapToScene(event.pos())
        self.viewport().update() # Trigger repaint for crosshair and grid
        self._update_magnifier(event.pos())

    def _update_magnifier(self, viewport_pos):
        # map cursor to scene and update magnifier view
        scene_pos = self.mapToScene(viewport_pos)
        # apply magnification transform
        t = QTransform()
        t.scale(self.magnifier_zoom, self.magnifier_zoom)
        self.magnifier_view.setTransform(t)
        # center magnifier on cursor
        self.magnifier_view.centerOn(scene_pos)
        # position magnifier at top-right corner of viewport
        margin = 10
        x = self.viewport().width() - self.magnifier_size - margin
        y = margin
        self.magnifier_view.move(x, y)
    
    def increase_magnifier_zoom(self):
        """Increase magnifier zoom via UI or shortcut."""
        self.magnifier_zoom = min(self.magnifier_zoom * 1.1, 30.0)
        self._update_magnifier(self.mapFromGlobal(QCursor.pos()))

    def decrease_magnifier_zoom(self):
        """Decrease magnifier zoom via UI or shortcut."""
        self.magnifier_zoom = max(self.magnifier_zoom * 0.9, 1.0)
        self._update_magnifier(self.mapFromGlobal(QCursor.pos()))