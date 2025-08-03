from PyQt6.QtWidgets import QGraphicsItem, QMenu
from PyQt6.QtCore import QPointF, Qt, QRectF
from PyQt6.QtGui import QCursor, QColor, QPen, QBrush, QPolygonF, QAction, QActionGroup

import constants

class PointItem(QGraphicsItem):
    """Interactive point item."""
    def __init__(self, pid: int, pos: QPointF, ptype: int = 0, visibility: int = constants.VISIBILITY_VISIBLE):
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        self.pid = pid
        self.ptype = ptype
        self.visibility = visibility
        self.setPos(pos)
        self._drag_start = None
        self.update_style()

    def boundingRect(self) -> QRectF:
        # Return the bounding rectangle for the largest possible shape (triangle)
        side = constants.TRIANGLE_SIDE
        return QRectF(-side / 2, -side / 2, side, side)

    def update_style(self):
        # border: solid type color; fill: semi-transparent
        base_color = QColor(constants.TYPE_COLORS[self.ptype])
        fill_color = QColor(base_color)
        
        if self.visibility == constants.VISIBILITY_VISIBLE:
            fill_color.setAlpha(100)
            self._pen = QPen(base_color)
            self._pen.setStyle(Qt.PenStyle.SolidLine)
        else: # Occluded
            fill_color.setAlpha(50)
            self._pen = QPen(base_color)
            self._pen.setStyle(Qt.PenStyle.DashLine)

        self._brush = QBrush(fill_color)
        self._pen.setWidth(1)
        self.update() # Request a repaint

    def paint(self, painter, option, widget):
        painter.setPen(self._pen)
        painter.setBrush(self._brush)

        if self.ptype == 0:  # T-junction: Red Circle
            radius = constants.POINT_RADIUS
            painter.drawEllipse(QPointF(0, 0), radius, radius)
        elif self.ptype == 1:  # Cross: Blue Square
            side = constants.SQUARE_SIDE
            painter.drawRect(-side / 2, -side / 2, side, side)
        elif self.ptype == 2:  # L-corner: Green Triangle
            side = constants.TRIANGLE_SIDE
            # Equilateral triangle pointing up
            points = QPolygonF([
                QPointF(0, -side / 2),
                QPointF(-side / 2 * (3**0.5 / 2), side / 2 * (1 / 2)),
                QPointF(side / 2 * (3**0.5 / 2), side / 2 * (1 / 2))
            ])
            painter.drawPolygon(points)

        # draw 1px red center dot
        painter.setPen(QPen(Qt.GlobalColor.red))
        painter.drawPoint(0, 0)

        if self.isSelected():
            # Draw yellow outline for selected state
            selection_pen = QPen(Qt.GlobalColor.yellow)
            selection_pen.setWidth(2)
            painter.setPen(selection_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if self.ptype == 0:
                radius = constants.POINT_RADIUS
                painter.drawEllipse(QPointF(0, 0), radius + 1, radius + 1) # Slightly larger for outline
            elif self.ptype == 1:
                side = constants.SQUARE_SIDE
                painter.drawRect(-side / 2 - 1, -side / 2 - 1, side + 2, side + 2)
            elif self.ptype == 2:
                side = constants.TRIANGLE_SIDE
                points = QPolygonF([
                    QPointF(0, -side / 2 - 1),
                    QPointF(-side / 2 * (3**0.5 / 2) - 1, side / 2 * (1 / 2) + 1),
                    QPointF(side / 2 * (3**0.5 / 2) + 1, side / 2 * (1 / 2) + 1)
                ])
                painter.drawPolygon(points)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = self.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_start is not None and self.pos() != self._drag_start:
            stack = self.scene().parent().window().undo_stack
            from commands import MovePointCommand
            stack.push(MovePointCommand(self, self._drag_start, self.pos()))
        self._drag_start = None
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu()
        delete_act = menu.addAction(f"Delete Point {self.pid}")
        delete_act.triggered.connect(lambda _=False: self.scene().parent().window().delete_selected())
        
        menu.addSeparator()

        # Visibility submenu
        visibility_menu = menu.addMenu("Visibility")
        vis_group = QActionGroup(visibility_menu)
        for vis_code, vis_name in constants.VISIBILITY_NAMES.items():
            act = QAction(f"Set to {vis_name}", visibility_menu)
            act.setCheckable(True)
            act.setChecked(self.visibility == vis_code)
            act.triggered.connect(lambda _, v=vis_code: self.scene().parent().window().change_visibility_selected(v))
            visibility_menu.addAction(act)
            vis_group.addAction(act)

        menu.addSeparator()
        
        # Type submenu
        type_menu = menu.addMenu("Type")
        type_group = QActionGroup(type_menu)
        for t, name in constants.TYPE_NAMES.items():
            act = QAction(f"Set type {t} – {name}", type_menu)
            act.setCheckable(True)
            act.setChecked(self.ptype == t)
            act.triggered.connect(lambda _, tt=t: self.scene().parent().window().change_type_selected(tt))
            type_menu.addAction(act)
            type_group.addAction(act)
            
        menu.exec(event.screenPos())