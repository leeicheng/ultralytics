from PyQt6.QtWidgets import QTableWidget, QAbstractItemView, QTableWidgetItem

from tools.label_tool import constants


class PointTable(QTableWidget):
    """Table widget to display point annotations."""
    HEADERS = ["ID", "X", "Y", "Type", "Visibility"]

    def __init__(self, parent=None):
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

    def load_points(self, points):
        self.setRowCount(len(points))
        for row, p in enumerate(points):
            self.setItem(row, 0, QTableWidgetItem(str(p.pid)))
            self.setItem(row, 1, QTableWidgetItem(f"{p.pos().x():.1f}"))
            self.setItem(row, 2, QTableWidgetItem(f"{p.pos().y():.1f}"))
            self.setItem(row, 3, QTableWidgetItem(constants.TYPE_NAMES[p.ptype]))
            visibility_str = constants.VISIBILITY_NAMES.get(p.visibility, "Unknown")
            self.setItem(row, 4, QTableWidgetItem(visibility_str))