#!/usr/bin/env python3
"""
Entry point for Badminton Court Annotator.
"""
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from window import MainWindow

def main():
    app = QApplication(sys.argv)
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    w = MainWindow(folder)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()