"""XAS / XMCD suite container."""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from tensorspec.gui.components.xas_panel import XasPanel


class XASSuite(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        layout = QVBoxLayout(self)
        self.panel = XasPanel()
        layout.addWidget(self.panel)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = XASSuite()
    window.resize(1000, 700)
    window.setWindowTitle("TensorSpec - XAS / XMCD Suite")
    window.show()
    sys.exit(app.exec())
