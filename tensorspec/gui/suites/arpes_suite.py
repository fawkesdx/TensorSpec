import sys
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QLabel, QApplication)
from PySide6.QtCore import Qt

from tensorspec.gui.components.arpes_panel import ARPESPanel

class ARPESSuite(QWidget):
    """
    Main container for all ARPES-related tools. 
    Now fully modularized: tabs pull from isolated component files.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._is_closing = False
        
        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        
        # Tab 1: Hierarchical Simulation Engine (Now loading from the new modular file!)
        self.simulation_tab = ARPESPanel()
        self.tabs.addTab(self.simulation_tab, "Matrix Element Simulator")
        
        # Tab 2: Experimental Data Viewer
        self.data_tab = QWidget()
        data_layout = QVBoxLayout(self.data_tab)
        data_layout.addWidget(QLabel("Real ARPES Data Viewer & Crosshairs will go here."))
        self.tabs.addTab(self.data_tab, "Data Viewer")
        
        self.layout.addWidget(self.tabs)
        
    def closeEvent(self, event):
        self._is_closing = True
        event.accept()

# Standalone runner for independent testing
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ARPESSuite()
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec())