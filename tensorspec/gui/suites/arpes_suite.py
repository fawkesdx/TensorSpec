from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from tensorspec.gui.suites.arpes_matrix_gui import MatrixElementGUI

class ARPESSuite(QWidget):
    """
    Main container for all ARPES-related tools. 
    Implements a tabbed architecture to separate simulation from real data analysis.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Initialize the Tabbed Architecture
        self.tabs = QTabWidget()
        
        # Tab 1: Matrix Element Simulator
        self.matrix_tab = MatrixElementGUI()
        self.tabs.addTab(self.matrix_tab, "Matrix Element Simulator")
        
        # Tab 2: Experimental Data Viewer (Placeholder for future roadmap items)
        self.data_tab = QWidget()
        data_layout = QVBoxLayout(self.data_tab)
        data_layout.addWidget(QLabel("Real ARPES Data Viewer & Crosshairs will go here."))
        self.tabs.addTab(self.data_tab, "Data Viewer")
        
        self.layout.addWidget(self.tabs)