from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from PySide6.QtCore import Qt
from tensorspec.gui.suites.arpes_matrix_gui import MatrixElementGUI

class ARPESSuite(QWidget):
    """
    Main container for all ARPES-related tools. 
    Implements a tabbed architecture to separate simulation from real data analysis.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 1. Protection and Logging
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._is_closing = False
        print("open suite ARPES Suite")
        
        self.layout = QVBoxLayout(self)
        
        # 2. Initialize the Tabbed Architecture (This was missing!)
        self.tabs = QTabWidget()
        
        # Tab 1: Matrix Element Simulator
        self.matrix_tab = MatrixElementGUI()
        self.tabs.addTab(self.matrix_tab, "Matrix Element Simulator")
        
        # Tab 2: Experimental Data Viewer
        self.data_tab = QWidget()
        data_layout = QVBoxLayout(self.data_tab)
        data_layout.addWidget(QLabel("Real ARPES Data Viewer & Crosshairs will go here."))
        self.tabs.addTab(self.data_tab, "Data Viewer")
        
        self.layout.addWidget(self.tabs)
    
    def closeEvent(self, event):
        """Safely shuts down ARPES rendering pipelines without breaking shared contexts."""
        self._is_closing = True
        print("close suite ARPES Suite")
        
        # Target the PyVista instance inside the Matrix Simulator tab
        if hasattr(self, 'matrix_tab'):
            # Check if the matrix GUI wraps the renderer in a class
            if hasattr(self.matrix_tab, 'renderer_gpu') and hasattr(self.matrix_tab.renderer_gpu, 'plotter'):
                try:
                    if hasattr(self.matrix_tab.renderer_gpu.plotter, 'iren'):
                        self.matrix_tab.renderer_gpu.plotter.iren.remove_all_observers()
                    self.matrix_tab.renderer_gpu.plotter.setParent(None)
                    self.matrix_tab.renderer_gpu.plotter.close()
                except Exception:
                    pass
            
            # Check if the matrix GUI mounts the plotter directly
            elif hasattr(self.matrix_tab, 'plotter'):
                try:
                    if hasattr(self.matrix_tab.plotter, 'iren'):
                        self.matrix_tab.plotter.iren.remove_all_observers()
                    self.matrix_tab.plotter.setParent(None)
                    self.matrix_tab.plotter.close()
                except Exception:
                    pass
                    
        event.accept()