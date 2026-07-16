import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyvista as pv
from pyvistaqt import QtInteractor

class DiagnosticWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyVista + PySide6 macOS Diagnostic")
        self.resize(800, 600)

        # Set up the central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Initialize the PyVista Qt Interactor
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        # Add a simple 3D sphere to test the rendering pipeline
        sphere = pv.Sphere()
        self.plotter.add_mesh(sphere, color="cyan", show_edges=True)
        self.plotter.reset_camera()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DiagnosticWindow()
    window.show()
    sys.exit(app.exec())