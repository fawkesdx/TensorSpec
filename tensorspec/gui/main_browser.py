import sys
import os
# --- Python Version Check ---
if sys.version_info < (3, 11):
    sys.exit("\n[ERROR] TensorSpec requires Python 3.11 or newer.\n"
             f"You are currently running Python {sys.version_info.major}.{sys.version_info.minor}.\n"
             "Please upgrade your Python environment or create a new virtual environment using python3.11.\n")
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QTreeWidget, QTreeWidgetItem, QToolBar, 
                               QSplitter, QTextEdit, QLabel, QApplication)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon
from tensorspec.gui.suites.crystal_suite import CrystalViewerSuite
from tensorspec.gui.suites.arpes_suite import ARPESSuite
from tensorspec.gui.suites.dft_suite import DFTSuite

from tensorspec.gui.components.data_viewer_panel import DataViewerPanel

from tensorspec.core.workspace import global_workspace
from PySide6.QtWidgets import QPushButton

class TensorSpecMainBrowser(QMainWindow):
    """
    The Central Data Workspace Browser for TensorSpec.
    Manages in-memory data trees and coordinates independent analytical suites.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorSpec Workspace Browser")
        self.resize(1000, 700)
        
        # Central Memory Workspace Pool (Dictionary tracking active variables)
        self.workspace_data = {}
        
        # Initialize UI Components
        self.init_menubar()
        self.init_launcher_toolbar()
        self.init_central_layout()
        
        # Populate with mock data for initial UI demonstration
        self.load_mock_workspace()

    def init_menubar(self):
        """Initializes top-level system menus."""
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        
        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def init_launcher_toolbar(self):
        """Builds the App Launcher Ribbon for launching independent functional suites."""
        launcher_toolbar = QToolBar("Suite Launcher")
        launcher_toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.TopToolBarArea, launcher_toolbar)
        
        # Define actions for all roadmap suites
        suites = [
            ("Crystal Viewer", self.launch_crystal_suite),
            ("DFT Suite", self.launch_dft_suite),
            ("ARPES Suite", self.launch_arpes_suite),
            ("PEEM Suite", self.launch_peem_suite),
            ("XAS Suite", self.launch_xas_suite),
            ("Transport Suite", self.launch_transport_suite),
            ("Machine Learning", self.launch_ml_suite),
        ]
        
        for name, slot in suites:
            action = QAction(name, self)
            action.triggered.connect(slot)
            launcher_toolbar.addAction(action)

    def init_central_layout(self):
        """Creates the split workspace: Left Data Tree and Right Inspector Panel."""
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left Panel: Data Tree Explorer
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # --- NEW: Workspace Controls ---
        top_left_layout = QHBoxLayout()
        top_left_layout.addWidget(QLabel("<b>Active Workspace Variables</b>"))
        self.btn_refresh_tree = QPushButton("🔄 Refresh Workspace")
        self.btn_refresh_tree.clicked.connect(self.refresh_workspace_tree)
        top_left_layout.addWidget(self.btn_refresh_tree)
        left_layout.addLayout(top_left_layout)
        
        self.data_tree_widget = QTreeWidget()
        self.data_tree_widget.setHeaderLabels(["Variable Name", "Type", "Dimensions / Shape"])
        self.data_tree_widget.itemClicked.connect(self.on_item_selected)
        left_layout.addWidget(self.data_tree_widget)
        
        main_splitter.addWidget(left_widget)
        
        # Right Panel: Quick Inspector / Preview
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("<b>Variable Metadata Inspector</b>"))
        
        self.metadata_inspector = QTextEdit()
        self.metadata_inspector.setReadOnly(True)
        self.metadata_inspector.setPlaceholderText("Select a workspace item to view properties...")
        right_layout.addWidget(self.metadata_inspector)
        
        # --- NEW: Launch Viewer Button ---
        self.btn_launch_viewer = QPushButton("📊 Launch Data Viewer for Selected Item")
        self.btn_launch_viewer.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2b5c8f; color: white;")
        self.btn_launch_viewer.setEnabled(False)  # Disabled until an item is selected
        self.btn_launch_viewer.clicked.connect(self.launch_standalone_viewer)
        right_layout.addWidget(self.btn_launch_viewer)
        
        main_splitter.addWidget(right_widget)
        
        # Adjust initial splitter sizes (35% left, 65% right)
        main_splitter.setSizes([350, 650])
        self.setCentralWidget(main_splitter)

    def load_mock_workspace(self):
        """Injects baseline samples into memory to verify tree hierarchy behavior."""
        # Mocking an ARPES DataTree
        self.workspace_data["maestro_sample_data"] = {
            "type": "xarray.DataTree (ARPES)",
            "dims": "['energy', 'slitangle'] -> (1024, 480)",
            "metadata": "Facility: Advanced Light Source\nBeamline: MAESTRO (7.0.2.1)\nTemperature: 12 K\nPhoton Energy: 92 eV\nNodes:\n  /raw\n  /processed\n  /analysis"
        }
        
        # Mocking a Crystal Structure
        self.workspace_data["TaIrTe4_monolayer"] = {
            "type": "Structure (CrystalViewer)",
            "dims": "Atoms: 12, SpaceGroup: Pmn2_1",
            "metadata": "Formula: Ta4 Ir4 Te8\nLattice Constants:\n  a: 3.78 Å\n  b: 12.42 Å\n  c: 13.10 Å\nCorrugated Te upper/lower layers configured."
        }

        # Render data dictionary items to the GUI Tree widget
        for name, info in self.workspace_data.items():
            item = QTreeWidgetItem(self.data_tree_widget)
            item.setText(0, name)
            item.setText(1, info["type"])
            item.setText(2, info["dims"])

    def on_item_selected(self, item, column):
        """Triggers preview update in the Inspector Panel upon clicking a variable."""
        var_name = item.text(0)
        self.current_selected_var = var_name  # Track the current selection
        
        if var_name in self.workspace_data:
            # This is your mock data from the startup method
            meta_text = self.workspace_data[var_name]["metadata"]
            self.metadata_inspector.setText(f"Variable: {var_name}\n" + "-"*40 + f"\n{meta_text}")
            self.btn_launch_viewer.setEnabled(False) # Cannot view mock text data
        else:
            # This is real data pulled from the global_workspace
            self.metadata_inspector.setText(f"Workspace Item: {var_name}\n" + "-"*40 + "\nData object ready for inspection.")
            self.btn_launch_viewer.setEnabled(True)  # Enable the launcher!

    def launch_standalone_viewer(self):
        """Spawns the agnostic N-Dimensional Viewer panel as a floating window."""
        if not hasattr(self, 'current_selected_var'): 
            return
            
        # Create a floating wrapper window to hold the panel
        self.viewer_window = QMainWindow()
        self.viewer_window.setWindowTitle(f"TensorSpec Data Viewer - {self.current_selected_var}")
        self.viewer_window.resize(950, 750)
        
        # Instantiate your new reusable panel
        self.active_viewer_panel = DataViewerPanel()
        self.viewer_window.setCentralWidget(self.active_viewer_panel)
        
        # NOTE: Once we add a 'pull_tensor_data()' method to global_workspace, 
        # we will fetch the data here and pass it in like this:
        # data_tensor = global_workspace.pull_tensor_data(self.current_selected_var)
        # self.active_viewer_panel.load_data(data_tensor)
        
        self.viewer_window.show()
        
    def launch_dft_suite(self):
        """Spawns the DFT & Tight Binding Suite window."""
        self.dft_window = DFTSuite()
        self.dft_window.resize(900, 600)
        self.dft_window.setWindowTitle("TensorSpec - DFT Suite")
        self.dft_window.show()

    def launch_arpes_suite(self):
        """Spawns the ARPES Suite window containing the Matrix Simulator."""
        self.arpes_window = ARPESSuite()
        self.arpes_window.resize(1100, 700)
        self.arpes_window.setWindowTitle("TensorSpec - ARPES Suite")
        self.arpes_window.show()

    def launch_peem_suite(self):
        print("Launching PEEM Suite Window...")

    def launch_xas_suite(self):
        print("Launching XAS Suite Window...")

    def launch_transport_suite(self):
        print("Launching Transport Suite Window...")

    def launch_ml_suite(self):
        print("Launching Machine Learning Suite Window...")
    
    def launch_crystal_suite(self):
        """Spawns the Crystal Viewer window and connects it to the central workspace."""
        # If the window exists but was closed/deleted, reset the reference
        if hasattr(self, 'crystal_window'):
            try:
                self.crystal_window.close()
            except RuntimeError:
                pass # The C++ object was already deleted by WA_DeleteOnClose

        self.crystal_window = CrystalViewerSuite(workspace_manager=self.workspace_data)
        self.crystal_window.resize(1100, 700)
        self.crystal_window.setWindowTitle("TensorSpec - Crystal Suite")
        self.crystal_window.show()
    
    def refresh_workspace_tree(self):
        """Pulls the actual active data from global_workspace and populates the tree."""
        self.data_tree_widget.clear()
        
        # 1. Fetch Crystal Structures
        for name in global_workspace.list_crystal_structures():
            item = QTreeWidgetItem(self.data_tree_widget)
            item.setText(0, name)
            item.setText(1, "Crystal Structure")
            item.setText(2, "3D Basis Vectors")
            
        # 2. Fetch Band Structures
        for name in global_workspace.list_band_structures():
            item = QTreeWidgetItem(self.data_tree_widget)
            item.setText(0, name)
            item.setText(1, "Band Structure")
            item.setText(2, "E(k) Dispersion Data")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TensorSpecMainBrowser()
    window.show()
    sys.exit(app.exec())