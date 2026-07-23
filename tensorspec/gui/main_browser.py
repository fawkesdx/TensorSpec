import sys
import os
# --- Python Version Check ---
if sys.version_info < (3, 11):
    sys.exit("\n[ERROR] TensorSpec requires Python 3.11 or newer.\n"
             f"You are currently running Python {sys.version_info.major}.{sys.version_info.minor}.\n"
             "Please upgrade your Python environment or create a new virtual environment using python3.11.\n")
             
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QTreeWidget, QTreeWidgetItem, QToolBar, 
                               QSplitter, QTextEdit, QLabel, QApplication, 
                               QPushButton, QListWidget, QMessageBox, QMenu)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QIcon

from tensorspec.gui.suites.crystal_suite import CrystalViewerSuite
from tensorspec.gui.suites.arpes_suite import ARPESSuite
from tensorspec.gui.suites.dft_suite import DFTSuite
from tensorspec.gui.components.data_viewer_panel import DataViewerPanel
from tensorspec.core.workspace import global_workspace

import time
import platform
import psutil
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QMessageBox

class TelemetryWindow(QWidget):
    """A hidden developer dashboard tracking and recording live system resource consumption."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorSpec Telemetry - Hardware Profiler")
        self.resize(800, 450)
        self.setWindowFlags(Qt.WindowStaysOnTopHint) 
        
        # --- Recording States ---
        self.is_recording = False
        self.history_time = []
        self.history_cpu = []
        self.history_ram = []
        self.start_time = None
        
        layout = QVBoxLayout(self)
        
        # --- UI: Top Control Ribbon ---
        ctrl_layout = QHBoxLayout()
        
        self.btn_record = QPushButton("🔴 Start Recording")
        self.btn_record.setStyleSheet("font-weight: bold; background-color: #d9534f; color: white; padding: 6px;")
        self.btn_record.clicked.connect(self.toggle_recording)
        ctrl_layout.addWidget(self.btn_record)
        
        self.btn_save = QPushButton("💾 Save Recorded Graph")
        self.btn_save.setEnabled(False) # Disabled until a recording finishes
        self.btn_save.clicked.connect(self.save_graph)
        ctrl_layout.addWidget(self.btn_save)
        
        layout.addLayout(ctrl_layout)
        
        # --- UI: Live Matplotlib Canvas ---
        self.fig = Figure(figsize=(8, 4), dpi=100, layout='tight')
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        self.ax_cpu = self.fig.add_subplot(121)
        self.ax_ram = self.fig.add_subplot(122)
        
        # Start tracking
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_metrics)
        self.timer.start(1000)
        self.update_metrics()

    def toggle_recording(self):
        """Switches between recording and stopped states."""
        if not self.is_recording:
            # Begin Recording
            self.is_recording = True
            self.history_time.clear()
            self.history_cpu.clear()
            self.history_ram.clear()
            self.start_time = time.time()
            
            self.btn_record.setText("⏹ Stop Recording")
            self.btn_record.setStyleSheet("font-weight: bold; background-color: #5bc0de; color: black; padding: 6px;")
            self.btn_save.setEnabled(False)
        else:
            # Stop Recording
            self.is_recording = False
            self.btn_record.setText("🔴 Start Recording")
            self.btn_record.setStyleSheet("font-weight: bold; background-color: #d9534f; color: white; padding: 6px;")
            self.btn_save.setEnabled(len(self.history_time) > 0)

    def update_metrics(self):
        """Polls the hardware every 1 second and draws the live bar charts."""
        self.ax_cpu.clear()
        self.ax_ram.clear()
        
        # 1. Hardware Polling
        cpu_percents = psutil.cpu_percent(percpu=True)
        total_cpu = psutil.cpu_percent()
        cores = range(len(cpu_percents))
        
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        
        # 2. Draw Live CPU Bar Chart
        self.ax_cpu.bar(cores, cpu_percents, color='#5bc0de', edgecolor='black')
        self.ax_cpu.set_ylim(0, 100)
        self.ax_cpu.set_title(f"CPU Usage per Thread (Total: {total_cpu}%)")
        self.ax_cpu.set_xlabel("Logical Core ID")
        self.ax_cpu.set_ylabel("Utilization (%)")
        
        # 3. Draw Live RAM Bar Chart
        mem_label = "Unified Memory" if platform.system() == "Darwin" else "System RAM"
        self.ax_ram.bar([mem_label], [mem.percent], color='#d9534f', edgecolor='black', width=0.4)
        self.ax_ram.set_ylim(0, 100)
        self.ax_ram.set_title(f"RAM Usage: {used_gb:.1f} GB / {total_gb:.1f} GB")
        self.ax_ram.set_ylabel("Capacity Used (%)")
        
        self.canvas.draw()
        
        # 4. Log the data if recording is active
        if self.is_recording:
            elapsed_seconds = time.time() - self.start_time
            self.history_time.append(elapsed_seconds)
            self.history_cpu.append(total_cpu)
            self.history_ram.append(mem.percent)

    def save_graph(self):
        """Generates a beautiful time-series plot of the recorded session and saves it."""
        if not self.history_time:
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Telemetry Graph", "tensorspec_stress_test.png", "PNG Images (*.png);;PDF Files (*.pdf)")
        if not path:
            return
            
        try:
            # Create a separate, hidden figure specifically for the exported line graph
            export_fig = Figure(figsize=(8, 5), dpi=150, layout='tight')
            ax = export_fig.add_subplot(111)
            
            # Plot the recorded data
            ax.plot(self.history_time, self.history_cpu, label='Total CPU Usage (%)', color='#5bc0de', linewidth=2.5)
            ax.plot(self.history_time, self.history_ram, label='RAM Usage (%)', color='#d9534f', linewidth=2.5)
            
            # Style it professionally for a presentation
            ax.set_title("TensorSpec Hardware Stress Test over Time", fontsize=14, fontweight='bold')
            ax.set_xlabel("Elapsed Time (Seconds)", fontsize=12)
            ax.set_ylabel("System Utilization (%)", fontsize=12)
            ax.set_ylim(0, 105)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper left', fontsize=11)
            
            export_fig.savefig(path)
            QMessageBox.information(self, "Export Successful", f"Time-series graph saved successfully to:\n{path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save telemetry graph:\n{str(e)}")

# --- NEW: Floating Window Wrapper ---
class FloatingViewerWindow(QMainWindow):
    """A wrapper that turns any panel into a tracked floating window."""
    window_closed = Signal(str)

    def __init__(self, win_id: str, title: str, inner_widget: QWidget, parent=None):
        super().__init__(parent)
        self.win_id = win_id
        self.setWindowTitle(f"TensorSpec Viewer: {title}")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(1200, 800)
        
        # Ensures the window floats independently but retains OS event focus
        self.setWindowFlags(Qt.Window) 

        self.setCentralWidget(inner_widget)

    def closeEvent(self, event):
        # Notify the Main Browser registry that this window is closing
        self.window_closed.emit(self.win_id)
        super().closeEvent(event)


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
        self.active_windows = {}  # NEW: Window Registry
        
        # Initialize UI Components
        self.init_menubar()
        self.init_launcher_toolbar()
        self.init_central_layout()
        
        # Populate with mock data for initial UI demonstration
        self.load_mock_workspace()
        self.refresh_workspace_tree()

        # Populate with mock data for initial UI demonstration
        self.load_mock_workspace()
        self.refresh_workspace_tree()
        
        # NEW: Secret keystroke buffer for the telemetry Easter egg
        self._secret_buffer = ""

    def keyPressEvent(self, event):
        """Listens for the secret Easter egg phrase to unlock the telemetry panel."""
        if event.text():
            # Append the typed character and keep the buffer lowercase
            self._secret_buffer += event.text().lower()
            
            # Keep the buffer from growing infinitely (the phrase is 21 chars long)
            self._secret_buffer = self._secret_buffer[-30:]
            
            # Check if the secret phrase is inside the buffer
            if "should i buy new mac?" in self._secret_buffer:
                self._secret_buffer = "" # Reset buffer
                self.launch_telemetry_panel()
                
        super().keyPressEvent(event)
    
    def show_secret_menu(self, pos):
        """Displays the hidden developer menu when the user right-clicks the designated label."""
        menu = QMenu(self)
        secret_act = menu.addAction("Should I buy a new mac?")
        
        # Map the local click position to global screen coordinates so the menu spawns on your mouse
        action = menu.exec(self.lbl_active_vars.mapToGlobal(pos))
        
        if action == secret_act:
            self.launch_telemetry_panel()

    def launch_telemetry_panel(self):
        """Spawns the hidden hardware profiler."""
        if not hasattr(self, 'telemetry_window') or not self.telemetry_window.isVisible():
            self.telemetry_window = TelemetryWindow()
            self.telemetry_window.show()

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
        
        # --- Left Panel: Data Tree Explorer & Window Registry ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        top_left_layout = QHBoxLayout()
        
        # --- NEW: Secret Right-Click Target ---
        self.lbl_active_vars = QLabel("<b>Active Workspace Variables</b>")
        self.lbl_active_vars.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lbl_active_vars.customContextMenuRequested.connect(self.show_secret_menu)
        top_left_layout.addWidget(self.lbl_active_vars)
        
        self.btn_refresh_tree = QPushButton("🔄 Refresh Workspace")
        self.btn_refresh_tree.clicked.connect(self.refresh_workspace_tree)
        top_left_layout.addWidget(self.btn_refresh_tree)
        left_layout.addLayout(top_left_layout)
        
        self.data_tree_widget = QTreeWidget()
        self.data_tree_widget.setHeaderLabels(["Variable Name", "Type", "Dimensions / Shape"])
        self.data_tree_widget.currentItemChanged.connect(self.on_item_selected)
        left_layout.addWidget(self.data_tree_widget)
        
        # NEW: Active Windows Registry Tracker
        left_layout.addWidget(QLabel("<b>Active Floating Windows</b>"))
        self.window_tracker_list = QListWidget()
        self.window_tracker_list.setMaximumHeight(150)
        self.window_tracker_list.itemClicked.connect(self.bring_window_to_front)
        left_layout.addWidget(self.window_tracker_list)
        
        main_splitter.addWidget(left_widget)
        
        # --- Right Panel: Quick Inspector / Preview ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("<b>Variable Metadata Inspector</b>"))
        
        self.metadata_inspector = QTextEdit()
        self.metadata_inspector.setReadOnly(True)
        self.metadata_inspector.setPlaceholderText("Select a workspace item to view properties...")
        right_layout.addWidget(self.metadata_inspector)
        
        self.btn_launch_viewer = QPushButton("📊 Launch Data Viewer for Selected Item")
        self.btn_launch_viewer.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2b5c8f; color: white;")
        self.btn_launch_viewer.setEnabled(False)
        self.btn_launch_viewer.clicked.connect(self.launch_standalone_viewer)
        right_layout.addWidget(self.btn_launch_viewer)
        
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([350, 650])
        self.setCentralWidget(main_splitter)

    def load_mock_workspace(self):
        """Injects baseline samples into memory to verify tree hierarchy behavior."""
        self.workspace_data["maestro_sample_data"] = {
            "type": "xarray.DataTree (ARPES)",
            "dims": "['energy', 'slitangle'] -> (1024, 480)",
            "metadata": "Facility: Advanced Light Source\nBeamline: MAESTRO (7.0.2.1)\nTemperature: 12 K\nPhoton Energy: 92 eV\nNodes:\n  /raw\n  /processed\n  /analysis"
        }
        
        self.workspace_data["TaIrTe4_monolayer"] = {
            "type": "Structure (CrystalViewer)",
            "dims": "Atoms: 12, SpaceGroup: Pmn2_1",
            "metadata": "Formula: Ta4 Ir4 Te8\nLattice Constants:\n  a: 3.78 Å\n  b: 12.42 Å\n  c: 13.10 Å\nCorrugated Te upper/lower layers configured."
        }

    def refresh_workspace_tree(self):
        """Pulls the actual active data from global_workspace and populates the tree."""
        self.data_tree_widget.clear()
        
        # Render mock data
        for name, info in self.workspace_data.items():
            item = QTreeWidgetItem(self.data_tree_widget)
            item.setText(0, name)
            item.setText(1, info["type"])
            item.setText(2, info["dims"])

        # Fetch Crystal Structures
        for name in global_workspace.list_crystal_structures():
            item = QTreeWidgetItem(self.data_tree_widget)
            item.setText(0, name)
            item.setText(1, "Crystal Structure")
            item.setText(2, "3D Basis Vectors")
            
        # Fetch Band Structures
        for name in global_workspace.list_band_structures():
            item = QTreeWidgetItem(self.data_tree_widget)
            item.setText(0, name)
            item.setText(1, "Band Structure")
            item.setText(2, "E(k) Dispersion Data")

        # Fetch Real Spectroscopy DataTrees (ARPES Data)
        for name, item_data in global_workspace._data.items():
            if item_data.get('type') == 'spectroscopy_tree':
                item = QTreeWidgetItem(self.data_tree_widget)
                item.setText(0, name)
                item.setText(1, "Spectroscopy DataTree")
                item.setText(2, "N-Dimensional Tensor")

    def on_item_selected(self, current_item, previous_item):
        """Triggers preview update in the Inspector Panel upon selecting a variable."""
        if not current_item: 
            return
            
        var_name = current_item.text(0)
        self.current_selected_var = var_name
        
        if var_name in self.workspace_data:
            meta_text = self.workspace_data[var_name]["metadata"]
            self.metadata_inspector.setText(f"Variable: {var_name}\n" + "-"*40 + f"\n{meta_text}")
            self.btn_launch_viewer.setEnabled(False) # Cannot view mock string data
        else:
            # Check real global workspace
            ws_item = global_workspace._data.get(var_name)
            if ws_item:
                item_type = ws_item.get('type', 'Unknown')
                self.metadata_inspector.setText(f"Workspace Item: {var_name}\nType: {item_type}\n" + "-"*40 + "\nData object ready for inspection.")
                
                # Only enable the launch button for supported viewer types
                if item_type == 'spectroscopy_tree':
                    self.btn_launch_viewer.setEnabled(True)
                else:
                    self.btn_launch_viewer.setEnabled(False)

    def launch_standalone_viewer(self):
        """Reads the workspace data type and spawns the correct floating component."""
        if not hasattr(self, 'current_selected_var'): return
        var_name = self.current_selected_var
        
        # Fetch item info from workspace
        item_info = global_workspace._data.get(var_name)
        if not item_info: return
        
        data_type = item_info.get('type')
        
        # Prevent opening the exact same dataset twice; bring it to front instead
        if var_name in self.active_windows:
            items = self.window_tracker_list.findItems(var_name, Qt.MatchExactly)
            if items:
                self.bring_window_to_front(items[0])
            return

        viewer_widget = None

        # Universal Dispatch Logic
        if data_type == 'spectroscopy_tree':
            viewer_widget = DataViewerPanel()
            
            # Safely pull the data with fallbacks
            tensor_data = global_workspace.pull_tensor_data(var_name)
            if not tensor_data:
                tensor_data = global_workspace.pull_tensor_data(var_name, node="raw")
            if not tensor_data:
                tensor_data = global_workspace.pull_tensor_data(var_name, node="/raw")
                
            if tensor_data:
                viewer_widget.load_data(tensor_data)
        
        # Wrap, Track, and Launch
        if viewer_widget:
            # Pass `self` as the parent to prevent the macOS focus-loss bug
            wrapper = FloatingViewerWindow(win_id=var_name, title=var_name, inner_widget=viewer_widget, parent=self)
            wrapper.window_closed.connect(self.unregister_window)
            
            self.active_windows[var_name] = wrapper
            self.window_tracker_list.addItem(var_name)
            wrapper.show()
            
            # Crucial: Load data AFTER the widget is anchored to a window structure
            if data_type == 'spectroscopy_tree' and tensor_data:
                viewer_widget.load_data(tensor_data)

    def bring_window_to_front(self, item):
        """Raises a specific floating window to the top of the user's screen."""
        win_id = item.text()
        if win_id in self.active_windows:
            window = self.active_windows[win_id]
            window.showNormal()
            window.raise_()
            window.activateWindow()

    def unregister_window(self, win_id: str):
        """Removes a closed window from the tracking registry and UI list."""
        if win_id in self.active_windows:
            del self.active_windows[win_id]
            
        items = self.window_tracker_list.findItems(win_id, Qt.MatchExactly)
        if items:
            self.window_tracker_list.takeItem(self.window_tracker_list.row(items[0]))

    def launch_dft_suite(self):
        self.dft_window = DFTSuite()
        self.dft_window.resize(900, 600)
        self.dft_window.setWindowTitle("TensorSpec - DFT Suite")
        self.dft_window.show()

    def launch_arpes_suite(self):
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
        if hasattr(self, 'crystal_window'):
            try:
                self.crystal_window.close()
            except RuntimeError:
                pass 

        self.crystal_window = CrystalViewerSuite(workspace_manager=self.workspace_data)
        self.crystal_window.resize(1100, 700)
        self.crystal_window.setWindowTitle("TensorSpec - Crystal Suite")
        self.crystal_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TensorSpecMainBrowser()
    window.show()
    sys.exit(app.exec())