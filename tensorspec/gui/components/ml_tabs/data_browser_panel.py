"""Left-column disk browser + RAM workspace for the ML suite."""
import os
import pickle

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from tensorspec.core.data_models import TensorData
from tensorspec.core.workspace import global_workspace
from tensorspec.gui.components.ml_tabs.legacy_data import convert_to_tensor_data, tensor_to_ml_dict
from tensorspec.gui.maestroai.maestro_loader import LoadWorker
from tensorspec.gui.maestroai.maestro_fermi_viewer import FermiViewerWindow
from tensorspec.gui.maestroai.maestroai_guides import MasterGuideDialog


class DataBrowserPanel(QWidget):
    """Disk file browser, workspace list, and ML session save/load."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._workspace_tensors = {}
        self._loader = None
        self._build()
        self.session.workspace_changed.connect(self._refresh_workspace_list)

    @property
    def workspace_tensors(self):
        """TensorData objects keyed by workspace name (MLSuite may read these)."""
        return self._workspace_tensors

    @property
    def workspace(self):
        return self.session.workspace

    @property
    def current_view_data(self):
        return self.session.current_view_data

    @current_view_data.setter
    def current_view_data(self, value):
        self.session.current_view_data = value

    def _build(self):
        layout = QVBoxLayout(self)

        self.btn_master_help = QPushButton("🚀 Getting Started: Workflow Guide")
        self.btn_master_help.setStyleSheet(
            "font-weight: bold; color: #ff7f0e; padding: 8px; font-size: 15px;"
        )
        self.btn_master_help.clicked.connect(self.show_master_guide)
        layout.addWidget(self.btn_master_help)
        layout.addSpacing(10)

        layout.addWidget(QLabel("<b>1. Files on Disk</b>"))

        dir_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Paste folder path and press Enter...")
        self.path_input.returnPressed.connect(self.load_directory_from_input)

        self.btn_dir = QPushButton("Browse")
        self.btn_dir.clicked.connect(self.select_directory)

        dir_layout.addWidget(self.path_input)
        dir_layout.addWidget(self.btn_dir)
        layout.addLayout(dir_layout)

        self.disk_list = QListWidget()
        self.btn_load = QPushButton("Load to Workspace")
        self.btn_load.clicked.connect(self.request_load)

        self.workspace_list = QListWidget()
        self.workspace_list.itemClicked.connect(self.activate_data)

        self.workspace_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.workspace_list.customContextMenuRequested.connect(self.show_workspace_menu)

        session_layout = QHBoxLayout()
        self.btn_save_session = QPushButton("Save ML Session")
        self.btn_save_session.clicked.connect(self.save_session)
        self.btn_load_session = QPushButton("Load ML Session")
        self.btn_load_session.clicked.connect(self.load_session)
        session_layout.addWidget(self.btn_save_session)
        session_layout.addWidget(self.btn_load_session)

        layout.addWidget(self.disk_list)
        layout.addWidget(self.btn_load)
        layout.addWidget(QLabel("<b>2. RAM Workspace</b>"))
        layout.addWidget(self.workspace_list)
        layout.addLayout(session_layout)

    def _refresh_workspace_list(self):
        self.workspace_list.clear()
        self.workspace_list.addItems(sorted(self.session.workspace.keys()))

    def show_master_guide(self):
        MasterGuideDialog(self).exec()

    def select_directory(self):
        start_path = self.session.current_folder or ""
        path = QFileDialog.getExistingDirectory(self, "Select Folder", start_path)
        if path:
            self.path_input.setText(path)
            self.load_directory(path)

    def load_directory_from_input(self):
        path = self.path_input.text().strip()
        self.load_directory(path)

    def load_directory(self, path):
        if os.path.isdir(path):
            self.session.current_folder = path
            self.disk_list.clear()
            self.disk_list.addItems(
                [f for f in sorted(os.listdir(path)) if f.endswith(".h5")]
            )
        else:
            QMessageBox.warning(
                self,
                "Invalid Path",
                "The specified folder does not exist. Please check the path and try again.",
            )

    def _refresh_viewer_ml_layers(self, select_layer: str | None = None):
        """Push ML domain/label arrays into the middle DataViewer without resetting the grid."""
        if not self.session.current_view_data or self.session.viewer is None:
            return
        self.session.viewer.sync_ml_layers(self.session.current_view_data)
        if select_layer:
            self.session.viewer.focus_spatial_layer(select_layer)

    def show_workspace_menu(self, pos):
        item = self.workspace_list.itemAt(pos)
        if not item:
            return
        data = self.workspace[item.text()]

        menu = QMenu(self)
        if data.get("kind") == "XY Scan (Cleaned)":
            compare_action = menu.addAction("Open in Floating Comparison Window")
            action = menu.exec(self.workspace_list.mapToGlobal(pos))
            if action == compare_action:
                self.open_floating_viewer(data, item.text())
        elif data.get("kind") == "Fermi Map (Cleaned)":
            open_action = menu.addAction("Open 3D Fermi Viewer")
            action = menu.exec(self.workspace_list.mapToGlobal(pos))
            if action == open_action:
                self.fermi_dialog = FermiViewerWindow(data, self)
                self.fermi_dialog.show()

    def open_floating_viewer(self, data, title):
        from tensorspec.gui.components.data_viewer_panel import DataViewerPanel

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Comparison Viewer: {title}")
        dialog.resize(1000, 600)
        layout = QVBoxLayout(dialog)
        floating_viewer = DataViewerPanel()
        floating_viewer.load_data(convert_to_tensor_data(data))
        layout.addWidget(floating_viewer)
        dialog.show()

    def save_session(self):
        if not self.session.current_view_data:
            QMessageBox.warning(
                self,
                "No Data",
                "Please select a loaded file from the Workspace first.",
            )
            return

        session_data = {}
        for key, val in self.session.current_view_data.items():
            if (
                key.startswith("embeddings_")
                or key.startswith("domains_")
                or key == "Supervised Probabilities"
            ):
                session_data[key] = val

        if not session_data:
            QMessageBox.warning(
                self,
                "No ML Data",
                "There are no embeddings or clustering labels to save yet. Run some analysis first!",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save ML Session", "MAESTRO_Session.pkl", "Pickle Files (*.pkl)"
        )
        if not path:
            return

        try:
            with open(path, "wb") as f:
                pickle.dump(session_data, f)
            self.session.set_status(0, f"✅ Saved lightweight ML session to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save session:\n{str(e)}")

    def load_session(self):
        if not self.session.current_view_data:
            QMessageBox.warning(
                self,
                "No Raw Data",
                "Please load and activate the raw .h5 file in the Workspace first so we have somewhere to put the labels!",
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Load ML Session", "", "Pickle Files (*.pkl)"
        )
        if not path:
            return

        try:
            with open(path, "rb") as f:
                session_data = pickle.load(f)

            for key, val in session_data.items():
                self.session.current_view_data[key] = val

            if self.session.viewer is not None:
                self.session.viewer.load_data(
                    convert_to_tensor_data(self.session.current_view_data)
                )
            self.session.activate(self.session.current_view_data)
            self.session.notify_domains()
            self.session.notify_embeddings()

            current = self.workspace_list.currentItem()
            name = current.text() if current else "workspace"
            self.session.set_status(
                0, f"✅ Successfully injected ML session into {name}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load session:\n{str(e)}")

    def request_load(self):
        item = self.disk_list.currentItem()
        if not item:
            return
        file_path = os.path.join(self.session.current_folder, item.text())
        var_name = "MAESTRO_" + item.text().replace(".h5", "").replace("-", "_")

        self.session.set_status(10, "Loading...")
        self.btn_load.setEnabled(False)
        self._loader = LoadWorker(file_path, var_name)
        self._loader.progress.connect(self._on_load_progress)
        self._loader.finished.connect(self.on_load_finish)
        self._loader.error.connect(
            lambda e: QMessageBox.critical(self, "Loader Error", str(e))
        )
        self._loader.start()

    def _on_load_progress(self, val, msg):
        self.session.set_status(val, msg)

    def on_load_finish(self, var_name, tensor_data):
        ml_data = tensor_to_ml_dict(tensor_data)
        self.session.add_dataset(var_name, ml_data)
        self._workspace_tensors[var_name] = tensor_data
        global_workspace.push_spectroscopy_data(var_name, tensor_data)
        if self.session.viewer is not None:
            self.session.viewer.load_data(tensor_data)
        self.session.set_status(0, "Ready.")
        self.btn_load.setEnabled(True)

    def activate_data(self, item):
        name = item.text()
        data = self._workspace_tensors.get(name, self.workspace[name])

        if isinstance(data, TensorData):
            ml_data = tensor_to_ml_dict(data)
            self.workspace[name] = ml_data
            self.session.activate(ml_data)
            if self.session.viewer is not None:
                self.session.viewer.load_data(data)
            return

        if data.get("kind") == "Fermi Map (Cleaned)":
            reply = QMessageBox.question(
                self,
                "3D Data Detected",
                "This is a 3D Fermi Map. The main dashboard is designed for 4D Spatial Scans.\n\n"
                "Would you like to open this map in a floating 3D viewer instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.fermi_dialog = FermiViewerWindow(data, self)
                self.fermi_dialog.show()
            return

        if data.get("kind") == "XY Scan (Cleaned)":
            self.session.activate(data)
            if self.session.viewer is not None:
                self.session.viewer.load_data(convert_to_tensor_data(data))
            self.session.notify_domains()
            self.session.notify_embeddings()
