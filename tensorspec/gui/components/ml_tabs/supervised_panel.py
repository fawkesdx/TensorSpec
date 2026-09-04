"""Supervised few-shot learning controls (scrollable, no canvas)."""
import os

import numpy as np
from PySide6.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from tensorspec.gui.components.ml_tabs.layout import scrollable
from tensorspec.gui.maestroai.maestroai_guides import SupGuideDialog
from tensorspec.gui.maestroai.maestroai_training_sup import SupTestWorker, SupTrainWorker


class SupervisedPanel(QWidget):
    """Supervised few-shot learning controls."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.sup_data = {}
        self.sup_coords = {}
        self.sup_buttons = []
        self.trained_sup_model = None
        self._build()

    def _build(self):
        sup_tab = QWidget()
        sup_layout = QVBoxLayout(sup_tab)

        self.btn_sup_help = QPushButton("🧑‍🏫 How does Few-Shot Learning work? Click Here")
        self.btn_sup_help.setStyleSheet(
            "font-weight: bold; color: #2ca02c; padding: 6px; font-size: 14px;"
        )
        self.btn_sup_help.clicked.connect(self.show_sup_guide)
        sup_layout.addWidget(self.btn_sup_help)

        sup_ctrl_group = QGroupBox("1. Define Target Labels")
        sup_ctrl_layout = QHBoxLayout(sup_ctrl_group)
        self.spin_sup_classes = QSpinBox()
        self.spin_sup_classes.setRange(2, 10)
        self.spin_sup_classes.setValue(3)
        self.btn_create_classes = QPushButton("Generate Label Buttons")
        self.btn_create_classes.clicked.connect(self.create_sup_buttons)
        sup_ctrl_layout.addWidget(QLabel("Number of Distinct Labels:"))
        sup_ctrl_layout.addWidget(self.spin_sup_classes)
        sup_ctrl_layout.addWidget(self.btn_create_classes)

        self.sup_btn_group = QGroupBox(
            "2. Collect Data (Move sliders to target, then click!)"
        )
        self.sup_btn_layout = QVBoxLayout(self.sup_btn_group)

        sup_act_group = QGroupBox("3. Train & Infer")
        sup_act_layout = QVBoxLayout(sup_act_group)

        sup_io_layout = QHBoxLayout()
        self.btn_sup_save = QPushButton("Save Training Set")
        self.btn_sup_save.clicked.connect(self.save_sup_data)
        self.btn_sup_load = QPushButton("Load Training Set")
        self.btn_sup_load.clicked.connect(self.load_sup_data)
        sup_io_layout.addWidget(self.btn_sup_save)
        sup_io_layout.addWidget(self.btn_sup_load)

        self.btn_sup_train = QPushButton("Train Model (Few-Shot)")
        self.btn_sup_train.clicked.connect(self.train_supervised)
        self.btn_sup_test = QPushButton("Test (Classify Entire Map)")
        self.btn_sup_test.clicked.connect(self.test_supervised)

        self.btn_sup_save_results = QPushButton("Save Classification Results")
        self.btn_sup_save_results.clicked.connect(self.save_sup_results)

        self.btn_sup_reset = QPushButton("Reset All Training Data")
        self.btn_sup_reset.clicked.connect(self.reset_supervised)

        sup_act_layout.addLayout(sup_io_layout)
        sup_act_layout.addWidget(self.btn_sup_train)
        sup_act_layout.addWidget(self.btn_sup_test)
        sup_act_layout.addWidget(self.btn_sup_save_results)
        sup_act_layout.addWidget(self.btn_sup_reset)

        sup_layout.addWidget(sup_ctrl_group)
        sup_layout.addWidget(self.sup_btn_group)
        sup_layout.addWidget(sup_act_group)
        sup_layout.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scrollable(sup_tab))

    def show_sup_guide(self):
        SupGuideDialog(self).exec()

    def create_sup_buttons(self):
        for btn in self.sup_buttons:
            self.sup_btn_layout.removeWidget(btn)
            btn.deleteLater()
        self.sup_buttons.clear()
        self.sup_data.clear()
        self.sup_coords.clear()

        num_classes = self.spin_sup_classes.value()
        for i in range(num_classes):
            self.sup_data[i] = []
            self.sup_coords[i] = []
            btn = QPushButton(f"Assign Target Coordinate to Label {i + 1} (Count: 0)")
            btn.clicked.connect(lambda checked, idx=i: self.add_sup_data(idx))
            self.sup_btn_layout.addWidget(btn)
            self.sup_buttons.append(btn)

    def add_sup_data(self, idx):
        if not self.session.current_view_data:
            return

        x_c, y_c = self.session.viewer.get_current_coords()

        val = self.session.current_view_data["value"]
        band = val[:, :, y_c, x_c]
        self.sup_data[idx].append(band)
        self.sup_coords[idx].append((x_c, y_c))
        count = len(self.sup_data[idx])
        self.sup_buttons[idx].setText(
            f"Assign Target Coordinate to Label {idx + 1} (Count: {count})"
        )

    def reset_supervised(self):
        for i in range(len(self.sup_buttons)):
            self.sup_data[i] = []
            self.sup_coords[i] = []
            self.sup_buttons[i].setText(
                f"Assign Target Coordinate to Label {i + 1} (Count: 0)"
            )
        self.trained_sup_model = None
        self.session.set_status(0, "Supervised training data cleared.")

    def save_sup_data(self):
        if not self.sup_coords or not any(self.sup_coords.values()):
            QMessageBox.warning(self, "No Data", "No training data to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Training Set", "MAESTRO_Sup_Training.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                f.write("Label_Index,X_Index,Y_Index\n")
                for label_idx, coords in self.sup_coords.items():
                    for x_c, y_c in coords:
                        f.write(f"{label_idx},{x_c},{y_c}\n")
            self.session.set_status(
                100, f"✅ Saved training data to {os.path.basename(path)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def load_sup_data(self):
        if not self.session.current_view_data:
            QMessageBox.warning(
                self, "No Data", "Load an ARPES scan into the workspace first."
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Training Set", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=int)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            max_label = np.max(data[:, 0])
            if len(self.sup_buttons) <= max_label:
                self.spin_sup_classes.setValue(max_label + 1)
                self.create_sup_buttons()
            val = self.session.current_view_data["value"]
            nY, nX = val.shape[2], val.shape[3]
            for row in data:
                label_idx, x_c, y_c = row[0], row[1], row[2]
                if y_c >= nY or x_c >= nX:
                    continue
                band = val[:, :, y_c, x_c]
                self.sup_data[label_idx].append(band)
                self.sup_coords[label_idx].append((x_c, y_c))
            for i in range(len(self.sup_buttons)):
                count = len(self.sup_data[i])
                self.sup_buttons[i].setText(
                    f"Assign Target Coordinate to Label {i + 1} (Count: {count})"
                )
            self.session.set_status(
                100, f"✅ Loaded training data from {os.path.basename(path)}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Could not load file:\n{str(e)}\n\nMake sure it is a valid Training CSV.",
            )

    def train_supervised(self):
        X, Y = [], []
        for label_idx, bands in self.sup_data.items():
            for b in bands:
                X.append(b)
                Y.append(label_idx)
        if len(X) == 0:
            QMessageBox.warning(
                self, "No Data", "Please assign coordinates to labels first!"
            )
            return
        self.btn_sup_train.setEnabled(False)
        self.session.set_status(1, "Training Few-Shot CNN...")
        X_arr, Y_arr = np.array(X), np.array(Y)
        num_classes = len(self.sup_data.keys())
        self.sup_train_worker = SupTrainWorker(X_arr, Y_arr, num_classes)
        self.sup_train_worker.progress.connect(self.session.set_status)
        self.sup_train_worker.finished.connect(self.on_sup_train_finish)
        self.sup_train_worker.start()

    def on_sup_train_finish(self, trained_model):
        self.trained_sup_model = trained_model
        self.btn_sup_train.setEnabled(True)
        self.session.set_status(100, "Supervised Model Trained Successfully!")

    def test_supervised(self):
        if self.trained_sup_model is None:
            QMessageBox.warning(
                self, "No Model", "You must Train the model first!"
            )
            return

        self.active_sup_test_target = self.session.current_view_data
        val = self.active_sup_test_target["value"]

        self.btn_sup_test.setEnabled(False)
        self.session.set_status(1, "Running Full Map Inference...")
        self.sup_test_worker = SupTestWorker(self.trained_sup_model, val)
        self.sup_test_worker.progress.connect(self.session.set_status)
        self.sup_test_worker.finished.connect(self.on_sup_test_finish)
        self.sup_test_worker.start()

    def on_sup_test_finish(self, prob_map):
        mode_name = "Supervised Probabilities"

        self.active_sup_test_target[mode_name] = prob_map

        if self.session.current_view_data is self.active_sup_test_target:
            self._refresh_viewer_ml_layers(mode_name)

        self.btn_sup_test.setEnabled(True)
        self.session.set_status(100, "Inference Complete! View updated.")

    def _refresh_viewer_ml_layers(self, select_layer=None):
        if not self.session.current_view_data or self.session.viewer is None:
            return
        self.session.viewer.sync_ml_layers(self.session.current_view_data)
        if select_layer:
            self.session.viewer.focus_spatial_layer(select_layer)

    def save_sup_results(self):
        if (
            not self.session.current_view_data
            or "Supervised Probabilities" not in self.session.current_view_data
        ):
            QMessageBox.warning(
                self,
                "No Results",
                "No classification results to save. Please run "
                "'Test (Classify Entire Map)' first.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Classification Results",
            "MAESTRO_Supervised_Results.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            prob_map = self.session.current_view_data["Supervised Probabilities"]
            nY, nX, nC = prob_map.shape
            X_arr, Y_arr = self.session.current_view_data["x"], self.session.current_view_data["y"]
            X_grid, Y_grid = np.meshgrid(X_arr, Y_arr)
            x_flat, y_flat = X_grid.flatten(), Y_grid.flatten()
            predicted_labels = np.argmax(prob_map, axis=2).flatten() + 1
            max_probs = np.max(prob_map, axis=2).flatten() * 100
            header = "X,Y,Predicted_Label,Confidence_Percent"
            cols = [x_flat, y_flat, predicted_labels, max_probs]
            for c in range(nC):
                header += f",Prob_L{c + 1}"
                cols.append(prob_map[:, :, c].flatten() * 100)
            out_matrix = np.column_stack(cols)
            np.savetxt(
                path, out_matrix, delimiter=",", header=header, comments="", fmt="%g"
            )
            self.session.set_status(
                100, f"✅ Saved classification results to {os.path.basename(path)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save file:\n{str(e)}")
