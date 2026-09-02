"""Active-learning simulation controls plus truth/prediction/uncertainty canvas."""
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from tensorspec.gui.components.ml_tabs.layout import split_panel
from tensorspec.gui.maestroai.maestroai_active_learning import SimulateALWorker
from tensorspec.gui.maestroai.maestroai_guides import SimulateALGuideDialog
from tensorspec.gui.maestroai.maestroai_viewers import MplCanvas


class SimulateALPanel(QWidget):
    """Active-learning simulation controls plus truth/prediction/uncertainty canvas."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.sim_measured_mask = None
        self.sim_next_idx = None
        self.sim_auto_steps = 0
        self._build()
        self.session.domains_changed.connect(self.set_domain_keys)

    def _build(self):
        controls = QWidget()
        sim_layout = QVBoxLayout(controls)

        self.btn_sim_help = QPushButton("🎮 How to use the AL Simulator? Click Here")
        self.btn_sim_help.setStyleSheet(
            "font-weight: bold; color: #17becf; padding: 6px; font-size: 14px;"
        )
        self.btn_sim_help.clicked.connect(self.show_sim_guide)
        sim_layout.addWidget(self.btn_sim_help)

        sim_ctrl_group = QGroupBox("Simulation Controls:")
        sim_c_layout = QVBoxLayout(sim_ctrl_group)

        self.combo_sim_domain = QComboBox()
        sim_c_layout.addWidget(QLabel("Ground Truth Domain (From Clustering):"))
        sim_c_layout.addWidget(self.combo_sim_domain)

        self.combo_sim_algo = QComboBox()
        self.combo_sim_algo.addItems([
            "Bayesian Network (GPU)", "Deep Ensembles (GPU)",
            "Evidential Deep Learning (GPU)",
            "Gaussian Process (CPU)", "Random Forest (CPU)",
        ])
        sim_c_layout.addWidget(QLabel("Steering Algorithm:"))
        sim_c_layout.addWidget(self.combo_sim_algo)

        h_sim = QHBoxLayout()
        self.spin_sim_points = QSpinBox()
        self.spin_sim_points.setRange(2, 50)
        self.spin_sim_points.setValue(10)
        h_sim.addWidget(QLabel("Initial Random Seed Points:"))
        h_sim.addWidget(self.spin_sim_points)

        self.spin_sim_ff = QSpinBox()
        self.spin_sim_ff.setRange(1, 500)
        self.spin_sim_ff.setValue(10)
        h_sim.addWidget(QLabel("Fast-Forward Steps:"))
        h_sim.addWidget(self.spin_sim_ff)
        sim_c_layout.addLayout(h_sim)

        btn_layout = QHBoxLayout()
        self.btn_sim_init = QPushButton("1. Initialize Simulation")
        self.btn_sim_init.setStyleSheet("font-weight: bold; color: #1f77b4;")
        self.btn_sim_init.clicked.connect(self.run_sim_init)

        self.btn_sim_step = QPushButton("2. Simulate 1 Step")
        self.btn_sim_step.setStyleSheet("font-weight: bold; color: #d62728;")
        self.btn_sim_step.setEnabled(False)
        self.btn_sim_step.clicked.connect(self.run_sim_step)

        self.btn_sim_ff = QPushButton("3. Fast-Forward")
        self.btn_sim_ff.setStyleSheet("font-weight: bold; color: #9467bd;")
        self.btn_sim_ff.setEnabled(False)
        self.btn_sim_ff.clicked.connect(self.run_sim_ff)

        btn_layout.addWidget(self.btn_sim_init)
        btn_layout.addWidget(self.btn_sim_step)
        btn_layout.addWidget(self.btn_sim_ff)

        self.sim_canvas = MplCanvas(
            self, width=5, height=4, is_3d=False, orientation="vertical_3"
        )
        self.ax_sim_truth, self.ax_sim_pred, self.ax_sim_uncert = self.sim_canvas.axes

        sim_layout.addWidget(sim_ctrl_group)
        sim_layout.addLayout(btn_layout)
        sim_layout.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split_panel(controls, self.sim_canvas, sizes=(300, 500)))

    def set_domain_keys(self, keys):
        """Repopulate the domain combo; replaces the write from activate_data."""
        self.combo_sim_domain.blockSignals(True)
        self.combo_sim_domain.clear()
        self.combo_sim_domain.addItems(keys)
        self.combo_sim_domain.blockSignals(False)

    def show_sim_guide(self):
        SimulateALGuideDialog(self).exec()

    def run_sim_init(self):
        if not self.session.current_view_data:
            return
        domain_key = self.combo_sim_domain.currentText()
        if not domain_key:
            QMessageBox.warning(
                self, "No Domains",
                "Please run Clustering first to define the ground truth!",
            )
            return
        labels_2d = self.session.current_view_data[domain_key]
        total_pixels = labels_2d.size
        self.sim_measured_mask = np.zeros(total_pixels, dtype=bool)
        self.sim_auto_steps = 0
        valid_indices = np.where(labels_2d.flatten() != -1)[0]
        n_start = self.spin_sim_points.value()
        start_indices = np.random.choice(valid_indices, n_start, replace=False)
        self.sim_measured_mask[start_indices] = True
        self.btn_sim_step.setEnabled(False)
        self.btn_sim_ff.setEnabled(False)
        self.execute_sim_worker()

    def run_sim_ff(self):
        self.sim_auto_steps = self.spin_sim_ff.value() - 1
        self.run_sim_step()

    def run_sim_step(self):
        if self.sim_next_idx is None or self.sim_next_idx == -1:
            return
        self.sim_measured_mask[self.sim_next_idx] = True
        self.btn_sim_step.setEnabled(False)
        self.btn_sim_ff.setEnabled(False)
        self.btn_sim_init.setEnabled(False)
        self.execute_sim_worker()

    def execute_sim_worker(self):
        domain_key = self.combo_sim_domain.currentText()
        algo = self.combo_sim_algo.currentText()
        labels_2d = self.session.current_view_data[domain_key]
        x_arr, y_arr = self.session.current_view_data["x"], self.session.current_view_data["y"]
        if self.sim_auto_steps > 0:
            self.session.set_status(
                1, f"Fast-Forwarding {algo} ({self.sim_auto_steps} steps remaining)..."
            )
        else:
            self.session.set_status(1, f"Simulating {algo}...")
        self.sim_worker = SimulateALWorker(
            x_arr, y_arr, labels_2d, algo, self.sim_measured_mask
        )
        self.sim_worker.setStackSize(32 * 1024 * 1024)
        self.sim_worker.progress.connect(self.session.set_status)
        self.sim_worker.finished.connect(self.on_sim_finish)
        self.sim_worker.error.connect(
            lambda e: self.session.set_status(0, f"Simulation Error: {e}")
        )
        self.sim_worker.start()

    def on_sim_finish(self, pred_map, uncert_map, next_idx):
        self.ax_sim_truth.clear()
        self.ax_sim_pred.clear()
        self.ax_sim_uncert.clear()
        x_arr, y_arr = self.session.current_view_data["x"], self.session.current_view_data["y"]
        X_grid, Y_grid = np.meshgrid(x_arr, y_arr)
        extent = (x_arr[0], x_arr[-1], y_arr[0], y_arr[-1])

        domain_key = self.combo_sim_domain.currentText()
        truth_map_1d = self.session.current_view_data[domain_key]
        truth_map_2d = truth_map_1d.reshape(X_grid.shape)

        self.ax_sim_truth.imshow(
            truth_map_2d, origin="lower", extent=extent, cmap="tab20",
            alpha=0.8, vmin=-0.5, vmax=19.5,
        )
        self.ax_sim_pred.imshow(
            pred_map, origin="lower", extent=extent, cmap="tab20",
            alpha=0.8, vmin=-0.5, vmax=19.5,
        )
        self.ax_sim_uncert.imshow(
            uncert_map, origin="lower", extent=extent, cmap="magma"
        )

        measured_x = X_grid.flatten()[self.sim_measured_mask]
        measured_y = Y_grid.flatten()[self.sim_measured_mask]
        for ax in [self.ax_sim_truth, self.ax_sim_pred, self.ax_sim_uncert]:
            ax.scatter(
                measured_x, measured_y, c="white", s=10, edgecolors="black",
                label="Measured Points",
            )
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")

        self.sim_next_idx = next_idx
        if next_idx != -1:
            nxt_x = X_grid.flatten()[next_idx]
            nxt_y = Y_grid.flatten()[next_idx]
            self.ax_sim_uncert.scatter(
                [nxt_x], [nxt_y], c="lime", s=80, marker="X",
                edgecolors="black", label="Next Scan Suggestion",
            )
            self.ax_sim_uncert.legend(loc="upper right", fontsize=8)

        total_measured = np.sum(self.sim_measured_mask)
        self.ax_sim_truth.set_title("Ground Truth Domain Map")
        self.ax_sim_pred.set_title(
            f"Simulation Phase Prediction\n(Trained on {total_measured} points)"
        )
        self.ax_sim_uncert.set_title(
            "Simulation Uncertainty Heatmap\n(Green X = Next Step)"
        )

        self.sim_canvas.figure.tight_layout()
        self.sim_canvas.draw_idle()

        if self.sim_auto_steps > 0:
            self.sim_auto_steps -= 1
            QTimer.singleShot(100, self.run_sim_step)
        else:
            self.btn_sim_init.setEnabled(True)
            self.btn_sim_step.setEnabled(True)
            self.btn_sim_ff.setEnabled(True)
            self.session.set_status(100, "Simulation step complete!")
