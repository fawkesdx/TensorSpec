"""Active-learning controls plus its prediction/uncertainty canvas."""
import matplotlib.patches as patches
from PySide6.QtWidgets import (
    QComboBox, QGroupBox, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from tensorspec.gui.components.ml_tabs.layout import split_panel
from tensorspec.gui.maestroai.maestroai_active_learning import ActiveLearningWorker
from tensorspec.gui.maestroai.maestroai_guides import ActiveLearningGuideDialog
from tensorspec.gui.maestroai.maestroai_viewers import MplCanvas


class ActiveLearningPanel(QWidget):
    """Active-learning controls plus its prediction/uncertainty canvas."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._build()
        self.session.domains_changed.connect(self.set_domain_keys)

    def _build(self):
        controls = QWidget()
        al_layout = QVBoxLayout(controls)

        self.btn_al_help = QPushButton("🧭 What is Active Learning? Click Here")
        self.btn_al_help.setStyleSheet(
            "font-weight: bold; color: #8c564b; padding: 6px; font-size: 14px;"
        )
        self.btn_al_help.clicked.connect(self.show_al_guide)
        al_layout.addWidget(self.btn_al_help)

        al_ctrl_group = QGroupBox("Active Learning Controls:")
        al_c_layout = QVBoxLayout(al_ctrl_group)

        self.combo_gp_domain = QComboBox()
        al_c_layout.addWidget(QLabel("Target Clustered Domain:"))
        al_c_layout.addWidget(self.combo_gp_domain)

        self.combo_al_algo = QComboBox()
        self.combo_al_algo.addItems([
            "Bayesian Network (GPU)", "Deep Ensembles (GPU)",
            "Evidential Deep Learning (GPU)",
            "Gaussian Process (CPU)", "Random Forest (CPU)",
        ])
        al_c_layout.addWidget(QLabel("Steering Algorithm:"))
        al_c_layout.addWidget(self.combo_al_algo)

        self.btn_run_al = QPushButton("Calculate Next Scan Suggestions")
        self.btn_run_al.setStyleSheet(
            "font-weight: bold; color: #8c564b; padding: 6px; font-size: 14px;"
        )
        self.btn_run_al.clicked.connect(self.run_active_learning)

        self.al_canvas = MplCanvas(self, width=5, height=4, is_3d=False, orientation="vertical")
        self.ax_al_pred, self.ax_al_uncert = self.al_canvas.axes

        al_layout.addWidget(al_ctrl_group)
        al_layout.addWidget(self.btn_run_al)
        al_layout.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split_panel(controls, self.al_canvas, sizes=(260, 540)))

    def set_domain_keys(self, keys):
        """Repopulate the domain combo; replaces the write from activate_data."""
        self.combo_gp_domain.blockSignals(True)
        self.combo_gp_domain.clear()
        self.combo_gp_domain.addItems(keys)
        self.combo_gp_domain.blockSignals(False)

    def show_al_guide(self):
        ActiveLearningGuideDialog(self).exec()

    def run_active_learning(self):
        if not self.session.current_view_data:
            return
        domain_key = self.combo_gp_domain.currentText()
        if not domain_key:
            QMessageBox.warning(
                self, "No Domains",
                "Please run Clustering first to define the phases!",
            )
            return
        algo = self.combo_al_algo.currentText()
        data = self.session.current_view_data
        labels_2d = data[domain_key]
        x_arr, y_arr = data["x"], data["y"]
        self.btn_run_al.setEnabled(False)
        self.session.set_status(1, f"Initializing {algo}...")
        self.al_worker = ActiveLearningWorker(x_arr, y_arr, labels_2d, algo)
        self.al_worker.setStackSize(32 * 1024 * 1024)
        self.al_worker.progress.connect(self.session.set_status)
        self.al_worker.finished.connect(self.on_al_finish)
        self.al_worker.error.connect(
            lambda e: self.session.set_status(0, f"Active Learning Error: {e}")
        )
        self.al_worker.start()

    def on_al_finish(self, pred_map, uncert_map, new_x, new_y, bounds):
        self.ax_al_pred.clear()
        self.ax_al_uncert.clear()
        extent = (new_x[0], new_x[-1], new_y[0], new_y[-1])
        self.ax_al_pred.imshow(
            pred_map, origin="lower", extent=extent, cmap="tab20", alpha=0.8
        )
        self.ax_al_pred.set_title("Extended Phase Prediction")
        self.ax_al_uncert.imshow(uncert_map, origin="lower", extent=extent, cmap="magma")
        self.ax_al_uncert.set_title("Uncertainty Heatmap\n(Bright = Scan Next)")
        for ax in [self.ax_al_pred, self.ax_al_uncert]:
            rect = patches.Rectangle(
                (bounds[0], bounds[2]),
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                linewidth=2, edgecolor="white", facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
        self.al_canvas.draw_idle()
        self.btn_run_al.setEnabled(True)
        self.session.set_status(100, "Steering Suggestions Computed!")
