"""3D alignment controls plus azimuth/tilt result canvases."""
import numpy as np
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

from tensorspec.gui.components.ml_tabs.layout import split_panel
from tensorspec.gui.maestroai.maestro_fermi_viewer import FermiViewerWindow
from tensorspec.gui.maestroai.maestroai_alignment import (
    AzimuthalTwistWorker, CoupledAzimuthTiltWorker, NormalTiltWorker,
)
from tensorspec.gui.maestroai.maestroai_guides import AlignmentGuideDialog
from tensorspec.gui.maestroai.maestroai_viewers import AzimuthTemplateViewer, MplCanvas


class AlignmentPanel(QWidget):
    """3D alignment controls plus azimuth/tilt result canvases."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._build()
        self.session.workspace_changed.connect(self.refresh_workspace_refs)

    def _build(self):
        controls = QWidget()
        align_layout = QVBoxLayout(controls)

        self.btn_align_help = QPushButton("📐 Guide to Alignment (Azimuth & Tilt)")
        self.btn_align_help.setStyleSheet(
            "font-weight: bold; color: #bcbd22; padding: 6px; font-size: 14px;"
        )
        self.btn_align_help.clicked.connect(
            lambda: AlignmentGuideDialog(self).exec()
        )
        align_layout.addWidget(self.btn_align_help)

        align_ctrl_group = QGroupBox("Alignment Controls:")
        align_c_layout = QVBoxLayout(align_ctrl_group)

        self.combo_align_ref = QComboBox()
        align_c_layout.addWidget(QLabel("Select Reference 3D Fermi Map:"))
        align_c_layout.addWidget(self.combo_align_ref)

        self.btn_inspect_ref = QPushButton("Inspect Selected Reference Map")
        self.btn_inspect_ref.setStyleSheet(
            "font-weight: bold; color: #1f77b4; padding: 4px;"
        )
        self.btn_inspect_ref.clicked.connect(self.open_fermi_viewer)
        align_c_layout.addWidget(self.btn_inspect_ref)

        self.btn_inspect_azimuth = QPushButton("Visualize Azimuthal Cuts")
        self.btn_inspect_azimuth.setStyleSheet(
            "font-weight: bold; color: #9467bd; padding: 4px;"
        )
        self.btn_inspect_azimuth.clicked.connect(self.open_azimuth_viewer)
        align_c_layout.addWidget(self.btn_inspect_azimuth)

        self.combo_align_mode = QComboBox()
        self.combo_align_mode.addItems([
            "Azimuthal Twist (In-Plane)",
            "Surface Normal Tilt (Out-of-Plane)",
            "Coupled Azimuth & Deflection Tilt",
        ])
        align_c_layout.addWidget(QLabel("Alignment Mode:"))
        align_c_layout.addWidget(self.combo_align_mode)

        gamma_layout = QHBoxLayout()
        self.spin_gamma_s = QDoubleSpinBox()
        self.spin_gamma_s.setRange(-45, 45)
        self.spin_gamma_s.setDecimals(2)
        self.spin_gamma_d = QDoubleSpinBox()
        self.spin_gamma_d.setRange(-45, 45)
        self.spin_gamma_d.setDecimals(2)
        gamma_layout.addWidget(QLabel("Γ Slit (°):"))
        gamma_layout.addWidget(self.spin_gamma_s)
        gamma_layout.addWidget(QLabel("Γ Defl (°):"))
        gamma_layout.addWidget(self.spin_gamma_d)
        align_c_layout.addWidget(QLabel("Center of Rotation (Γ-Point Target):"))
        align_c_layout.addLayout(gamma_layout)

        self.btn_run_align = QPushButton("Run Global Alignment Search")
        self.btn_run_align.setStyleSheet(
            "font-weight: bold; color: #2ca02c; padding: 6px; font-size: 14px;"
        )
        self.btn_run_align.clicked.connect(self.run_alignment)

        self.align_canvas = MplCanvas(
            self, width=5, height=4, is_3d=False, orientation="horizontal_3"
        )
        self.ax_align_1, self.ax_align_2, self.ax_align_3 = self.align_canvas.axes

        align_layout.addWidget(align_ctrl_group)
        align_layout.addWidget(self.btn_run_align)
        align_layout.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split_panel(controls, self.align_canvas, sizes=(380, 420)))

    def refresh_workspace_refs(self):
        """Repopulate the reference Fermi-map combo from workspace keys."""
        self.combo_align_ref.blockSignals(True)
        self.combo_align_ref.clear()
        self.combo_align_ref.addItems(list(self.session.workspace.keys()))
        self.combo_align_ref.blockSignals(False)

    def run_alignment(self):
        if not self.session.current_view_data:
            QMessageBox.warning(
                self, "No Target Map",
                "Please activate an XY Scan in the workspace first.",
            )
            return
        ref_name = self.combo_align_ref.currentText()
        if not ref_name:
            QMessageBox.warning(
                self, "No Reference Map",
                "Please load a standard 3D Fermi map into the workspace.",
            )
            return
        mode = self.combo_align_mode.currentText()
        gamma_s_deg, gamma_d_deg = self.spin_gamma_s.value(), self.spin_gamma_d.value()
        ref_data = self.session.workspace[ref_name]
        gamma_s_px = int(np.argmin(np.abs(ref_data["angle"] - gamma_s_deg)))
        gamma_d_px = int(np.argmin(np.abs(ref_data["x"] - gamma_d_deg)))

        self.btn_run_align.setEnabled(False)
        self.session.set_status(1, f"Initializing {mode} Search...")

        if "Coupled" in mode:
            self.align_worker = CoupledAzimuthTiltWorker(
                self.session.current_view_data, ref_data, gamma_s_px, gamma_d_px
            )
        elif "Azimuth" in mode:
            self.align_worker = AzimuthalTwistWorker(
                self.session.current_view_data, ref_data, gamma_s_px, gamma_d_px
            )
        else:
            self.align_worker = NormalTiltWorker(
                self.session.current_view_data, ref_data, gamma_s_px, gamma_d_px
            )

        self.align_worker.progress.connect(self.session.set_status)
        self.align_worker.finished.connect(self.on_align_finish)
        self.align_worker.error.connect(
            lambda e: self.session.set_status(0, f"Alignment Error: {e}")
        )
        self.align_worker.start()

    def on_align_finish(self, map1, map2, map3, mode):
        self.align_canvas.figure.clf()
        self.ax_align_1, self.ax_align_2, self.ax_align_3 = (
            self.align_canvas.figure.subplots(1, 3)
        )
        x_arr, y_arr = self.session.current_view_data["x"], self.session.current_view_data["y"]
        nX, nY = len(x_arr), len(y_arr)
        px_x = (x_arr[-1] - x_arr[0]) / max(1, nX - 1) if nX > 1 else 0.1
        px_y = (y_arr[-1] - y_arr[0]) / max(1, nY - 1) if nY > 1 else 0.1
        extent = (
            x_arr[0] - px_x / 2, x_arr[-1] + px_x / 2,
            y_arr[0] - px_y / 2, y_arr[-1] + px_y / 2,
        )

        if "Coupled" in mode:
            title1, key1, cmap1 = r"Best Azimuth Rotation ($\phi^\circ$)", "domains_Align_Azimuth_Coupled", "hsv"
            title2, key2, cmap2 = r"Deflection Tilt Shift (pixels)", "domains_Align_Defl_Coupled", "PiYG"
            title3, key3, cmap3 = "Match Quality Score", "domains_Align_Score_Coupled", "magma"
        elif "Azimuth" in mode:
            title1, key1, cmap1 = r"Best Azimuth Rotation ($\phi^\circ$)", "domains_Align_Azimuth", "hsv"
            title2, key2, cmap2 = r"Momentum Slit Shift (pixels)", "domains_Align_Slit_Shift", "PiYG"
            title3, key3, cmap3 = "Match Quality Score", "domains_Align_Score", "magma"
        else:
            title1, key1, cmap1 = r"Deflection Tilt Shift (pixels)", "domains_Align_Defl_Tilt", "PiYG"
            title2, key2, cmap2 = r"Momentum Slit Shift (pixels)", "domains_Align_Slit_Tilt", "PiYG"
            title3, key3, cmap3 = "Match Quality Score", "domains_Align_Score_Tilt", "magma"

        for ax, data_map, title, cmap in zip(
            [self.ax_align_1, self.ax_align_2, self.ax_align_3],
            [map1, map2, map3],
            [title1, title2, title3],
            [cmap1, cmap2, cmap3],
        ):
            im = ax.imshow(data_map, origin="lower", extent=extent, cmap=cmap, aspect="auto")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            self.align_canvas.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        self.align_canvas.figure.tight_layout()
        self.align_canvas.draw_idle()
        self.session.current_view_data[key1] = map1.flatten()
        self.session.current_view_data[key2] = map2.flatten()
        self.session.current_view_data[key3] = map3.flatten()

        if self.session.viewer is not None:
            for k in [key1, key2, key3]:
                self.session.viewer.add_overlay_mode(k)

        self.btn_run_align.setEnabled(True)
        self.session.set_status(100, f"{mode} successfully computed!")

    def open_fermi_viewer(self):
        ref_name = self.combo_align_ref.currentText()
        if not ref_name:
            QMessageBox.warning(
                self, "No Reference Map",
                "Please load a standard 3D Fermi map into the workspace.",
            )
            return
        ref_data = self.session.workspace[ref_name]
        if len(ref_data["value"].shape) != 4 or ref_data["value"].shape[2] != 1:
            QMessageBox.warning(
                self, "Invalid Format",
                "The selected item is an XY Spatial Scan, not a 3D Deflection Map!",
            )
            return
        self.fermi_dialog = FermiViewerWindow(ref_data, self)
        self.fermi_dialog.show()

    def open_azimuth_viewer(self):
        ref_name = self.combo_align_ref.currentText()
        if not ref_name:
            QMessageBox.warning(
                self, "No Reference Map",
                "Please load a standard 3D Fermi map into the workspace.",
            )
            return
        gamma_s_deg = self.spin_gamma_s.value()
        gamma_d_deg = self.spin_gamma_d.value()
        self.az_viewer = AzimuthTemplateViewer(
            self.session.workspace[ref_name], gamma_s_deg, gamma_d_deg, self
        )
        self.az_viewer.show()
