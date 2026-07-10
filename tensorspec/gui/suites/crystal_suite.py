# File: tensorspec/gui/suites/crystal_suite.py
import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QFileDialog, QLabel, QSpinBox, QDoubleSpinBox, 
                               QComboBox, QColorDialog, QTabWidget, QCheckBox, 
                               QGroupBox, QGridLayout, QSlider, QSplitter, 
                               QInputDialog, QMessageBox, QFrame, QScrollArea)
from PySide6.QtCore import Qt
from pymatgen.core import Structure

# Clean modular imports from our established architecture
from tensorspec.core.crystallography import CrystalEngine
import platform

# --- SMART ENGINE ROUTER ---
is_legacy_mac = (platform.system() == "Darwin" and platform.machine() == "x86_64")

if is_legacy_mac:
    print("🔌 Crystal Suite: Routing to crash-proof Matplotlib backend.")
    from tensorspec.plotting.backends.matplotlib_engine import MatplotlibCrystalBackend as ActiveCrystalBackend
else:
    print("🔌 Crystal Suite: Routing to high-performance PyVista backend.")
    from tensorspec.plotting.backends.pyvista_engine import PyVistaCrystalBackend as ActiveCrystalBackend


# Default CPK Colors for dynamic UI generation
CPK_COLORS = {
    "H": "#FFFFFF", "C": "#333333", "N": "#2233FF", "O": "#FF2200",
    "Te": "#FF8C00", "Fe": "#E06633", "Ta": "#B041FF", "Ir": "#0080FF",
    "Nb": "#7A378B", "W": "#4682B4", "Mo": "#5F9EA0", "Bonds": "#d3d3d3"
}


class StackLayerRow(QFrame):
    """Mini UI Row Controller for individual 2D sheets in the Heterostructure Tab."""
    def __init__(self, name: str, struct: Structure, default_z: float):
        super().__init__()
        self.struct = struct
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background-color: #f0f0f0; border: 1px solid #cccccc; border-radius: 5px; padding: 4px; }")
        
        main_layout = QVBoxLayout(self) # Stacked vertically now!
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # --- Top Row: Name and Supercell ---
        top_row = QHBoxLayout()
        self.lbl_name = QLabel(f"<b>{name}</b>")
        top_row.addWidget(self.lbl_name)
        
        top_row.addWidget(QLabel("SC:"))
        self.spin_x = QSpinBox(); self.spin_x.setRange(1, 50); self.spin_x.setValue(1)
        self.spin_y = QSpinBox(); self.spin_y.setRange(1, 50); self.spin_y.setValue(1)
        top_row.addWidget(self.spin_x); top_row.addWidget(self.spin_y)
        main_layout.addLayout(top_row)
        
        # --- Bottom Row: Z-shift, Twist, and Action Buttons ---
        bot_row = QHBoxLayout()
        bot_row.addWidget(QLabel("z (Å):"))
        self.spin_z = QDoubleSpinBox(); self.spin_z.setRange(-100.0, 100.0); self.spin_z.setSingleStep(0.5); self.spin_z.setValue(default_z)
        bot_row.addWidget(self.spin_z)
        
        bot_row.addWidget(QLabel("θ (°):"))
        self.spin_twist = QDoubleSpinBox(); self.spin_twist.setRange(-360.0, 360.0); self.spin_twist.setSingleStep(1.0)
        bot_row.addWidget(self.spin_twist)
        
        self.btn_up = QPushButton("▲"); self.btn_up.setFixedWidth(25)
        self.btn_down = QPushButton("▼"); self.btn_down.setFixedWidth(25)
        self.btn_save_tpl = QPushButton("💾"); self.btn_save_tpl.setStyleSheet("background-color: #5cb85c; color: white;"); self.btn_save_tpl.setFixedWidth(25)
        self.btn_delete = QPushButton("X"); self.btn_delete.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;"); self.btn_delete.setFixedWidth(28)
        
        bot_row.addWidget(self.btn_up); bot_row.addWidget(self.btn_down)
        bot_row.addWidget(self.btn_save_tpl); bot_row.addWidget(self.btn_delete)
        main_layout.addLayout(bot_row)

    def get_layer_dict(self) -> dict:
        return {
            "struct": self.struct, "sc_x": self.spin_x.value(),
            "sc_y": self.spin_y.value(), "z_shift": self.spin_z.value(), "twist": self.spin_twist.value()
        }


class CrystalViewerSuite(QWidget):
    """
    Main Crystal Suite UI Controller.
    Links user actions to core crystallography math and 3D PyVista rendering.
    """
    def __init__(self, workspace_manager=None, parent=None):
        super().__init__(parent)
        self.workspace = workspace_manager
        
        self.current_structure = None
        self.active_supercell = None
        self.active_colors = CPK_COLORS.copy()
        self.stack_layer_rows = []
        self.erased_atoms = set()
        self.erased_bonds = set()
        
        self.init_layout()

    def init_layout(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 1. Left Control Panel (Tabs)
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(340)
        splitter.addWidget(self.tabs)
        
        self.init_tab_view_edit()
        self.init_tab_cdw()
        self.init_tab_heterostructure()
        self.init_tab_bz()

        # 2. Right 3D Viewport (Stacked Backends)
        from PySide6.QtWidgets import QStackedWidget
        from tensorspec.plotting.backends.matplotlib_engine import MatplotlibCrystalBackend
        from tensorspec.plotting.backends.pyvista_engine import PyVistaCrystalBackend
        
        self.viewer_stack = QStackedWidget()
        self.renderer_cpu = MatplotlibCrystalBackend(parent=self)
        self.renderer_gpu = PyVistaCrystalBackend(parent=self)
        
        self.viewer_stack.addWidget(self.renderer_cpu.plotter) # Index 0: Matplotlib
        self.viewer_stack.addWidget(self.renderer_gpu.plotter) # Index 1: PyVista
        splitter.addWidget(self.viewer_stack)
        
        # Auto-detect default based on hardware, but allow manual switching
        if is_legacy_mac:
            self.viewer_stack.setCurrentIndex(0)
            self.renderer = self.renderer_cpu
        else:
            self.viewer_stack.setCurrentIndex(1)
            self.renderer = self.renderer_gpu
        
        splitter.setSizes([380, 820])

    # ================= TAB 1: VIEW & EDIT =================
    def init_tab_view_edit(self):
        tab = QWidget()
        main_tab_layout = QVBoxLayout(tab)
        main_tab_layout.setContentsMargins(0, 0, 0, 0)
        
        # Wrap everything in a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        container = QWidget()
        layout = QVBoxLayout(container) # ALL your existing UI groups go into this layout now
        
        self.btn_load = QPushButton("📂 Load CIF File")
        self.btn_load.clicked.connect(self.handle_load_cif)
        layout.addWidget(self.btn_load)
        
        self.lbl_file = QLabel("No file loaded"); self.lbl_file.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_file)
        
        # --- RESTORED: Space Group Label ---
        self.lbl_sym = QLabel("Space Group: N/A")
        self.lbl_sym.setStyleSheet("color: #aaaaaa; font-style: italic;")
        layout.addWidget(self.lbl_sym)

        # --- RESTORED: Manual Backend Switcher ---
        layout.addWidget(QLabel("Graphics Backend:"))
        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["CPU Safe Mode (Matplotlib)", "GPU Fast Mode (PyVista)"])
        self.combo_backend.setCurrentIndex(0 if is_legacy_mac else 1)
        self.combo_backend.currentIndexChanged.connect(self.switch_backend)
        layout.addWidget(self.combo_backend)

        # 1. Geometry & Scaling
        from PySide6.QtWidgets import QRadioButton, QButtonGroup
        group_geom = QGroupBox("Geometry & Scaling")
        geom_layout = QVBoxLayout(group_geom)
        
        self.radio_conv = QRadioButton("Conventional Basis")
        self.radio_prim = QRadioButton("Primitive Basis")
        self.radio_conv.setChecked(True)
        self.radio_conv.toggled.connect(self.handle_draw)
        basis_layout = QHBoxLayout()
        basis_layout.addWidget(self.radio_conv); basis_layout.addWidget(self.radio_prim)
        geom_layout.addLayout(basis_layout)
        
        sc_layout = QHBoxLayout()
        sc_layout.addWidget(QLabel("Supercell (X,Y,Z):"))
        self.spin_scx = QSpinBox(); self.spin_scx.setValue(1); self.spin_scx.setRange(1, 20)
        self.spin_scy = QSpinBox(); self.spin_scy.setValue(1); self.spin_scy.setRange(1, 20)
        self.spin_scz = QSpinBox(); self.spin_scz.setValue(1); self.spin_scz.setRange(1, 20)
        sc_layout.addWidget(self.spin_scx); sc_layout.addWidget(self.spin_scy); sc_layout.addWidget(self.spin_scz)
        geom_layout.addLayout(sc_layout)

        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Atom Radius Scale:"))
        self.spin_radius = QDoubleSpinBox(); self.spin_radius.setRange(0.1, 3.0); self.spin_radius.setValue(0.5); self.spin_radius.setSingleStep(0.1)
        scale_layout.addWidget(self.spin_radius)
        scale_layout.addWidget(QLabel("Bond Thick:"))
        self.spin_bond_thick = QDoubleSpinBox(); self.spin_bond_thick.setRange(0.01, 1.0); self.spin_bond_thick.setValue(0.10); self.spin_bond_thick.setSingleStep(0.02)
        scale_layout.addWidget(self.spin_bond_thick)
        geom_layout.addLayout(scale_layout)
        layout.addWidget(group_geom)

        # 2. Styles & Rendering
        group_render = QGroupBox("Styles & Rendering")
        render_layout = QVBoxLayout(group_render)
        
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("Connections:"))
        self.combo_style = QComboBox()
        self.combo_style.addItems(["Bonds (Sticks)", "Polyhedra (Planes)", "None"])
        self.combo_style.currentIndexChanged.connect(self.refresh_render)
        style_layout.addWidget(self.combo_style)
        style_layout.addWidget(QLabel("Bond Thresh:"))
        self.spin_bond_thresh = QDoubleSpinBox(); self.spin_bond_thresh.setRange(0.1, 5.0); self.spin_bond_thresh.setValue(1.15)
        style_layout.addWidget(self.spin_bond_thresh)
        render_layout.addLayout(style_layout)

        chk_layout = QHBoxLayout()
        self.chk_shiny = QCheckBox("PBR Shiny"); self.chk_shiny.stateChanged.connect(self.refresh_render)
        self.chk_axes = QCheckBox("Show Axes"); self.chk_axes.setChecked(True); self.chk_axes.stateChanged.connect(self.refresh_render)
        self.chk_show_conventional = QCheckBox("Show Conv. Box"); self.chk_show_conventional.setChecked(True); self.chk_show_conventional.stateChanged.connect(self.refresh_render)
        self.chk_show_primitive = QCheckBox("Show Prim. Box"); self.chk_show_primitive.stateChanged.connect(self.refresh_render)
        chk_layout.addWidget(self.chk_shiny); chk_layout.addWidget(self.chk_axes); chk_layout.addWidget(self.chk_show_conventional); chk_layout.addWidget(self.chk_show_primitive)
        render_layout.addLayout(chk_layout)
        layout.addWidget(group_render)

        # 3. Interactive Eraser
        self.chk_edit_mode = QCheckBox("Enable Interactive Eraser Brush")
        self.chk_edit_mode.setStyleSheet("color: #d9534f; font-weight: bold;")
        self.chk_edit_mode.stateChanged.connect(self.toggle_edit_mode)
        layout.addWidget(self.chk_edit_mode)

        # 4. Dynamic Color Pickers
        self.group_colors = QGroupBox("Dynamic Element Colors")
        self.colors_layout = QGridLayout(self.group_colors)
        layout.addWidget(self.group_colors)

        # 5. Camera & Projection
        self.group_camera = QGroupBox("Camera & Projection")
        self.cam_layout = QVBoxLayout(self.group_camera)
        
        self.combo_projection = QComboBox()
        self.combo_projection.addItems(["Perspective Projection", "Orthogonal Projection"])
        self.combo_projection.currentIndexChanged.connect(self.handle_camera_update)
        self.cam_layout.addWidget(self.combo_projection)
        
        quick_view_layout = QHBoxLayout()
        quick_view_layout.addWidget(QLabel("Quick View:"))
        for axis, label in [('x', '+a'), ('y', '+b'), ('z', '+c'), ('iso', '111')]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, a=axis: self.renderer.set_camera_preset(a) if hasattr(self.renderer, 'set_camera_preset') else None)
            quick_view_layout.addWidget(btn)
        self.cam_layout.addLayout(quick_view_layout)
        
        azel_layout = QHBoxLayout()
        azel_layout.addWidget(QLabel("Azimuth:")); self.spin_azimuth = QDoubleSpinBox(); self.spin_azimuth.setRange(-360.0, 360.0); self.spin_azimuth.setSingleStep(5.0); azel_layout.addWidget(self.spin_azimuth)
        azel_layout.addWidget(QLabel("Elevation:")); self.spin_elevation = QDoubleSpinBox(); self.spin_elevation.setRange(-90.0, 90.0); self.spin_elevation.setSingleStep(5.0); azel_layout.addWidget(self.spin_elevation)
        self.cam_layout.addLayout(azel_layout)
        layout.addWidget(self.group_camera)

        # 6. Crystallography Tools
        self.group_cryst = QGroupBox("Crystallography Tools")
        self.cryst_layout = QVBoxLayout(self.group_cryst)
        
        hkl_layout = QHBoxLayout()
        hkl_layout.addWidget(QLabel("View [h k l]:"))
        self.spin_h = QDoubleSpinBox(); self.spin_h.setRange(-10, 10); self.spin_k = QDoubleSpinBox(); self.spin_k.setRange(-10, 10); self.spin_l = QDoubleSpinBox(); self.spin_l.setRange(-10, 10)
        hkl_layout.addWidget(self.spin_h); hkl_layout.addWidget(self.spin_k); hkl_layout.addWidget(self.spin_l)
        self.btn_align_hkl = QPushButton("Align"); self.btn_align_hkl.clicked.connect(self.align_to_hkl)
        hkl_layout.addWidget(self.btn_align_hkl)
        self.cryst_layout.addLayout(hkl_layout)

        plane_layout = QHBoxLayout()
        self.chk_show_plane = QCheckBox("Show Cut Plane"); plane_layout.addWidget(self.chk_show_plane)
        self.combo_plane_color = QComboBox(); self.combo_plane_color.addItems(["cyan", "magenta", "yellow", "white", "gray"]); plane_layout.addWidget(self.combo_plane_color)
        self.combo_plane_orient = QComboBox(); self.combo_plane_orient.addItems(["Lock to Camera", "Lock to [h k l]"]); plane_layout.addWidget(self.combo_plane_orient)
        self.cryst_layout.addLayout(plane_layout)
        
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Depth:")); self.slider_plane_depth = QSlider(Qt.Horizontal); self.slider_plane_depth.setRange(-100, 100); self.slider_plane_depth.setValue(0)
        depth_layout.addWidget(self.slider_plane_depth)
        self.cryst_layout.addLayout(depth_layout)
        layout.addWidget(self.group_cryst)

        # 7. Export Elements
        self.group_export = QGroupBox("Export Elements")
        exp_layout = QVBoxLayout(self.group_export)
        chk_layout = QHBoxLayout()
        self.chk_exp_atoms = QCheckBox("Atoms/Bonds"); self.chk_exp_atoms.setChecked(True)
        self.chk_exp_cell = QCheckBox("Unit Cell"); self.chk_exp_cell.setChecked(True)
        self.chk_exp_bz = QCheckBox("Brillouin Zone"); self.chk_exp_bz.setChecked(True)
        chk_layout.addWidget(self.chk_exp_atoms); chk_layout.addWidget(self.chk_exp_cell); chk_layout.addWidget(self.chk_exp_bz)
        exp_layout.addLayout(chk_layout)
        
        btn_layout = QHBoxLayout()
        self.btn_export_max = QPushButton("Export 3ds Max"); self.btn_export_max.setStyleSheet("background-color: #0F6A8B; color: white; font-weight: bold;"); self.btn_export_max.clicked.connect(self.export_scripts)
        self.btn_export_blend = QPushButton("Export Blender"); self.btn_export_blend.setStyleSheet("background-color: #E87D0D; color: white; font-weight: bold;"); self.btn_export_blend.clicked.connect(self.export_scripts)
        btn_layout.addWidget(self.btn_export_max); btn_layout.addWidget(self.btn_export_blend)
        exp_layout.addLayout(btn_layout)
        layout.addWidget(self.group_export)

        # 8. Action Buttons
        self.btn_draw = QPushButton("🎨 Render Structure")
        self.btn_draw.setStyleSheet("background-color: #2b5c8f; color: white; font-weight: bold; padding: 6px;")
        self.btn_draw.clicked.connect(self.handle_draw)
        layout.addWidget(self.btn_draw)

        self.btn_save = QPushButton("📸 Save High-Res Image")
        self.btn_save.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 6px;")
        self.btn_save.clicked.connect(self.save_image)
        layout.addWidget(self.btn_save)

        layout.addStretch()
        # Cap off the scroll container and add it to the tab
        scroll.setWidget(container)
        main_tab_layout.addWidget(scroll)
        self.tabs.addTab(tab, "1. View & Edit")

    # --- RESTORED LOGIC METHODS ---
    def switch_backend(self, index):
        """Swaps the active rendering engine with a safety warning for old hardware."""
        # 1. Guardrail for older hardware
        if index == 1 and is_legacy_mac:
            reply = QMessageBox.warning(
                self, "Hardware Warning", 
                "Your older MacBook (Intel/OCLP) does not support the modern Metal drivers required for PyVista.\n\nSwitching to GPU Fast Mode will likely cause a segmentation fault and crash the application instantly.\n\nDo you still want to proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                # Silently revert the dropdown back to Matplotlib without triggering an infinite loop
                self.combo_backend.blockSignals(True)
                self.combo_backend.setCurrentIndex(0)
                self.combo_backend.blockSignals(False)
                return

        # 2. Proceed with the switch if it's safe (or if you forced it!)
        self.viewer_stack.setCurrentIndex(index)
        self.renderer = self.renderer_cpu if index == 0 else self.renderer_gpu
        self.handle_draw()

    def handle_camera_update(self):
        """Passes camera alignment requests to the active rendering backend."""
        if self.combo_projection.currentIndex() == 1:
            if hasattr(self.renderer.plotter, 'camera'):
                self.renderer.plotter.camera.enable_parallel_projection()
        else:
            if hasattr(self.renderer.plotter, 'camera'):
                self.renderer.plotter.camera.disable_parallel_projection()
        
        if hasattr(self.renderer, 'plotter') and hasattr(self.renderer.plotter, 'render'):
            self.renderer.plotter.render()
        elif hasattr(self.renderer, 'canvas'):
            self.renderer.canvas.draw_idle()

    def toggle_edit_mode(self, *args):
        is_erasing = self.chk_edit_mode.isChecked()
        if hasattr(self.renderer, 'plotter'):
            if is_erasing:
                if hasattr(self.renderer.plotter, 'enable_trackball_actor_style'):
                    self.renderer.plotter.enable_trackball_actor_style()
                QMessageBox.information(self, "Eraser Mode", "Left-click and drag to delete atoms/bonds. Right-click to exit.")
            else:
                if hasattr(self.renderer.plotter, 'enable_trackball_style'):
                    self.renderer.plotter.enable_trackball_style()

    def save_image(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
        if not fname: return
        
        if hasattr(self.renderer, 'plotter') and hasattr(self.renderer.plotter, 'screenshot'):
            self.renderer.plotter.screenshot(fname, transparent_background=True, window_size=[3840, 2160])
        elif hasattr(self.renderer, 'figure'):
            self.renderer.figure.savefig(fname, dpi=600, bbox_inches='tight', transparent=True)
            
        QMessageBox.information(self, "Saved", f"High resolution image saved to:\n{fname}")

    def export_scripts(self):
        if not getattr(self, 'active_supercell', None):
            QMessageBox.warning(self, "Export Error", "Please draw a structure first!")
            return
            
        sender = self.sender()
        software = "3ds Max" if sender == self.btn_export_max else "Blender"
        file_path, _ = QFileDialog.getSaveFileName(self, f"Export {software} Script", "", "Python Files (*.py)")
        
        if not file_path: return

        import numpy as np
        from tensorspec.core.io.exporters import SceneExporter
        from tensorspec.core.crystallography import CrystalEngine

        atoms_data, bonds_data, lattice_data = [], [], []
        bz_solid_data = None

        if self.chk_exp_atoms.isChecked():
            scale_mod = self.spin_radius.value()
            for i, site in enumerate(self.active_supercell):
                if i in self.erased_atoms: continue
                radius = float((site.specie.atomic_radius if site.specie.atomic_radius else 1.0) * scale_mod)
                color = self.active_colors.get(site.specie.symbol, "#008080")
                atoms_data.append((float(site.coords[0]), float(site.coords[1]), float(site.coords[2]), radius, color))

        if self.chk_exp_cell.isChecked() and getattr(self, 'current_structure', None):
            def get_edges(matrix, color):
                a, b, c = matrix[0], matrix[1], matrix[2]
                v = np.zeros((8, 3))
                v[1], v[2], v[3] = a, b, c; v[4], v[5], v[6], v[7] = a + b, a + c, b + c, a + b + c
                edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
                for p1, p2 in edges:
                    lattice_data.append((float(v[p1][0]), float(v[p1][1]), float(v[p1][2]), float(v[p2][0]), float(v[p2][1]), float(v[p2][2]), color))

            if self.chk_show_conventional.isChecked():
                get_edges(self.current_structure.lattice.matrix, "#FF5733")

        if software == "3ds Max":
            SceneExporter.export_3dsmax(file_path, atoms_data, bonds_data, lattice_data, bz_solid_data)
        else:
            SceneExporter.export_blender(file_path, atoms_data, bonds_data, lattice_data, bz_solid_data)
            
        QMessageBox.information(self, "Export Complete", f"Successfully saved {software} script!")

    # ================= TAB 2: CDW MODULATOR =================
    def init_tab_cdw(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.chk_cdw = QCheckBox("Enable CDW Modulation")
        self.chk_cdw.setStyleSheet("font-weight: bold; color: #2b5c8f;")
        self.chk_cdw.stateChanged.connect(self.refresh_render)
        layout.addWidget(self.chk_cdw)
        
        layout.addWidget(QLabel("Target Element:"))
        self.combo_cdw_el = QComboBox(); self.combo_cdw_el.addItem("All Elements")
        layout.addWidget(self.combo_cdw_el)

        group_q = QGroupBox("Wavevector q (rlu)")
        q_layout = QVBoxLayout(group_q)
        self.spin_qx = QDoubleSpinBox(); self.spin_qx.setRange(-5.0, 5.0); self.spin_qx.setSingleStep(0.05)
        self.spin_qy = QDoubleSpinBox(); self.spin_qy.setRange(-5.0, 5.0); self.spin_qy.setSingleStep(0.05)
        self.spin_qz = QDoubleSpinBox(); self.spin_qz.setRange(-5.0, 5.0); self.spin_qz.setSingleStep(0.05)
        for spin, lbl in [(self.spin_qx, "q_a:"), (self.spin_qy, "q_b:"), (self.spin_qz, "q_c:")]:
            row = QHBoxLayout(); row.addWidget(QLabel(lbl)); row.addWidget(spin); q_layout.addLayout(row)
            spin.valueChanged.connect(self.refresh_render)
        layout.addWidget(group_q)

        group_amp = QGroupBox("Amplitude A (Å)")
        amp_layout = QVBoxLayout(group_amp)
        self.spin_ax = QDoubleSpinBox(); self.spin_ax.setRange(-2.0, 2.0); self.spin_ax.setSingleStep(0.02)
        self.spin_ay = QDoubleSpinBox(); self.spin_ay.setRange(-2.0, 2.0); self.spin_ay.setSingleStep(0.02)
        self.spin_az = QDoubleSpinBox(); self.spin_az.setRange(-2.0, 2.0); self.spin_az.setSingleStep(0.02)
        for spin, lbl in [(self.spin_ax, "Δx:"), (self.spin_ay, "Δy:"), (self.spin_az, "Δz:")]:
            row = QHBoxLayout(); row.addWidget(QLabel(lbl)); row.addWidget(spin); amp_layout.addLayout(row)
            spin.valueChanged.connect(self.refresh_render)
        layout.addWidget(group_amp)

        # --- RESTORED: Phase Adjuster ---
        layout.addWidget(QLabel("Phase Shift φ (degrees):"))
        self.spin_cdw_phase = QDoubleSpinBox()
        self.spin_cdw_phase.setRange(0.0, 360.0); self.spin_cdw_phase.setSingleStep(15.0)
        self.spin_cdw_phase.valueChanged.connect(self.refresh_render)
        layout.addWidget(self.spin_cdw_phase)

        layout.addStretch()
        self.tabs.addTab(tab, "2. CDW Modulator")

    # ================= TAB 3: STACK & TWIST =================
    def init_tab_heterostructure(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        tpl_group = QGroupBox("2D Material Templates")
        tpl_layout = QVBoxLayout(tpl_group)
        self.combo_tpl = QComboBox()
        self.combo_tpl.addItems(["Graphene (Monolayer)", "Graphene (AB Bilayer)", "TaIrTe4 (1T')", "MoS2", "h-BN"])
        tpl_layout.addWidget(self.combo_tpl)
        
        btn_add_tpl = QPushButton("➕ Add Template Sheet")
        btn_add_tpl.clicked.connect(self.handle_add_template)
        tpl_layout.addWidget(btn_add_tpl)
        layout.addWidget(tpl_group)

        # --- RESTORED: Bulk Exfoliator ---
        exf_group = QGroupBox("Bulk Exfoliator")
        exf_layout = QVBoxLayout(exf_group)

        exf_layout.addWidget(QLabel("Cleavage / Cut Engine Mode:"))
        self.combo_exfoliate_mode = QComboBox()
        self.combo_exfoliate_mode.addItems([
            "Automatic van der Waals Gap Detection",
            "Manual [h k l] Miller Index Cleavage"
        ])
        exf_layout.addWidget(self.combo_exfoliate_mode)

        self.btn_extract_bulk = QPushButton("✂️ Extract Monolayer from Bulk CIF")
        self.btn_extract_bulk.setStyleSheet("background-color: #f0ad4e; color: black; font-weight: bold; padding: 5px;")
        self.btn_extract_bulk.clicked.connect(self.extract_monolayer_from_bulk)
        exf_layout.addWidget(self.btn_extract_bulk)
        layout.addWidget(exf_group)

        # Dynamic Scroll Area for Layer Controls
        self.scroll_area = QScrollArea(); self.scroll_area.setWidgetResizable(True)
        self.scroll_container = QWidget(); self.scroll_layout = QVBoxLayout(self.scroll_container)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.scroll_container)
        layout.addWidget(self.scroll_area)

        btn_draw_stack = QPushButton("🧱 Render Heterostructure Stack")
        btn_draw_stack.setStyleSheet("background-color: #2b5c8f; color: white; font-weight: bold; padding: 6px;")
        btn_draw_stack.clicked.connect(self.handle_draw_stack)
        layout.addWidget(btn_draw_stack)

        self.btn_moire = QPushButton("🌀 Calculate Moiré Superlattice")
        self.btn_moire.setStyleSheet("background-color: #8A2BE2; color: white; font-weight: bold; padding: 6px;")
        self.btn_moire.clicked.connect(self.handle_moire)
        layout.addWidget(self.btn_moire)

        self.lbl_moire = QLabel("Status: Waiting for stack...")
        self.lbl_moire.setStyleSheet("background-color: #1e1e24; color: white; padding: 4px;")
        layout.addWidget(self.lbl_moire)

        self.tabs.addTab(tab, "3. Stack & Twist")

    # ================= TAB 4: BRILLOUIN ZONE =================
    def init_tab_bz(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("<b>Reciprocal Space (Wigner-Seitz Cell)</b>"))
        
        btn_bz = QPushButton("🛑 Render 1st Brillouin Zone")
        btn_bz.setStyleSheet("background-color: #8A2BE2; color: white; font-weight: bold; padding: 6px;")
        btn_bz.clicked.connect(self.handle_draw_bz)
        layout.addWidget(btn_bz)
        
        layout.addStretch()
        self.tabs.addTab(tab, "4. Brillouin Zone")

    # ================= LOGIC AND EVENT CONTROLLERS =================
    def align_to_hkl(self):
        """Calculates the physical Cartesian vector from Miller Indices and snaps the camera."""
        if not getattr(self, 'current_structure', None): return
        h, k, l = self.spin_h.value(), self.spin_k.value(), self.spin_l.value()
        if h == 0 and k == 0 and l == 0: return
        
        import numpy as np
        recip_matrix = self.current_structure.lattice.reciprocal_lattice.matrix
        cart_vec = h * recip_matrix[0] + k * recip_matrix[1] + l * recip_matrix[2]
        
        # Check specifically for a PyVista camera
        if hasattr(self.renderer, 'plotter') and hasattr(self.renderer.plotter, 'camera'):
            fp = np.array(self.renderer.plotter.camera.focal_point)
            dist = self.renderer.plotter.camera.distance
            norm_vec = cart_vec / np.linalg.norm(cart_vec)
            self.renderer.plotter.camera.position = fp + norm_vec * dist
            self.renderer.plotter.camera.up = (0, 0, 1)
            self.renderer.plotter.render()
            
        # Check specifically for a Matplotlib axis
        elif hasattr(self.renderer, 'ax'):
            dist = np.linalg.norm(cart_vec)
            az = np.degrees(np.arctan2(cart_vec[1], cart_vec[0]))
            el = np.degrees(np.arcsin(cart_vec[2] / dist))
            self.renderer.ax.view_init(elev=el, azim=az)
            if hasattr(self.renderer, 'canvas'):
                self.renderer.canvas.draw_idle()

    def handle_camera_update(self):
        """Passes camera alignment requests to the active rendering backend."""
        # Note: We will implement the full HKL normal math in the core engine later,
        # but this registers the button clicks safely for now.
        if self.combo_projection.currentIndex() == 1:
            if hasattr(self.renderer.plotter, 'camera'):
                self.renderer.plotter.camera.enable_parallel_projection()
        else:
            if hasattr(self.renderer.plotter, 'camera'):
                self.renderer.plotter.camera.disable_parallel_projection()
        
        if hasattr(self.renderer, 'plotter') and hasattr(self.renderer.plotter, 'render'):
            self.renderer.plotter.render()
        elif hasattr(self.renderer, 'canvas'):
            self.renderer.canvas.draw_idle()

    def export_scripts(self):
        """Gathers crystal data and passes it to the external script formatting engine."""
        if not getattr(self, 'active_supercell', None):
            QMessageBox.warning(self, "Export Error", "Please draw a structure first!")
            return
            
        sender = self.sender()
        software = "3ds Max" if sender == self.btn_export_max else "Blender"
        file_path, _ = QFileDialog.getSaveFileName(self, f"Export {software} Script", "", "Python Files (*.py)")
        
        if not file_path: return

        import numpy as np
        from tensorspec.core.io.exporters import SceneExporter
        from tensorspec.core.crystallography import CrystalEngine

        atoms_data, bonds_data, lattice_data = [], [], []
        bz_solid_data = None

        # 1. Gather Atoms & Bonds
        if self.chk_exp_atoms.isChecked():
            scale_mod = self.spin_radius.value()
            for i, site in enumerate(self.active_supercell):
                if i in self.erased_atoms: continue
                radius = float((site.specie.atomic_radius if site.specie.atomic_radius else 1.0) * scale_mod)
                color = self.active_colors.get(site.specie.symbol, "#008080")
                atoms_data.append((float(site.coords[0]), float(site.coords[1]), float(site.coords[2]), radius, color))

            # Gather bonds if stick style is active
            if hasattr(self, 'combo_style') and self.combo_style.currentIndex() == 0:
                cyl_radius = float(self.spin_bond_thick.value())
                bond_color = self.active_colors.get("Bonds", "#FFFFFF")
                coords = self.active_supercell.cart_coords
                radii = np.array([s.specie.atomic_radius if s.specie.atomic_radius else 1.2 for s in self.active_supercell])
                thresh = self.spin_bond_thresh.value()
                
                dist_mat = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=-1)
                threshold_mat = (radii[:, np.newaxis] + radii[np.newaxis, :]) * thresh
                valid_pairs = np.triu((dist_mat > 0.5) & (dist_mat <= threshold_mat), k=1)
                
                for i, j in np.argwhere(valid_pairs):
                    if (i, j) in self.erased_bonds or i in self.erased_atoms or j in self.erased_atoms:
                        continue
                    bonds_data.append((float(coords[i][0]), float(coords[i][1]), float(coords[i][2]), 
                                       float(coords[j][0]), float(coords[j][1]), float(coords[j][2]), 
                                       cyl_radius, bond_color))

        # 2. Gather Unit Cell Wireframes
        if self.chk_exp_cell.isChecked() and getattr(self, 'current_structure', None):
            def get_edges(matrix, color):
                a, b, c = matrix[0], matrix[1], matrix[2]
                v = np.zeros((8, 3))
                v[1], v[2], v[3] = a, b, c
                v[4], v[5], v[6], v[7] = a + b, a + c, b + c, a + b + c
                edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
                for p1, p2 in edges:
                    lattice_data.append((float(v[p1][0]), float(v[p1][1]), float(v[p1][2]), 
                                         float(v[p2][0]), float(v[p2][1]), float(v[p2][2]), color))

            if self.chk_show_conventional.isChecked():
                get_edges(self.current_structure.lattice.matrix, "#FF5733")
            if self.chk_show_primitive.isChecked():
                prim_matrix = CrystalEngine.get_universal_primitive_matrix(self.current_structure)
                get_edges(prim_matrix, "#33FF57")

        # 3. Gather Brillouin Zone Data
        if self.chk_exp_bz.isChecked():
            # Check if BZ has actually been calculated/rendered yet
            if hasattr(self.renderer, 'bz_export_edges'):
                for p1, p2 in self.renderer.bz_export_edges:
                    lattice_data.append((float(p1[0]), float(p1[1]), float(p1[2]), 
                                         float(p2[0]), float(p2[1]), float(p2[2]), "#FF00FF"))
                
                if self.chk_bz_solid.isChecked() and hasattr(self.renderer, 'bz_hull_pts'):
                    bz_solid_data = {
                        'verts': self.renderer.bz_hull_pts.tolist(),
                        'faces': self.renderer.bz_hull_simplices.tolist()
                    }

        # 4. Route to external exporter engine
        if software == "3ds Max":
            SceneExporter.export_3dsmax(file_path, atoms_data, bonds_data, lattice_data, bz_solid_data)
        else:
            SceneExporter.export_blender(file_path, atoms_data, bonds_data, lattice_data, bz_solid_data)
            
        print(f"✅ Extracted scene data successfully exported to {file_path}")
        QMessageBox.information(self, "Export Complete", f"Successfully saved {software} script!")

    def handle_load_cif(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open CIF', '', "CIF files (*.cif)")
        if not fname: return
        try:
            self.current_structure = Structure.from_file(fname)
            self.lbl_file.setText(fname.split('/')[-1])
            
            # Delegate math to core
            sym_info = CrystalEngine.get_symmetry_info(self.current_structure)
            self.lbl_sym.setText(f"Space Group: {sym_info['spacegroup']} | V_conv = {sym_info['volume_ratio']}× V_prim")
            
            # Update UI color controls
            unique_els = sorted(list(set([s.specie.symbol for s in self.current_structure])))
            self.build_color_panel(unique_els)
            
            self.combo_cdw_el.clear()
            self.combo_cdw_el.addItem("All Elements")
            self.combo_cdw_el.addItems(unique_els)
            
            # Push reference to global workspace if manager exists
            if self.workspace:
                self.workspace[fname.split('/')[-1]] = {"type": "Structure", "data": self.current_structure}
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to parse CIF: {e}")

    def build_color_panel(self, elements: list):
        """Rebuilds color pickers dynamically when a new crystal is loaded."""
        while self.colors_layout.count():
            item = self.colors_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        for idx, el in enumerate(elements + ["Bonds"]):
            color = self.active_colors.get(el, "#008080")
            lbl = QLabel(f"{el}:")
            btn = QPushButton()
            btn.setStyleSheet(f"background-color: {color}; border: 1px solid white;")
            btn.clicked.connect(lambda checked, key=el, b=btn: self.pick_color(key, b))
            self.colors_layout.addWidget(lbl, idx, 0)
            self.colors_layout.addWidget(btn, idx, 1)

    def pick_color(self, key: str, button: QPushButton):
        color = QColorDialog.getColor()
        if color.isValid():
            hex_str = color.name()
            self.active_colors[key] = hex_str
            button.setStyleSheet(f"background-color: {hex_str}; border: 1px solid white;")
            self.refresh_render()

    def handle_draw(self):
        if not self.current_structure: return
        # Generate base supercell
        self.active_supercell = self.current_structure * (self.spin_scx.value(), self.spin_scy.value(), self.spin_scz.value())
        self.refresh_render()
        self.renderer.set_camera_preset('iso')

    def refresh_render(self):
        """Routes structure state through core math transformations, then pushes to PyVista."""
        if not self.active_supercell: return
        
        # 1. Apply CDW math if enabled
        render_struct = self.active_supercell.copy()
        if getattr(self, 'chk_cdw', None) and self.chk_cdw.isChecked():
            import numpy as np
            q_vec = (self.spin_qx.value(), self.spin_qy.value(), self.spin_qz.value())
            amp_vec = (self.spin_ax.value(), self.spin_ay.value(), self.spin_az.value())
            
            # Grab phase if it exists, default to 0 if not
            phase_rad = np.radians(self.spin_cdw_phase.value()) if hasattr(self, 'spin_cdw_phase') else 0.0
            
            render_struct = CrystalEngine.apply_cdw_distortion(
                render_struct, 
                self.combo_cdw_el.currentText(), 
                q_vec, 
                amp_vec, 
                phase_rad
            )

        # 2. Push hardware rendering pipeline
        self.renderer.clear_scene()
        is_shiny = self.chk_shiny.isChecked()
        scale = self.spin_radius.value()
        
        self.renderer.draw_atoms(render_struct, self.active_colors, scale_mod=scale, is_shiny=is_shiny)
        self.renderer.draw_bonds(render_struct, self.active_colors, is_shiny=is_shiny)
        
        # 3. Handle Lattice Boxes & Axes with the split Conventional/Primitive logic
        if self.current_structure:
            conv_mat = self.current_structure.lattice.matrix if self.chk_show_conventional.isChecked() else None
            prim_mat = CrystalEngine.get_universal_primitive_matrix(self.current_structure) if self.chk_show_primitive.isChecked() else None
            
            if getattr(self, 'chk_axes', None) and self.chk_axes.isChecked():
                self.renderer.draw_axes(conventional_matrix=conv_mat, primitive_matrix=prim_mat)
                
            if conv_mat is not None or prim_mat is not None:
                self.renderer.draw_lattice_boxes(conventional_matrix=conv_mat, primitive_matrix=prim_mat)
            
        # Support both PyVista and Matplotlib refresh commands
        if hasattr(self.renderer, 'plotter') and hasattr(self.renderer.plotter, 'render'):
            self.renderer.plotter.render()
        elif hasattr(self.renderer, 'canvas'):
            self.renderer.canvas.draw_idle()

    def handle_add_template(self):
        name = self.combo_tpl.currentText()
        struct = CrystalEngine.generate_template_structure(name)
        if struct: self.append_stack_layer(name, struct)
    
    def move_layer_up(self, row):
        idx = self.stack_layer_rows.index(row)
        if idx > 0:
            self.stack_layer_rows.pop(idx)
            self.stack_layer_rows.insert(idx - 1, row)
            self.scroll_layout.insertWidget(idx - 1, row)

    def move_layer_down(self, row):
        idx = self.stack_layer_rows.index(row)
        if idx < len(self.stack_layer_rows) - 1:
            self.stack_layer_rows.pop(idx)
            self.stack_layer_rows.insert(idx + 1, row)
            self.scroll_layout.insertWidget(idx + 1, row)

    def save_layer_as_template(self, layer_widget):
        import json
        clean_name = layer_widget.lbl_name.text().replace("<b>", "").replace("</b>", "")
        name, ok = QInputDialog.getText(self, "Save Template", "Enter a name for this custom template:", text=clean_name)
        if ok and name:
            templates = {}
            if os.path.exists("user_templates.json"):
                with open("user_templates.json", "r") as f:
                    templates = json.load(f)
            templates[name] = layer_widget.struct.as_dict()
            with open("user_templates.json", "w") as f:
                json.dump(templates, f)
            QMessageBox.information(self, "Success", f"'{name}' has been saved to your templates!")

    def extract_monolayer_from_bulk(self, *args):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Bulk CIF', '', "CIF files (*.cif)")
        if not fname: return

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bulk_struct = Structure.from_file(fname, primitive=False) 
                bulk_struct.translate_sites(list(range(len(bulk_struct))), [0.01, 0.01, 0.01], to_unit_cell=True)
        except Exception as e:
            QMessageBox.critical(self, "Parser Error", f"Could not parse structure file: {str(e)}")
            return

        mode = self.combo_exfoliate_mode.currentText()

        if "van der Waals" in mode:
            try:
                mono, gap = CrystalEngine.extract_monolayer_vdw(bulk_struct)
                name = fname.split('/')[-1].replace('.cif', '') + " (vdW Auto-Mono)"
                self.append_stack_layer(name, mono)
            except Exception as e:
                QMessageBox.critical(self, "vdW Cleave Error", str(e))
        else:
            num_layers, ok1 = QInputDialog.getInt(self, "Exfoliator", "Number of layers to extract:", 1, 1, 10)
            if not ok1: return
            hkl_str, ok2 = QInputDialog.getText(self, "Exfoliator", "Cleavage Plane (h k l):", text="0 0 1")
            if not ok2: return
            
            try:
                hkl = tuple(map(int, hkl_str.replace(',', ' ').split()))
                from pymatgen.core.surface import SlabGenerator
                
                approx_thickness = 0.1 if num_layers == 1 else (num_layers * 3.0) - 1.5 
                
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    slabgen = SlabGenerator(bulk_struct, miller_index=hkl, min_slab_size=approx_thickness, min_vacuum_size=25.0, center_slab=True)
                    slabs = slabgen.get_slabs()
                    
                if not slabs: raise ValueError(f"Could not generate slabs for plane {hkl}")
                
                # --- THE ULTIMATE FIX: Topological Chemical Graphing ---
                # We ignore the Z-axis completely. We build a network of chemical bonds.
                # A 2D layer is defined as a single connected component of atoms!
                raw_slab = slabs[0]
                dist_mat = raw_slab.distance_matrix # PyMatgen's periodic distance matrix
                
                bond_threshold = 3.2 # Ångströms (Safely captures covalent bonds, avoids vdW gaps)
                num_sites = len(raw_slab)
                visited = set()
                layers = []
                import numpy as np
                
                # Breadth-First Search (BFS) to cluster bonded atoms into layers
                for i in range(num_sites):
                    if i not in visited:
                        queue = [i]
                        current_layer = []
                        while queue:
                            curr = queue.pop(0)
                            if curr not in visited:
                                visited.add(curr)
                                current_layer.append(curr)
                                # Find all atoms bonded to this one
                                neighbors = np.where((dist_mat[curr] > 0.1) & (dist_mat[curr] < bond_threshold))[0]
                                queue.extend([n for n in neighbors if n not in visited])
                        layers.append(current_layer)
                
                # Sort the detected discrete layers by their average Z-height
                layers.sort(key=lambda indices: np.mean([raw_slab[idx].coords[2] for idx in indices]))
                
                # Grab EXACTLY the requested number of discrete layers
                target_layers = layers[:min(num_layers, len(layers))]
                sites_to_keep = []
                for layer_indices in target_layers:
                    sites_to_keep.extend([raw_slab[idx] for idx in layer_indices])
                
                # Reconstruct the structure safely
                new_matrix = raw_slab.lattice.matrix.copy()
                new_matrix[2] = [0, 0, 25.0] 
                from pymatgen.core import Lattice
                new_lat = Lattice(new_matrix)
                
                final_coords = [s.coords for s in sites_to_keep]
                z_vals = [c[2] for c in final_coords]
                z_center = (max(z_vals) + min(z_vals)) / 2.0
                final_coords = [[c[0], c[1], c[2] - z_center + 12.5] for c in final_coords]
                
                mono_struct = Structure(new_lat, [s.specie for s in sites_to_keep], final_coords, coords_are_cartesian=True)
                
                name = fname.split('/')[-1].replace('.cif', '') + f" ({num_layers}-Layer {hkl_str.replace(' ', '')})"
                self.append_stack_layer(name, mono_struct)
            except Exception as e:
                QMessageBox.critical(self, "Exfoliation Error", str(e))

    def append_stack_layer(self, name: str, struct: Structure):
        default_z = max([r.spin_z.value() for r in self.stack_layer_rows]) + 3.4 if self.stack_layer_rows else 0.0
        row = StackLayerRow(name, struct, default_z)
        
        # Connect all buttons!
        row.btn_delete.clicked.connect(lambda: self.remove_stack_layer(row))
        row.btn_up.clicked.connect(lambda: self.move_layer_up(row))
        row.btn_down.clicked.connect(lambda: self.move_layer_down(row))
        row.btn_save_tpl.clicked.connect(lambda: self.save_layer_as_template(row))
        
        self.scroll_layout.addWidget(row)
        self.stack_layer_rows.append(row)
        
        for el in set([s.specie.symbol for s in struct]):
            if el not in self.active_colors: self.active_colors[el] = CPK_COLORS.get(el, "#008080")

    def remove_stack_layer(self, row_widget):
        self.scroll_layout.removeWidget(row_widget)
        row_widget.deleteLater()
        self.stack_layer_rows.remove(row_widget)

    def handle_draw_stack(self):
        if not self.stack_layer_rows: return
        layers_data = [row.get_layer_dict() for row in self.stack_layer_rows]
        
        # 1. Build the math supercell
        self.active_supercell = CrystalEngine.build_heterostructure_stack(layers_data)
        self.current_structure = self.active_supercell
        
        # 2. Extract elements to wake up Tab 1's Color Panel
        if "layer_tag" in self.active_supercell.site_properties:
            unique_elements = sorted(list(set(self.active_supercell.site_properties["layer_tag"])))
        else:
            unique_elements = sorted(list(set([site.specie.symbol for site in self.active_supercell])))

        # Ensure CPK fallback colors are loaded for any new elements
        for el in unique_elements:
            base_el = el.split('_')[0] if '_' in el else el
            if el not in self.active_colors: 
                self.active_colors[el] = CPK_COLORS.get(base_el, "#008080")

        self.build_color_panel(unique_elements)
        
        # 3. Draw to screen
        self.refresh_render()
        if hasattr(self.renderer, 'set_camera_preset'):
            self.renderer.set_camera_preset('z')

    def handle_moire(self):
        if len(self.stack_layer_rows) != 2:
            self.lbl_moire.setText("Error: Requires exactly 2 stacked layers.")
            return
        
        l1, l2 = self.stack_layer_rows[0].get_layer_dict(), self.stack_layer_rows[1].get_layer_dict()
        result = CrystalEngine.calculate_moire_superlattice(l1['struct'], l2['struct'], l1['twist'], l2['twist'])
        
        if result["status"] == "commensurate":
            self.lbl_moire.setText(f"🟢 Commensurate! Periodicity: {result['periodicity']:.2f} Å ({result['n_cells']}×{result['n_cells']})")
            z_vals = [r.spin_z.value() for r in self.stack_layer_rows]
            self.renderer.draw_moire_envelope(result["matrix"], min(z_vals)-1.5, max(z_vals)+1.5)
            self.renderer.plotter.render()
        else:
            self.lbl_moire.setText(f"🟡 Incommensurate / Error: {result.get('message', 'Strain required.')}")

    def handle_draw_bz(self):
        if not self.current_structure: return
        bz_data = CrystalEngine.calculate_brillouin_zone(self.current_structure)
        if bz_data:
            self.renderer.clear_scene()
            self.renderer.draw_brillouin_zone(bz_data["points"], bz_data["simplices"], solid=True)
            self.renderer.plotter.render()

    def closeEvent(self, event):
        """Safely shuts down ALL rendering pipelines (visible and hidden) to prevent segfaults."""
        # 1. Safely kill the hidden PyVista GPU Engine (The main culprit for crashes)
        if hasattr(self, 'renderer_gpu') and hasattr(self.renderer_gpu, 'plotter'):
            try:
                self.renderer_gpu.plotter.close()
            except Exception:
                pass
                
        # 2. Safely kill the Matplotlib CPU Engine
        if hasattr(self, 'renderer_cpu') and hasattr(self.renderer_cpu, 'figure'):
            try:
                import matplotlib.pyplot as plt
                plt.close(self.renderer_cpu.figure)
            except Exception:
                pass
        
        # Now that all graphics memory is detached, allow the window to close
        event.accept()