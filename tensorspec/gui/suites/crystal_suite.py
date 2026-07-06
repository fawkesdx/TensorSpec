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
from tensorspec.plotting.backends.pyvista_engine import PyVistaCrystalBackend


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
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.lbl_name = QLabel(f"<b>{name}</b>")
        self.lbl_name.setFixedWidth(120)
        layout.addWidget(self.lbl_name)
        
        layout.addWidget(QLabel("SC:"))
        self.spin_x = QSpinBox(); self.spin_x.setRange(1, 50); self.spin_x.setValue(1)
        self.spin_y = QSpinBox(); self.spin_y.setRange(1, 50); self.spin_y.setValue(1)
        layout.addWidget(self.spin_x); layout.addWidget(self.spin_y)
        
        layout.addWidget(QLabel("z (Å):"))
        self.spin_z = QDoubleSpinBox(); self.spin_z.setRange(-100.0, 100.0); self.spin_z.setSingleStep(0.5); self.spin_z.setValue(default_z)
        layout.addWidget(self.spin_z)
        
        layout.addWidget(QLabel("θ (°):"))
        self.spin_twist = QDoubleSpinBox(); self.spin_twist.setRange(-360.0, 360.0); self.spin_twist.setSingleStep(1.0)
        layout.addWidget(self.spin_twist)
        
        self.btn_delete = QPushButton("X")
        self.btn_delete.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        self.btn_delete.setFixedWidth(28)
        layout.addWidget(self.btn_delete)

    def get_layer_dict(self) -> dict:
        """Packages UI parameters for the mathematical core engine."""
        return {
            "struct": self.struct,
            "sc_x": self.spin_x.value(),
            "sc_y": self.spin_y.value(),
            "z_shift": self.spin_z.value(),
            "twist": self.spin_twist.value()
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

        # 2. Right 3D Viewport (PyVista Hardware Engine)
        self.renderer = PyVistaCrystalBackend(parent=self)
        splitter.addWidget(self.renderer.plotter)
        
        splitter.setSizes([380, 820])

    # ================= TAB 1: VIEW & EDIT =================
    def init_tab_view_edit(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.btn_load = QPushButton("📂 Load CIF File")
        self.btn_load.clicked.connect(self.handle_load_cif)
        layout.addWidget(self.btn_load)
        
        self.lbl_file = QLabel("No file loaded"); self.lbl_file.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_file)
        
        self.lbl_sym = QLabel("Space Group: N/A"); self.lbl_sym.setStyleSheet("color: #aaaaaa; font-style: italic;")
        layout.addWidget(self.lbl_sym)

        # Geometry & Scaling
        group_geom = QGroupBox("Geometry & Supercell")
        geom_layout = QVBoxLayout(group_geom)
        
        sc_layout = QHBoxLayout()
        sc_layout.addWidget(QLabel("Supercell (X,Y,Z):"))
        self.spin_scx = QSpinBox(); self.spin_scx.setValue(1); self.spin_scx.setRange(1, 20)
        self.spin_scy = QSpinBox(); self.spin_scy.setValue(1); self.spin_scy.setRange(1, 20)
        self.spin_scz = QSpinBox(); self.spin_scz.setValue(1); self.spin_scz.setRange(1, 20)
        sc_layout.addWidget(self.spin_scx); sc_layout.addWidget(self.spin_scy); sc_layout.addWidget(self.spin_scz)
        geom_layout.addLayout(sc_layout)

        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Radius Scale:"))
        self.spin_rad = QDoubleSpinBox(); self.spin_rad.setRange(0.1, 3.0); self.spin_rad.setValue(0.5); self.spin_rad.setSingleStep(0.1)
        scale_layout.addWidget(self.spin_rad)
        geom_layout.addLayout(scale_layout)
        layout.addWidget(group_geom)

        # Rendering & Styles
        group_render = QGroupBox("Styles & Lighting")
        render_layout = QVBoxLayout(group_render)
        self.chk_shiny = QCheckBox("High Quality PBR Shading"); self.chk_shiny.stateChanged.connect(self.refresh_render)
        self.chk_axes = QCheckBox("Show Lattice Axes"); self.chk_axes.setChecked(True); self.chk_axes.stateChanged.connect(self.refresh_render)
        self.chk_box = QCheckBox("Show Unit Cell Box"); self.chk_box.setChecked(True); self.chk_box.stateChanged.connect(self.refresh_render)
        render_layout.addWidget(self.chk_shiny); render_layout.addWidget(self.chk_axes); render_layout.addWidget(self.chk_box)
        layout.addWidget(group_render)

        # Dynamic Color Pickers
        self.group_colors = QGroupBox("Dynamic Element Colors")
        self.colors_layout = QGridLayout(self.group_colors)
        layout.addWidget(self.group_colors)

        self.btn_draw = QPushButton("🎨 Render Structure")
        self.btn_draw.setStyleSheet("background-color: #2b5c8f; color: white; font-weight: bold; padding: 6px;")
        self.btn_draw.clicked.connect(self.handle_draw)
        layout.addWidget(self.btn_draw)

        layout.addStretch()
        self.tabs.addTab(tab, "1. View & Edit")

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

        # Exfoliator
        exf_group = QGroupBox("Bulk Exfoliator")
        exf_layout = QVBoxLayout(exf_group)
        btn_vdw = QPushButton("✂️ Auto vdW Cleave Bulk CIF")
        btn_vdw.clicked.connect(self.handle_vdw_cleave)
        exf_layout.addWidget(btn_vdw)
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
        if self.chk_cdw.isChecked():
            q_vec = (self.spin_qx.value(), self.spin_qy.value(), self.spin_qz.value())
            amp_vec = (self.spin_ax.value(), self.spin_ay.value(), self.spin_az.value())
            render_struct = CrystalEngine.apply_cdw_distortion(render_struct, self.combo_cdw_el.currentText(), q_vec, amp_vec, 0.0)

        # 2. Push hardware rendering pipeline
        self.renderer.clear_scene()
        is_shiny = self.chk_shiny.isChecked()
        scale = self.spin_rad.value()
        
        self.renderer.draw_atoms(render_struct, self.active_colors, scale_mod=scale, is_shiny=is_shiny)
        self.renderer.draw_bonds(render_struct, self.active_colors, is_shiny=is_shiny)
        
        if self.chk_axes.isChecked():
            self.renderer.draw_axes(conventional_matrix=self.current_structure.lattice.matrix)
        if self.chk_box.isChecked():
            self.renderer.draw_lattice_boxes(conventional_matrix=self.current_structure.lattice.matrix)
            
        self.renderer.plotter.render()

    def handle_add_template(self):
        name = self.combo_tpl.currentText()
        struct = CrystalEngine.generate_template_structure(name)
        if struct: self.append_stack_layer(name, struct)

    def handle_vdw_cleave(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Bulk CIF', '', "CIF files (*.cif)")
        if not fname: return
        try:
            bulk = Structure.from_file(fname)
            mono, gap = CrystalEngine.extract_monolayer_vdw(bulk)
            self.append_stack_layer(f"vdW Cleave ({gap:.1f}Å gap)", mono)
        except Exception as e:
            QMessageBox.critical(self, "Cleave Error", str(e))

    def append_stack_layer(self, name: str, struct: Structure):
        default_z = max([r.spin_z.value() for r in self.stack_layer_rows]) + 3.4 if self.stack_layer_rows else 0.0
        row = StackLayerRow(name, struct, default_z)
        row.btn_delete.clicked.connect(lambda: self.remove_stack_layer(row))
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
        self.active_supercell = CrystalEngine.build_heterostructure_stack(layers_data)
        self.current_structure = self.active_supercell
        self.refresh_render()
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