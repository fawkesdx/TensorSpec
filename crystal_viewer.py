import os
import sys
import platform
import numpy as np
from scipy.spatial import ConvexHull

# --- Cross-Platform Environment Standardization ---
if platform.system() == "Darwin":  
    os.environ["QT_API"] = "pyside6"
    os.environ["QT_MAC_WANTS_LAYER"] = "1"
    BASIC_GRAPHICS = True 
else:  
    os.environ["QT_API"] = os.environ.get("QT_API", "pyside6")
    BASIC_GRAPHICS = False

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                               QSpinBox, QDoubleSpinBox, QComboBox, QColorDialog,
                               QStackedWidget, QTabWidget, QCheckBox, QGroupBox, QGridLayout)
from PySide6.QtCore import Qt

from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN

# --- ENGINE 1: PyVista (GPU Fast Mode) ---
import pyvista as pv
from pyvistaqt import QtInteractor
pv.global_theme.depth_peeling.enabled = False
pv.global_theme.anti_aliasing = None

# --- ENGINE 2: Matplotlib (CPU Safe Mode) ---
import matplotlib
matplotlib.use('qtagg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Default Fallback Colors
CPK_COLORS = {
    "H": "#FFFFFF", "C": "#333333", "N": "#2233FF", "O": "#FF2200",
    "V": "#999999", "Te": "#FF8C00", "Fe": "#E06633", "Cu": "#C88033"
}

class TensorSpecApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorSpec - Crystal Suite")
        self.resize(1300, 800) 

        self.current_structure = None
        self.active_colors = {"Bonds": "#d3d3d3"} # Stores current colors for rendering

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # ================= L E F T   P A N E L  (T A B S) =================
        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(320)
        self.main_layout.addWidget(self.tabs)

        # --- TAB 1: VIEW & EDIT ---
        self.tab1 = QWidget()
        self.ui_panel = QVBoxLayout(self.tab1)
        
        self.btn_load = QPushButton("Load CIF File")
        self.btn_load.clicked.connect(self.load_file)
        self.ui_panel.addWidget(self.btn_load)
        
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui_panel.addWidget(self.lbl_file)

        self.ui_panel.addWidget(QLabel("Graphics Backend:"))
        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["CPU Safe Mode (Matplotlib)", "GPU Fast Mode (PyVista)"])
        self.combo_backend.currentIndexChanged.connect(self.switch_backend)
        self.ui_panel.addWidget(self.combo_backend)

        # --- Supercell & Scaling Group ---
        group_geom = QGroupBox("Geometry & Scaling")
        geom_layout = QVBoxLayout(group_geom)
        
        geom_layout.addWidget(QLabel("Supercell (X, Y, Z):"))
        self.spin_x = QSpinBox(); self.spin_x.setValue(1); self.spin_x.setMinimum(1)
        self.spin_y = QSpinBox(); self.spin_y.setValue(1); self.spin_y.setMinimum(1)
        self.spin_z = QSpinBox(); self.spin_z.setValue(1); self.spin_z.setMinimum(1)
        for spin in [self.spin_x, self.spin_y, self.spin_z]:
            geom_layout.addWidget(spin)

        geom_layout.addWidget(QLabel("Atom Radius Scale:"))
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0.1, 5.0); self.spin_radius.setValue(0.5); self.spin_radius.setSingleStep(0.1)
        geom_layout.addWidget(self.spin_radius)

        geom_layout.addWidget(QLabel("Bond/Stick Thickness:"))
        self.spin_bond_thick = QDoubleSpinBox()
        self.spin_bond_thick.setRange(0.01, 1.0); self.spin_bond_thick.setValue(0.10); self.spin_bond_thick.setSingleStep(0.02)
        geom_layout.addWidget(self.spin_bond_thick)
        self.ui_panel.addWidget(group_geom)

        # --- Appearance & Colors Group (Dynamic) ---
        self.group_colors = QGroupBox("Styles & Colors")
        self.colors_layout = QGridLayout(self.group_colors)
        
        self.colors_layout.addWidget(QLabel("Connections:"), 0, 0)
        self.combo_style = QComboBox()
        self.combo_style.addItems(["Bonds (Sticks)", "Polyhedra (Planes)", "None"])
        self.colors_layout.addWidget(self.combo_style, 0, 1)
        self.ui_panel.addWidget(self.group_colors)

       # --- Interactive Mode ---
        self.chk_edit_mode = QCheckBox("Enable Interactive Delete (Drag to erase)")
        self.chk_edit_mode.setStyleSheet("color: #d9534f; font-weight: bold;")
        self.chk_edit_mode.stateChanged.connect(self.toggle_edit_mode)
        self.ui_panel.addWidget(self.chk_edit_mode)

        # --- Action Buttons ---
        self.btn_draw = QPushButton("Draw Structure")
        self.btn_draw.clicked.connect(self.draw_structure)
        self.btn_draw.setStyleSheet("background-color: #2b5c8f; font-weight: bold; color: white; padding: 5px;")
        self.ui_panel.addWidget(self.btn_draw)

        self.btn_save = QPushButton("Save High-Res Image")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setStyleSheet("background-color: #28a745; font-weight: bold; color: white; padding: 5px;")
        self.ui_panel.addWidget(self.btn_save)

        self.ui_panel.addStretch()
        self.tabs.addTab(self.tab1, "1. View & Edit")

        # --- TAB 2: CDW MODULATOR ---
        self.tab2 = QWidget()
        tab2_layout = QVBoxLayout(self.tab2)
        
        self.chk_cdw_enable = QCheckBox("Enable CDW Distortion")
        self.chk_cdw_enable.setStyleSheet("font-weight: bold; color: #2b5c8f;")
        self.chk_cdw_enable.stateChanged.connect(self.draw_structure)
        tab2_layout.addWidget(self.chk_cdw_enable)
        
        tab2_layout.addWidget(QLabel("Target Atom Type:"))
        self.combo_cdw_element = QComboBox()
        self.combo_cdw_element.addItem("All Elements")
        self.combo_cdw_element.currentIndexChanged.connect(self.draw_structure)
        tab2_layout.addWidget(self.combo_cdw_element)
        
        # Wavevector Modulation Group
        group_q = QGroupBox("Modulation Wavevector q (rlu)")
        q_layout = QVBoxLayout(group_q)
        self.spin_qx = QDoubleSpinBox(); self.spin_qx.setRange(-5.0, 5.0); self.spin_qx.setSingleStep(0.05); self.spin_qx.setDecimals(3)
        self.spin_qy = QDoubleSpinBox(); self.spin_qy.setRange(-5.0, 5.0); self.spin_qy.setSingleStep(0.05); self.spin_qy.setDecimals(3)
        self.spin_qz = QDoubleSpinBox(); self.spin_qz.setRange(-5.0, 5.0); self.spin_qz.setSingleStep(0.05); self.spin_qz.setDecimals(3)
        for spin, lbl in [(self.spin_qx, "q_a :"), (self.spin_qy, "q_b :"), (self.spin_qz, "q_c :")]:
            q_layout.addWidget(QLabel(lbl))
            q_layout.addWidget(spin)
            spin.valueChanged.connect(self.draw_structure)
        tab2_layout.addWidget(group_q)
        
        # Displacement Amplitude Group
        group_amp = QGroupBox("Displacement Amplitude A (Å)")
        amp_layout = QVBoxLayout(group_amp)
        self.spin_ax = QDoubleSpinBox(); self.spin_ax.setRange(-2.0, 2.0); self.spin_ax.setSingleStep(0.01); self.spin_amp_decimals = 3
        self.spin_ay = QDoubleSpinBox(); self.spin_ay.setRange(-2.0, 2.0); self.spin_ay.setSingleStep(0.01)
        self.spin_az = QDoubleSpinBox(); self.spin_az.setRange(-2.0, 2.0); self.spin_az.setSingleStep(0.01)
        for spin, lbl in [(self.spin_ax, "Δx (along a):"), (self.spin_ay, "Δy (along b):"), (self.spin_az, "Δz (along c):")]:
            amp_layout.addWidget(QLabel(lbl))
            amp_layout.addWidget(spin)
            spin.valueChanged.connect(self.draw_structure)
        tab2_layout.addWidget(group_amp)
        
        # Phase Adjuster
        tab2_layout.addWidget(QLabel("Phase Shift φ (degrees):"))
        self.spin_cdw_phase = QDoubleSpinBox()
        self.spin_cdw_phase.setRange(0.0, 360.0); self.spin_cdw_phase.setSingleStep(15.0)
        self.spin_cdw_phase.valueChanged.connect(self.draw_structure)
        tab2_layout.addWidget(self.spin_cdw_phase)
        
        tab2_layout.addStretch()
        self.tabs.addTab(self.tab2, "2. CDW Modulator")

        # --- TAB 3: BRILLOUIN ZONE (Placeholder kept safe) ---
        self.tab3 = QWidget()
        tab3_layout = QVBoxLayout(self.tab3)
        tab3_layout.addWidget(QLabel("Wigner-Seitz cells and surface projections will go here."))
        tab3_layout.addStretch()
        self.tabs.addTab(self.tab3, "3. Brillouin Zone")


        # ================= R I G H T   P A N E L (Stacked Viewers) =================
        self.viewer_stack = QStackedWidget()
        self.main_layout.addWidget(self.viewer_stack)

        # Widget 0: Matplotlib
        self.figure = Figure(facecolor="#1e1e24")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor("#1e1e24")
        self.ax.axis('off')
        self.canvas.mpl_connect('pick_event', self.on_matplotlib_pick)
        
        # Track dragging events for the Matplotlib backend
        self.canvas.mpl_connect('button_press_event', self.start_matplotlib_eraser)
        self.canvas.mpl_connect('button_release_event', self.stop_matplotlib_eraser)
        self.canvas.mpl_connect('motion_notify_event', self.matplotlib_eraser_brush)
        
        self.viewer_stack.addWidget(self.canvas)

        # Widget 1: PyVista
        self.plotter = QtInteractor(self.central_widget)
        self.plotter.set_background("#1e1e24")
        self.plotter.enable_mesh_picking(callback=self.on_pyvista_pick, show=False, show_message=False)
        
        # --- NEW: Eraser Brush Observers ---
        self.is_eraser_dragging = False
        # The priority number 10 forces our eraser to handle mouse inputs BEFORE the camera style does
        self.plotter.iren.add_observer("LeftButtonPressEvent", self.start_eraser, 10)
        self.plotter.iren.add_observer("LeftButtonReleaseEvent", self.stop_eraser, 10)
        self.plotter.iren.add_observer("MouseMoveEvent", self.eraser_brush, 10)
        
        self.viewer_stack.addWidget(self.plotter)

    # ================= C O L O R   &   U I   L O G I C =================
    def build_dynamic_color_panel(self, elements):
        """Builds VESTA-style color pickers based on the elements in the CIF"""
        # Clear existing dynamic color buttons (skip row 0 which is the combo box)
        for i in reversed(range(self.colors_layout.count())):
            item = self.colors_layout.itemAt(i)
            if item.widget() and item.widget() != self.combo_style and not isinstance(item.widget(), QLabel):
                item.widget().setParent(None)
                
        # Re-add label for combo style just in case
        self.colors_layout.addWidget(QLabel("Connections:"), 0, 0)
        self.colors_layout.addWidget(self.combo_style, 0, 1)

        row = 1
        self.active_colors = {"Bonds": "#d3d3d3"} # Reset and initialize bonds

        for el in elements:
            default_color = CPK_COLORS.get(el, "#008080")
            self.active_colors[el] = default_color
            
            lbl = QLabel(f"{el} Color:")
            btn = QPushButton()
            btn.setStyleSheet(f"background-color: {default_color}; border: 1px solid white;")
            btn.clicked.connect(lambda checked, key=el, b=btn: self.pick_color(key, b))
            
            self.colors_layout.addWidget(lbl, row, 0)
            self.colors_layout.addWidget(btn, row, 1)
            row += 1

        # Add Bonds color picker
        lbl_bond = QLabel("Bond Color:")
        btn_bond = QPushButton()
        btn_bond.setStyleSheet(f"background-color: {self.active_colors['Bonds']}; border: 1px solid white;")
        btn_bond.clicked.connect(lambda checked, key="Bonds", b=btn_bond: self.pick_color(key, b))
        
        self.colors_layout.addWidget(lbl_bond, row, 0)
        self.colors_layout.addWidget(btn_bond, row, 1)

    def pick_color(self, key, button):
        color = QColorDialog.getColor()
        if color.isValid():
            hex_color = color.name()
            self.active_colors[key] = hex_color
            button.setStyleSheet(f"background-color: {hex_color}; border: 1px solid white;")
            self.draw_structure() # Auto-redraw to apply

    def switch_backend(self, index):
        self.viewer_stack.setCurrentIndex(index)
        self.draw_structure() 

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "CIF files (*.cif)")
        if fname:
            self.lbl_file.setText(fname.split('/')[-1])
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.current_structure = Structure.from_file(fname)
                    
                    # Extract unique elements and build UI
                    unique_elements = sorted(list(set([site.specie.symbol for site in self.current_structure])))
                    self.build_dynamic_color_panel(unique_elements)

                    # Populate CDW target selection dropdown dynamically
                    self.combo_cdw_element.clear()
                    self.combo_cdw_element.addItem("All Elements")
                    self.combo_cdw_element.addItems(unique_elements)
                    
            except Exception as e:
                self.lbl_file.setText("Error loading CIF")

    def save_image(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
        if fname:
            if self.combo_backend.currentIndex() == 0:
                self.figure.savefig(fname, dpi=300, bbox_inches='tight', facecolor=self.figure.get_facecolor())
            else:
                self.plotter.screenshot(fname)

    def draw_structure(self):
        if self.current_structure is None: return
        backend = self.combo_backend.currentIndex()
        supercell = self.current_structure * (self.spin_x.value(), self.spin_y.value(), self.spin_z.value())
        
        # --- Apply CDW Distortion Propagation ---
        if hasattr(self, 'chk_cdw_enable') and self.chk_cdw_enable.isChecked():
            target_el = self.combo_cdw_element.currentText()
            qx, qy, qz = self.spin_qx.value(), self.spin_qy.value(), self.spin_qz.value()
            ax, ay, az = self.spin_ax.value(), self.spin_ay.value(), self.spin_az.value()
            phase = np.radians(self.spin_cdw_phase.value())
            
            for site in supercell:
                if target_el == "All Elements" or site.specie.symbol == target_el:
                    # Use fractional coordinates (r) to calculate the wave factor: q · r
                    fx, fy, fz = site.frac_coords
                    wave_argument = 2 * np.pi * (qx * fx + qy * fy + qz * fz) + phase
                    
                    # Compute spatial displacement u = A * cos(2π q·r + φ)
                    dx = ax * np.cos(wave_argument)
                    dy = ay * np.cos(wave_argument)
                    dz = az * np.cos(wave_argument)
                    
                    # Shift the atom's Cartesian space coordinates in place for rendering
                    site.coords += np.array([dx, dy, dz])

        if backend == 0: self.draw_matplotlib(supercell)
        else: self.draw_pyvista(supercell)

    def toggle_edit_mode(self):
        """Swaps the interactor style to physically disable camera rotation during edit mode."""
        if self.combo_backend.currentIndex() == 1: # PyVista backend
            if self.chk_edit_mode.isChecked():
                # Apply a blank interaction style that physically cannot rotate or pan
                blank_style = pv.vtk.vtkInteractorStyleUser()
                self.plotter.iren.set_interactor_style(blank_style)
            else:
                # Restore normal 3D rotation controls
                self.plotter.enable_trackball_style()

    def on_pyvista_pick(self, mesh):
        """Single-click deletion fallback."""
        if self.chk_edit_mode.isChecked() and mesh is not None:
            self.plotter.remove_actor(mesh)

    def start_eraser(self, obj, event):
        if self.chk_edit_mode.isChecked():
            self.is_eraser_dragging = True
            obj.SetAbortFlag(1)  # Intercept event to freeze camera rotation

    def stop_eraser(self, obj, event):
        if self.chk_edit_mode.isChecked():
            self.is_eraser_dragging = False
            obj.SetAbortFlag(1)  # Intercept event

    def eraser_brush(self, obj, event):
        """Continuously deletes actors under the cursor while dragging, keeping camera frozen."""
        if self.chk_edit_mode.isChecked() and getattr(self, 'is_eraser_dragging', False):
            obj.SetAbortFlag(1)  # Completely absorb the dragging event so camera stays locked
            
            click_x, click_y = self.plotter.iren.get_event_position()
            picker = pv.vtk.vtkPropPicker()
            picker.Pick(click_x, click_y, 0, self.plotter.renderer)
            actor = picker.GetActor()
            if actor:
                self.plotter.remove_actor(actor)

    def toggle_edit_mode(self):
        """Freezes the 3D perspective camera when editing is active."""
        if self.combo_backend.currentIndex() == 0:  # Matplotlib backend
            if self.chk_edit_mode.isChecked():
                self.ax.disable_mouse_rotation()  # Turn off Matplotlib built-in rotation
            else:
                self.ax.mouse_init()  # Restore standard rotation controls
        elif self.combo_backend.currentIndex() == 1:  # PyVista backend
            if self.chk_edit_mode.isChecked():
                blank_style = pv.vtk.vtkInteractorStyleUser()
                self.plotter.iren.set_interactor_style(blank_style)
            else:
                self.plotter.enable_trackball_style()

    def on_matplotlib_pick(self, event):
        """Fallback single click selection."""
        if self.chk_edit_mode.isChecked():
            event.artist.remove()
            self.canvas.draw()

    def start_matplotlib_eraser(self, event):
        if self.chk_edit_mode.isChecked() and event.button == 1:
            self.is_eraser_dragging = True

    def stop_matplotlib_eraser(self, event):
        if event.button == 1:
            self.is_eraser_dragging = False

    def matplotlib_eraser_brush(self, event):
        """Wipes out atoms and bonds under the crosshair continuously while dragging."""
        if self.chk_edit_mode.isChecked() and getattr(self, 'is_eraser_dragging', False) and event.inaxes == self.ax:
            # Check if mouse is hovering over an atom collection
            for collection in list(self.ax.collections):
                contained, _ = collection.contains(event)
                if contained:
                    collection.remove()
                    self.canvas.draw_idle()
                    return
            # Check if mouse is hovering over a bond line
            for line in list(self.ax.lines):
                contained, _ = line.contains(event)
                if contained:
                    line.remove()
                    self.canvas.draw_idle()
                    return

    # ================= M A T P L O T L I B   R E N D E R =================
    def draw_matplotlib(self, supercell):
        self.ax.clear()
        self.ax.set_facecolor("#1e1e24")
        self.ax.axis('off')

        scale_mod = self.spin_radius.value()
        style = self.combo_style.currentIndex()

        for site in supercell:
            x, y, z = site.coords
            radius = (site.specie.atomic_radius if site.specie.atomic_radius else 1.0) * scale_mod
            atom_color = self.active_colors.get(site.specie.symbol, "#008080")
            self.ax.scatter(x, y, z, s=(radius * 800), c=atom_color, edgecolors='black', picker=True)

        if style == 0: self.draw_matplotlib_bonds(supercell)
        elif style == 1: self.draw_matplotlib_polyhedra(supercell)
        self.canvas.draw()

    def draw_matplotlib_bonds(self, supercell):
        line_thick = self.spin_bond_thick.value() * 40 # Scale thickness for Matplotlib
        bond_color = self.active_colors["Bonds"]
        for i in range(len(supercell)):
            for j in range(i + 1, len(supercell)):
                a, b = supercell[i], supercell[j]
                dist = np.linalg.norm(a.coords - b.coords)
                ra = a.specie.atomic_radius if a.specie.atomic_radius else 1.2
                rb = b.specie.atomic_radius if b.specie.atomic_radius else 1.2
                if 0.5 < dist <= (ra + rb) * 1.35:
                    self.ax.plot([a.coords[0], b.coords[0]], [a.coords[1], b.coords[1]], [a.coords[2], b.coords[2]], 
                                 color=bond_color, linewidth=line_thick, picker=True)

    def draw_matplotlib_polyhedra(self, supercell):
        nn_finder = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, site in enumerate(supercell):
                if site.specie.symbol in ["Te", "O", "S", "Se"]: continue 
                try:
                    neighbors = nn_finder.get_nn_info(supercell, i)
                    pts = np.array([n['site'].coords for n in neighbors if tuple(n['image']) == (0,0,0)])
                    if len(pts) >= 4:
                        hull = ConvexHull(pts)
                        faces = [pts[simplex] for simplex in hull.simplices]
                        poly_color = self.active_colors.get(site.specie.symbol, "#008080")
                        poly = Poly3DCollection(faces, alpha=0.3, facecolor=poly_color, edgecolor='black', picker=True)
                        self.ax.add_collection3d(poly)
                except:
                    continue

    # ================= P Y V I S T A   R E N D E R =================
    def draw_pyvista(self, supercell):
        self.plotter.clear()
        self.plotter.set_background("#1e1e24")
        scale_mod = self.spin_radius.value()
        style = self.combo_style.currentIndex()

        for site in supercell:
            coords = site.coords
            radius = (site.specie.atomic_radius if site.specie.atomic_radius else 1.0) * scale_mod
            atom_color = self.active_colors.get(site.specie.symbol, "#008080")
            sphere = pv.Sphere(radius=radius, center=coords)
            self.plotter.add_mesh(sphere, color=atom_color, smooth_shading=not BASIC_GRAPHICS, pickable=True)

        if style == 0: self.draw_pyvista_bonds(supercell)
        elif style == 1: self.draw_pyvista_polyhedra(supercell)
        self.plotter.reset_camera()

    def draw_pyvista_bonds(self, supercell):
        cyl_radius = self.spin_bond_thick.value()
        bond_color = self.active_colors["Bonds"]
        for i in range(len(supercell)):
            for j in range(i + 1, len(supercell)):
                a, b = supercell[i], supercell[j]
                dist = np.linalg.norm(a.coords - b.coords)
                ra = a.specie.atomic_radius if a.specie.atomic_radius else 1.2
                rb = b.specie.atomic_radius if b.specie.atomic_radius else 1.2
                if 0.5 < dist <= (ra + rb) * 1.35:
                    vec = b.coords - a.coords
                    mid = a.coords + vec / 2.0
                    stick = pv.Cylinder(center=mid, direction=vec, radius=cyl_radius, height=dist)
                    self.plotter.add_mesh(stick, color=bond_color, smooth_shading=not BASIC_GRAPHICS, pickable=True)

    def draw_pyvista_polyhedra(self, supercell):
        nn_finder = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, site in enumerate(supercell):
                if site.specie.symbol in ["Te", "O", "S", "Se"]: continue
                try:
                    neighbors = nn_finder.get_nn_info(supercell, i)
                    pts = np.array([n['site'].coords for n in neighbors if tuple(n['image']) == (0,0,0)])
                    if len(pts) >= 4:
                        hull = ConvexHull(pts)
                        faces = np.column_stack((np.full(len(hull.simplices), 3), hull.simplices)).flatten()
                        poly = pv.PolyData(pts, faces)
                        poly_color = self.active_colors.get(site.specie.symbol, "#008080")
                        self.plotter.add_mesh(poly, color=poly_color, opacity=0.4, show_edges=True, pickable=True)
                except:
                    continue

if __name__ == '__main__':
    app = QApplication.instance()
    if not app: app = QApplication(sys.argv)
    window = TensorSpecApp()
    window.show()
    sys.exit(app.exec())