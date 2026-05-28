import os
import sys
import platform
import numpy as np

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
                               QSpinBox, QDoubleSpinBox, QFrame, QComboBox, QStackedWidget)
from PySide6.QtCore import Qt

from pymatgen.core import Structure

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

CPK_COLORS = {
    "H": "#FFFFFF", "C": "#333333", "N": "#2233FF", "O": "#FF2200",
    "V": "#999999",  # Vanadium: Grey
    "Te": "#FF8C00", # Tellurium: Dark Orange
    "Fe": "#E06633", "Cu": "#C88033", "Zn": "#7D80B0"
}

class TensorSpecApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorSpec - Ultimate Crystal Viewer")
        self.resize(1100, 700) 

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # ================= L E F T   P A N E L =================
        self.ui_panel = QVBoxLayout()
        
        self.btn_load = QPushButton("Load CIF File")
        self.btn_load.clicked.connect(self.load_file)
        self.ui_panel.addWidget(self.btn_load)
        
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ui_panel.addWidget(self.lbl_file)

        # Graphics Engine Toggle
        self.ui_panel.addWidget(QLabel("Graphics Backend:"))
        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["CPU Safe Mode (Matplotlib)", "GPU Fast Mode (PyVista)"])
        self.combo_backend.currentIndexChanged.connect(self.switch_backend)
        self.ui_panel.addWidget(self.combo_backend)

        # Supercell
        self.ui_panel.addWidget(QLabel("Supercell (X, Y, Z):"))
        self.spin_x = QSpinBox(); self.spin_x.setValue(1); self.spin_x.setMinimum(1)
        self.spin_y = QSpinBox(); self.spin_y.setValue(1); self.spin_y.setMinimum(1)
        self.spin_z = QSpinBox(); self.spin_z.setValue(1); self.spin_z.setMinimum(1)
        for spin in [self.spin_x, self.spin_y, self.spin_z]:
            self.ui_panel.addWidget(spin)

        # Global Radius Scale
        self.ui_panel.addWidget(QLabel("Global Radius Scale:"))
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0.1, 5.0)
        self.spin_radius.setValue(0.5) # Default to smaller atoms to see bonds better
        self.spin_radius.setSingleStep(0.1)
        self.ui_panel.addWidget(self.spin_radius)

        # Camera Angles
        self.ui_panel.addWidget(QLabel("Camera View (Azimuth, Elevation):"))
        self.spin_azim = QSpinBox(); self.spin_azim.setRange(-360, 360); self.spin_azim.setValue(45)
        self.spin_elev = QSpinBox(); self.spin_elev.setRange(-360, 360); self.spin_elev.setValue(30)
        self.ui_panel.addWidget(self.spin_azim)
        self.ui_panel.addWidget(self.spin_elev)

        # Draw Button
        self.btn_draw = QPushButton("Draw Structure")
        self.btn_draw.clicked.connect(self.draw_structure)
        self.btn_draw.setStyleSheet("background-color: #2b5c8f; font-weight: bold; color: white;")
        self.ui_panel.addWidget(self.btn_draw)

        # Save Button
        self.btn_save = QPushButton("Save High-Res Image")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setStyleSheet("background-color: #2e8b57; font-weight: bold; color: white;")
        self.ui_panel.addWidget(self.btn_save)

        self.ui_panel.addStretch()

        control_frame = QFrame()
        control_frame.setLayout(self.ui_panel)
        control_frame.setFixedWidth(250)
        self.main_layout.addWidget(control_frame)

        # ================= R I G H T   P A N E L (Stacked) =================
        self.viewer_stack = QStackedWidget()
        self.main_layout.addWidget(self.viewer_stack)

        # Widget 0: Matplotlib
        self.figure = Figure(facecolor="#1e1e24")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor("#1e1e24")
        self.ax.axis('off')
        self.viewer_stack.addWidget(self.canvas)

        # Widget 1: PyVista
        self.plotter = QtInteractor(self.central_widget)
        self.plotter.set_background("#1e1e24")
        self.viewer_stack.addWidget(self.plotter)

        self.current_structure = None

    def switch_backend(self, index):
        """Swaps the visible 3D viewer based on the dropdown."""
        self.viewer_stack.setCurrentIndex(index)
        self.draw_structure() # Redraw in the new backend

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "CIF files (*.cif)")
        if fname:
            self.lbl_file.setText(fname.split('/')[-1])
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.current_structure = Structure.from_file(fname)
            except Exception as e:
                self.lbl_file.setText("Error loading CIF")
                print(f"Failed to load: {e}")

    def draw_structure(self):
        if self.current_structure is None:
            return

        backend = self.combo_backend.currentIndex()
        supercell = self.current_structure * (self.spin_x.value(), self.spin_y.value(), self.spin_z.value())
        
        if backend == 0:
            self.draw_matplotlib(supercell)
        else:
            self.draw_pyvista(supercell)

    # ================= M A T P L O T L I B   R E N D E R =================
    def draw_matplotlib(self, supercell):
        self.ax.clear()
        self.ax.set_facecolor("#1e1e24")
        self.ax.axis('off')

        all_x, all_y, all_z = [], [], []
        present_elements = set()

        # Scale modifier from UI
        scale_mod = self.spin_radius.value()

        # Atoms
        for site in supercell:
            x, y, z = site.coords
            all_x.append(x); all_y.append(y); all_z.append(z)
            
            radius = (site.specie.atomic_radius if site.specie.atomic_radius else 1.0) * scale_mod
            symbol = site.specie.symbol
            present_elements.add(symbol)
            atom_color = CPK_COLORS.get(symbol, "#008080")
            
            self.ax.scatter(x, y, z, s=(radius * 800), c=atom_color, edgecolors='black', alpha=1.0, depthshade=True)

        # Bonds
        for i in range(len(supercell)):
            for j in range(i + 1, len(supercell)):
                site_a = supercell[i]
                site_b = supercell[j]
                dist = np.linalg.norm(site_a.coords - site_b.coords)
                
                # Increased bond tolerance to ensure Te sticks render
                r_a = site_a.specie.atomic_radius if site_a.specie.atomic_radius else 1.2
                r_b = site_b.specie.atomic_radius if site_b.specie.atomic_radius else 1.2
                max_bond_distance = (r_a + r_b) * 1.35 
                
                if 0.5 < dist <= max_bond_distance:
                    self.ax.plot([site_a.coords[0], site_b.coords[0]], 
                                 [site_a.coords[1], site_b.coords[1]], 
                                 [site_a.coords[2], site_b.coords[2]], 
                                 color="#d3d3d3", linewidth=4, zorder=1)

        # Lattice Box
        matrix = supercell.lattice.matrix
        corners = [i * matrix[0] + j * matrix[1] + k * matrix[2] for i in [0,1] for j in [0,1] for k in [0,1]]
        box_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
        for start, end in box_edges:
            self.ax.plot([corners[start][0], corners[end][0]], 
                         [corners[start][1], corners[end][1]], 
                         [corners[start][2], corners[end][2]], 
                         color="#888888", linewidth=1, linestyle='--')

        # Fix Proportions & Set Camera
        if all_x:
            max_range = np.array([max(all_x)-min(all_x), max(all_y)-min(all_y), max(all_z)-min(all_z)]).max() / 2.0
            mid_x = (max(all_x) + min(all_x)) * 0.5
            mid_y = (max(all_y) + min(all_y)) * 0.5
            mid_z = (max(all_z) + min(all_z)) * 0.5
            self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

        self.ax.view_init(elev=self.spin_elev.value(), azim=self.spin_azim.value())
        
        legend_text = "Legend:\n" + "\n".join([f"• {sym}: {CPK_COLORS.get(sym, 'Teal')}" for sym in present_elements])
        self.figure.text(0.02, 0.95, legend_text, color='white', fontsize=11, va='top')
        self.canvas.draw()

    # ================= P Y V I S T A   R E N D E R =================
    def draw_pyvista(self, supercell):
        self.plotter.clear()
        self.plotter.set_background("#1e1e24")
        present_elements = set()
        scale_mod = self.spin_radius.value()

        # Atoms
        for site in supercell:
            coords = site.coords
            radius = (site.specie.atomic_radius if site.specie.atomic_radius else 1.0) * scale_mod
            symbol = site.specie.symbol
            present_elements.add(symbol)
            atom_color = CPK_COLORS.get(symbol, "#008080")
            sphere = pv.Sphere(radius=radius, center=coords)
            self.plotter.add_mesh(sphere, color=atom_color, smooth_shading=not BASIC_GRAPHICS)

        # Bonds
        for i in range(len(supercell)):
            for j in range(i + 1, len(supercell)):
                site_a = supercell[i]
                site_b = supercell[j]
                dist = np.linalg.norm(site_a.coords - site_b.coords)
                r_a = site_a.specie.atomic_radius if site_a.specie.atomic_radius else 1.2
                r_b = site_b.specie.atomic_radius if site_b.specie.atomic_radius else 1.2
                max_bond_distance = (r_a + r_b) * 1.35
                
                if 0.5 < dist <= max_bond_distance:
                    vec = site_b.coords - site_a.coords
                    midpoint = site_a.coords + vec / 2.0
                    stick = pv.Cylinder(center=midpoint, direction=vec, radius=0.10, height=dist)
                    self.plotter.add_mesh(stick, color="#d3d3d3", smooth_shading=not BASIC_GRAPHICS)

        # Lattice Box
        matrix = supercell.lattice.matrix
        corners = [i * matrix[0] + j * matrix[1] + k * matrix[2] for i in [0,1] for j in [0,1] for k in [0,1]]
        box_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
        for start, end in box_edges:
            line = pv.Line(corners[start], corners[end])
            self.plotter.add_mesh(line, color="#888888", line_width=2)

        legend_text = "Legend:\n" + "\n".join([f"• {sym}: {CPK_COLORS.get(sym, 'Teal')}" for sym in present_elements])
        self.plotter.add_text(legend_text, position="upper_left", font_size=11, color="white")

        self.plotter.camera.azimuth = self.spin_azim.value()
        self.plotter.camera.elevation = self.spin_elev.value()
        self.plotter.reset_camera()

    # ================= S A V E   I M A G E =================
    def save_image(self):
        if self.current_structure is None:
            return
            
        fname, _ = QFileDialog.getSaveFileName(self, "Save Image", "crystal_render.png", "PNG Files (*.png)")
        if fname:
            backend = self.combo_backend.currentIndex()
            if backend == 0:
                # Save Matplotlib at 300 DPI for paper publication
                self.figure.savefig(fname, dpi=300, bbox_inches='tight', facecolor="#1e1e24")
            else:
                # Save PyVista
                self.plotter.screenshot(fname)
            print(f"Saved successfully to: {fname}")


if __name__ == '__main__':
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
        
    window = TensorSpecApp()
    window.show()
    sys.exit(app.exec())