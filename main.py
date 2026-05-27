import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel, QSpinBox)
from pyvistaqt import QtInteractor
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
import numpy as np
import pyvista as pv

class TensorSpecApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorSpec - Crystal Visualization")
        self.resize(1200, 800)

        # Main Widget and Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # 1. Left Panel (UI Controls)
        self.ui_panel = QVBoxLayout()
        self.main_layout.addLayout(self.ui_panel, stretch=1)

        # File loading UI
        self.btn_load = QPushButton("Load CIF File")
        self.btn_load.clicked.connect(self.load_file)
        self.ui_panel.addWidget(self.btn_load)
        
        self.lbl_file = QLabel("No file loaded")
        self.ui_panel.addWidget(self.lbl_file)

        # Supercell UI (Requirement 2 preview)
        self.ui_panel.addWidget(QLabel("Supercell (X, Y, Z):"))
        self.spin_x = QSpinBox(); self.spin_x.setValue(1)
        self.spin_y = QSpinBox(); self.spin_y.setValue(1)
        self.spin_z = QSpinBox(); self.spin_z.setValue(1)
        
        for spin in [self.spin_x, self.spin_y, self.spin_z]:
            self.ui_panel.addWidget(spin)

        self.btn_draw = QPushButton("Draw Structure")
        self.btn_draw.clicked.connect(self.draw_structure)
        self.ui_panel.addWidget(self.btn_draw)

        self.ui_panel.addStretch() # Pushes everything to the top

        # 2. Right Panel (PyVista 3D Renderer)
        self.plotter = QtInteractor(self.central_widget)
        self.main_layout.addWidget(self.plotter.interactor, stretch=4)
        
        # State variables
        self.current_structure = None

    def load_file(self):
        # Open file dialog
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "CIF files (*.cif)")
        if fname:
            self.lbl_file.setText(fname.split('/')[-1])
            # Parse with Pymatgen
            self.current_structure = Structure.from_file(fname)

    def draw_structure(self):
        if self.current_structure is None:
            return

        self.plotter.clear()

        # 1. Create Supercell
        supercell = self.current_structure * (self.spin_x.value(), self.spin_y.value(), self.spin_z.value())

        # 2. Draw Atoms
        for site in supercell:
            coords = site.coords
            radius = site.specie.atomic_radius if site.specie.atomic_radius else 0.5 
            
            sphere = pv.Sphere(radius=radius, center=coords)
            
            self.plotter.add_mesh(sphere, color="teal", 
                                  pbr=True, metallic=0.6, roughness=0.2, 
                                  smooth_shading=True)

        # 3. Draw Nearest Neighbor Sticks
        self.draw_sticks(supercell)

        # Render
        self.plotter.enable_anti_aliasing()
        self.plotter.reset_camera()

    def draw_sticks(self, supercell):
        """Calculates nearest neighbors and draws connecting cylinders."""
        # Initialize the nearest-neighbor finder
        nn_finder = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False)
        
        # Keep track of drawn bonds so we don't draw overlapping cylinders (A->B and B->A)
        drawn_bonds = set()

        for i, site in enumerate(supercell):
            try:
                # Get the nearest neighbors for atom 'i'
                neighbors = nn_finder.get_nn_info(supercell, i)
            except ValueError:
                # CrystalNN can occasionally fail on highly irregular structures; skip if so
                continue 

            for neighbor in neighbors:
                j = neighbor['site_index']
                
                # Create a unique ID for this bond
                bond_id = tuple(sorted([i, j]))
                if bond_id in drawn_bonds:
                    continue
                drawn_bonds.add(bond_id)

                # Extract coordinates
                coords_a = site.coords
                coords_b = neighbor['site'].coords

                # Vector math for the 3D cylinder
                vec = coords_b - coords_a
                length = np.linalg.norm(vec)
                midpoint = coords_a + vec / 2.0

                # Generate the cylinder mesh
                stick = pv.Cylinder(center=midpoint, direction=vec, radius=0.15, height=length)
                
                # Add to plotter with a glossy, metallic appearance
                self.plotter.add_mesh(stick, color="silver", 
                                      pbr=True, metallic=0.8, roughness=0.3, 
                                      smooth_shading=True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TensorSpecApp()
    window.show()
    sys.exit(app.exec())