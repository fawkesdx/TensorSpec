import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QFormLayout, QDoubleSpinBox, QPushButton, QGroupBox, 
                               QLabel, QSpinBox)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import the core math engine
from tensorspec.core.dft_engine import TightBindingEngine
from tensorspec.core.workspace import global_workspace
from PySide6.QtWidgets import QMessageBox, QComboBox, QInputDialog

class DFTSuite(QWidget):
    """
    Main UI Controller for the DFT Suite.
    Currently implements the Toy Tight Binding engine for Graphene.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TensorSpec - DFT & Tight Binding Suite")
        self.resize(900, 600)
        
        # Initialize Core Math Engine
        self.engine = TightBindingEngine()
        
        self._setup_ui()
        self._connect_signals()
        
        # Trigger an initial calculation so the canvas is not blank
        self.calculate_bands()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # --- Left Panel: Controls ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(320)
        
        # 1. Workspace Integration Panel
        ws_group = QGroupBox("Crystal Structure (Workspace)")
        ws_layout = QVBoxLayout(ws_group)
        
        row1 = QHBoxLayout()
        self.ws_combo = QComboBox()
        self.ws_combo.addItem("No structures available")
        self.btn_ws_refresh = QPushButton("🔄 Refresh")
        row1.addWidget(self.ws_combo)
        row1.addWidget(self.btn_ws_refresh)
        
        self.btn_ws_load = QPushButton("📥 Load Basis into Engine")
        ws_layout.addLayout(row1)
        ws_layout.addWidget(self.btn_ws_load)
        
        # 2. Tight Binding Parameters
        tb_group = QGroupBox("Tight Binding Parameters")
        tb_form = QFormLayout(tb_group)
        
        self.combo_solver = QComboBox()
        self.combo_solver.addItems(["Toy Graphene Model (Built-in)", "Workspace Structure"])
        tb_form.addRow("Solver Mode:", self.combo_solver)

        # --- NEW: Dimension Toggle ---
        self.combo_k_mode = QComboBox()
        self.combo_k_mode.addItems(["1D High-Symmetry Path", "2D k-Mesh (kx, ky)"])
        tb_form.addRow("k-Space Grid:", self.combo_k_mode)
        
        # --- NEW: K-Path Template Toggle ---
        self.combo_k_template = QComboBox()
        self.combo_k_template.addItems(["Hexagonal", "Rectangular / Orthorhombic", "Square / Tetragonal"])
        tb_form.addRow("1D Path Template:", self.combo_k_template)
        
        # --- NEW: Resolution Control ---
        self.spin_k_res = QSpinBox()
        self.spin_k_res.setRange(10, 2000)
        self.spin_k_res.setValue(100)
        self.spin_k_res.setSingleStep(50)
        tb_form.addRow("Points per Segment:", self.spin_k_res)
        
        self.spin_onsite = QDoubleSpinBox()
        self.spin_onsite.setRange(-10.0, 10.0); self.spin_onsite.setValue(0.0); self.spin_onsite.setSingleStep(0.1)
        tb_form.addRow("On-site E (eV):", self.spin_onsite)

        # Hopping Shells (t1, t2, t3)
        self.spin_t1 = QDoubleSpinBox(); self.spin_t1.setRange(-10.0, 10.0); self.spin_t1.setValue(2.7)
        self.spin_r1 = QDoubleSpinBox(); self.spin_r1.setRange(0.0, 10.0); self.spin_r1.setValue(1.6)
        
        self.spin_t2 = QDoubleSpinBox(); self.spin_t2.setRange(-10.0, 10.0); self.spin_t2.setValue(0.0)
        self.spin_r2 = QDoubleSpinBox(); self.spin_r2.setRange(0.0, 10.0); self.spin_r2.setValue(2.6)
        
        self.spin_t3 = QDoubleSpinBox(); self.spin_t3.setRange(-10.0, 10.0); self.spin_t3.setValue(0.0)
        self.spin_r3 = QDoubleSpinBox(); self.spin_r3.setRange(0.0, 10.0); self.spin_r3.setValue(3.1)

        # Create compact rows for each shell: [ t_value | max_dist ]
        for i, (t_spin, r_spin) in enumerate([(self.spin_t1, self.spin_r1), 
                                              (self.spin_t2, self.spin_r2), 
                                              (self.spin_t3, self.spin_r3)], start=1):
            row = QHBoxLayout()
            row.addWidget(t_spin)
            row.addWidget(QLabel("Max Å:"))
            row.addWidget(r_spin)
            tb_form.addRow(f"Hopping t{i}:", row)
        
        self.btn_calculate = QPushButton("⚙️ Calculate Band Structure")
        self.btn_calculate.setStyleSheet("background-color: #2b5c8f; color: white; font-weight: bold; padding: 8px;")
        
        # --- NEW: Push Bands Button ---
        self.btn_push_bands = QPushButton("📤 Push Bands to Workspace")
        self.btn_push_bands.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold; padding: 8px;")
        self.btn_push_bands.setEnabled(False) # Disabled until calculation is done
        
        control_layout.addWidget(ws_group)
        control_layout.addWidget(tb_group)
        control_layout.addWidget(self.btn_calculate)
        control_layout.addWidget(self.btn_push_bands)
        control_layout.addStretch()
        
        # --- Right Panel: Band Structure Canvas ---
        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.canvas, stretch=1)

    def _connect_signals(self):
        self.btn_calculate.clicked.connect(self.calculate_bands)
        self.btn_ws_refresh.clicked.connect(self.refresh_workspace_list)
        self.btn_ws_load.clicked.connect(self.load_workspace_structure)
        self.btn_push_bands.clicked.connect(self.push_bands_to_workspace)

    def refresh_workspace_list(self):
        self.ws_combo.clear()
        structures = global_workspace.list_crystal_structures()
        if structures:
            self.ws_combo.addItems(structures)
        else:
            self.ws_combo.addItem("No structures available")

    def load_workspace_structure(self):
        target = self.ws_combo.currentText()
        if not target or target == "No structures available":
            QMessageBox.warning(self, "Warning", "No valid structure selected.")
            return
        if self.engine.load_structure_from_workspace(target):
            QMessageBox.information(self, "Success", f"Loaded '{target}' into DFT engine.")
        else:
            QMessageBox.critical(self, "Error", "Failed to load structure.")

    def calculate_bands(self):
        onsite_val = self.spin_onsite.value()
        solver_mode = self.combo_solver.currentText()
        is_2d = hasattr(self, 'combo_k_mode') and self.combo_k_mode.currentIndex() == 1
        
        # 1. Generate k-points
        if not is_2d:
            template_name = self.combo_k_template.currentText().split()[0].lower()
            
            # Match the lattice constants to the solver
            if solver_mode == "Toy Graphene Model (Built-in)":
                lattice_a = np.sqrt(3)
                lattice_b = np.sqrt(3)
            else:
                # Dynamically extract 'a' and 'b' from the loaded PyMatgen structure
                if hasattr(self.engine.crystal_structure, 'lattice'):
                    lattice_a = self.engine.crystal_structure.lattice.a
                    lattice_b = self.engine.crystal_structure.lattice.b
                    
                    # --- NEW: Dummy Canvas Bypass ---
                    # If the structure is sitting in a giant 500A vacuum box, 
                    # shrink the k-space boundaries back to the physical WTe2 size!
                    if lattice_a >= 400.0:
                        lattice_a = 3.477
                        lattice_b = 6.249
                else:
                    lattice_a = 3.47 
                    lattice_b = 6.24
                    
            k_points, k_labels = self.engine.get_kpath_template(template_name, a=lattice_a, b=lattice_b)
            k_vecs, k_dist, node_idx, labels = self.engine.generate_k_path(
                k_points, k_labels, points_per_segment=self.spin_k_res.value()
            )
        else:
            self.res = 150 # 2D Grid resolution
            self.kx_vals = np.linspace(-4.5, 4.5, self.res)
            self.ky_vals = np.linspace(-4.5, 4.5, self.res)
            Kx, Ky = np.meshgrid(self.kx_vals, self.ky_vals)
            k_vecs = np.column_stack([Kx.ravel(), Ky.ravel(), np.zeros_like(Kx.ravel())])
            k_dist, node_idx, labels = None, None, None

        # 2. Route to solver
        try:
            if solver_mode == "Workspace Structure":
                eigenvalues, eigenvectors, orb_labels = self.engine.solve_workspace_structure(k_vecs)
                title = f"Multi-Orbital Bands ({template_name.capitalize()} Path)"
            else:
                t_val = self.spin_t1.value()
                eigenvalues, eigenvectors, orb_labels = self.engine.solve_toy_graphene(k_vecs, t=t_val, onsite=onsite_val)
                title = f"Toy Graphene Bands (t={t_val} eV)"
        except Exception as e:
            QMessageBox.warning(self, "Calculation Error", str(e))
            return

        # 3. Cache data for pushing
        self.active_bands_data = {
            'type': 'band_structure',
            'is_2d': is_2d,
            'k_vecs': k_vecs,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
        }
        if is_2d:
            self.active_bands_data.update({'kx': self.kx_vals, 'ky': self.ky_vals, 'grid_shape': (self.res, self.res)})
        else:
            self.active_bands_data.update({'k_dist': k_dist, 'node_idx': node_idx, 'labels': labels})
        self.btn_push_bands.setEnabled(True)

        # 4. Render Plot
        self.ax.clear()
        if not is_2d:
            num_bands = eigenvalues.shape[1]
            
            # --- NEW: Orbital Projection (Fat Bands) ---
            unique_elements = list(set([lbl.split('_')[0] for lbl in orb_labels]))
            
            if len(unique_elements) >= 2:
                # Calculate probability weight |psi|^2 for the first element (e.g., 'W')
                el1 = unique_elements[0] 
                el1_indices = [i for i, lbl in enumerate(orb_labels) if lbl.startswith(el1)]
                
                probs = np.abs(eigenvectors)**2
                weights = np.sum(probs[:, el1_indices, :], axis=1) # Shape: (N_k, N_bands)
                
                # Flatten arrays for matplotlib scatter mapping
                x = np.tile(k_dist, (num_bands, 1)).T.flatten()
                y = eigenvalues.flatten()
                c = weights.flatten()
                
                # Draw the scatter plot using a Red/Blue colormap
                scatter = self.ax.scatter(x, y, c=c, cmap='coolwarm', s=8, zorder=2, vmin=0, vmax=1)
                
                if not hasattr(self, 'cbar'):
                    self.cbar = self.figure.colorbar(scatter, ax=self.ax)
                else:
                    self.cbar.update_normal(scatter)
                self.cbar.set_label(f"Orbital Character (Red={el1}, Blue=Other)", fontsize=10)
                
            else:
                # Fallback to standard blue lines for single-element materials (Graphene)
                for b in range(num_bands):
                    self.ax.plot(k_dist, eigenvalues[:, b], color='blue', linewidth=2)
                if hasattr(self, 'cbar'):
                    self.cbar.remove()
                    del self.cbar
            
            # Formatting
            self.ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=1)
            self.ax.set_xlim(0, k_dist[-1])
            self.ax.set_xticks([k_dist[i] for i in node_idx])
            self.ax.set_xticklabels(labels, fontsize=14)
            self.ax.set_ylabel("Energy (eV)", fontsize=12)
            
            for i in node_idx:
                self.ax.axvline(k_dist[i], color='black', linewidth=0.8, zorder=1)
                
            self.ax.legend(loc='upper right', fontsize=10)
        else:
            # Plot the top valence band (band index 0) for the 2D map
            valence_band = eigenvalues[:, 0].reshape((self.res, self.res))
            self.ax.pcolormesh(self.kx_vals, self.ky_vals, valence_band, shading='auto', cmap='viridis')
            self.ax.set_aspect('equal')
            self.ax.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)", fontsize=12)
            self.ax.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)", fontsize=12)

        self.ax.set_title(title, fontsize=14)
        self.figure.tight_layout()
        self.canvas.draw()
    
    def push_bands_to_workspace(self):
        if not hasattr(self, 'active_bands_data'):
            QMessageBox.warning(self, "Warning", "No band structure calculated yet.")
            return
            
        # Determine default name based on dimensionality
        dim_str = "2D" if self.active_bands_data.get('is_2d') else "1D"
        default_name = f"TB_Bands_{dim_str}"
        
        # Pop up a dialog asking the user to name the variable
        name, ok = QInputDialog.getText(self, "Save to Workspace", 
                                        "Enter variable name for workspace:", 
                                        text=default_name)
        
        # If the user clicks Cancel or enters nothing, abort the save
        if not ok or not name.strip():
            return
            
        name = name.strip()
        
        # Directly inject the data dictionary into the workspace
        global_workspace._data[name] = self.active_bands_data
        
        dim_str_display = "2D Mesh" if self.active_bands_data.get('is_2d') else "1D Path"
        QMessageBox.information(self, "Success", f"Band structure '{name}' ({dim_str_display}) pushed to Workspace!\nYou can now load it in the ARPES Suite.")


# Standalone runner for independent testing
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DFTSuite()
    window.show()
    sys.exit(app.exec())