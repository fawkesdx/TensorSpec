import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QFormLayout, QDoubleSpinBox, QPushButton, QGroupBox, 
                               QLabel, QSpinBox, QLineEdit, QMessageBox, QComboBox, QInputDialog)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import the core math engine
from tensorspec.core.dft_engine import ChinookTightBindingEngine
from tensorspec.core.workspace import global_workspace

class DFTSuite(QWidget):
    """
    Main UI Controller for the DFT Suite.
    Currently implements the Toy Tight Binding engine for Graphene.
    """
    def __init__(self, parent=None):
        print("open suite DFT")
        super().__init__(parent)
        self.setWindowTitle("TensorSpec - DFT & Tight Binding Suite")
        self.resize(900, 600)
        
        # Initialize Core Math Engine
        self.engine = ChinookTightBindingEngine()
        
        self._setup_ui()
        self._connect_signals()
        

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # --- Left Panel: Controls ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(380)
        
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
        

        # --- NEW: Dimension Toggle ---
        self.combo_k_mode = QComboBox()
        self.combo_k_mode.addItems(["1D High-Symmetry Path", "2D k-Mesh (kx, ky)"])
        tb_form.addRow("k-Space Grid:", self.combo_k_mode)

        # --- NEW: ARPES Isoenergy Control ---
        self.spin_iso = QDoubleSpinBox()
        self.spin_iso.setRange(-20.0, 20.0)
        self.spin_iso.setValue(0.0)
        self.spin_iso.setSingleStep(0.1)
        self.spin_iso.setEnabled(False) # Disabled by default since 1D is default
        tb_form.addRow("2D Isoenergy Cut (eV):", self.spin_iso)
        
        # Only enable the slider when in 2D mode
        self.combo_k_mode.currentTextChanged.connect(
            lambda text: self.spin_iso.setEnabled("2D" in text)
        )
        
        # --- NEW: K-Path Template Toggle ---
        self.combo_k_template = QComboBox()
        self.combo_k_template.addItems([
            "Auto-Detect BZ Path (PyMatgen)",
            "Arbitrary Custom Path",
            "Hexagonal (Template)", 
            "Rectangular / Orthorhombic (Template)", 
            "Square / Tetragonal (Template)"
        ])
        tb_form.addRow("1D Path Template:", self.combo_k_template)

        # --- NEW: Spin-Orbit Coupling Controls ---
        soc_layout = QHBoxLayout()
        from PySide6.QtWidgets import QCheckBox # Ensure this is imported at top
        self.chk_soc = QCheckBox("Enable Spin-Orbit Coupling")
        self.chk_soc.setStyleSheet("font-weight: bold; color: #8A2BE2;")
        
        self.spin_soc_val = QDoubleSpinBox()
        self.spin_soc_val.setRange(0.0, 5.0)
        self.spin_soc_val.setValue(0.5)
        self.spin_soc_val.setSingleStep(0.1)
        self.spin_soc_val.setSuffix(" eV")
        self.spin_soc_val.setToolTip("SOC Strength (\u03BB)")
        
        soc_layout.addWidget(self.chk_soc)
        soc_layout.addWidget(self.spin_soc_val)
        tb_form.addRow("Relativistic:", soc_layout)

        # --- NEW: Fat Band Projection Controls ---
        self.combo_projection = QComboBox()
        self.combo_projection.addItem("None (Standard Lines)")
        tb_form.addRow("Fat Band Target:", self.combo_projection)
        
        # Disable the value box unless the checkbox is ticked
        self.spin_soc_val.setEnabled(False)
        self.chk_soc.stateChanged.connect(lambda state: self.spin_soc_val.setEnabled(bool(state)))

        # --- NEW: Wannier90 Importer ---
        w90_group = QGroupBox("Ab Initio Import (Wannier90)")
        w90_layout = QVBoxLayout(w90_group)
        
        from PySide6.QtWidgets import QFileDialog # Ensure this is imported at top
        self.btn_load_w90 = QPushButton("📂 Load wannier90_hr.dat")
        self.btn_load_w90.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold;")
        self.lbl_w90_status = QLabel("Status: Using Manual Slater-Koster parameters.")
        self.lbl_w90_status.setStyleSheet("color: gray; font-size: 10px;")
        
        w90_layout.addWidget(self.btn_load_w90)
        w90_layout.addWidget(self.lbl_w90_status)
        tb_form.addRow(w90_group)
        
        # State variable to track if a file is loaded
        self.active_w90_file = None
        self.btn_load_w90.clicked.connect(self.load_w90_file)
        
        # --- NEW: Custom Arbitrary Path Inputs ---
        self.custom_k_widget = QWidget()
        custom_layout = QFormLayout(self.custom_k_widget)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        
        self.line_k_coords = QLineEdit("0,0,0; 0.5,0,0; 0.5,0.5,0; 0,0,0")
        self.line_k_coords.setToolTip("Format: x1,y1,z1 ; x2,y2,z2 ; ...")
        self.line_k_labels = QLineEdit("G; X; M; G")
        self.line_k_labels.setToolTip("Format: Label1 ; Label2 ; ...")
        
        # --- NEW: Hamiltonian Mode Toggle ---
        self.combo_tb_mode = QComboBox()
        self.combo_tb_mode.addItems(["Simple Scalar (Isotropic)", "Slater-Koster (Rigorous)"])
        tb_form.addRow("Hamiltonian Mode:", self.combo_tb_mode)

        custom_layout.addRow("Coords (frac):", self.line_k_coords)
        custom_layout.addRow("Labels:", self.line_k_labels)
        tb_form.addRow(self.custom_k_widget)
        
        # Hide custom inputs unless "Arbitrary" is selected
        self.custom_k_widget.setVisible(False)
        self.combo_k_template.currentTextChanged.connect(
            lambda text: self.custom_k_widget.setVisible("Arbitrary" in text)
        )
        
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
        self.spin_t1 = QDoubleSpinBox(); self.spin_t1.setRange(-10.0, 10.0); self.spin_t1.setValue(2.7); self.spin_t1.setMinimumWidth(65)
        self.spin_r1 = QDoubleSpinBox(); self.spin_r1.setRange(0.0, 10.0); self.spin_r1.setValue(1.6); self.spin_r1.setMinimumWidth(65)
        
        self.spin_t2 = QDoubleSpinBox(); self.spin_t2.setRange(-10.0, 10.0); self.spin_t2.setValue(0.0); self.spin_t2.setMinimumWidth(65)
        self.spin_r2 = QDoubleSpinBox(); self.spin_r2.setRange(0.0, 10.0); self.spin_r2.setValue(2.6); self.spin_r2.setMinimumWidth(65)
        
        self.spin_t3 = QDoubleSpinBox(); self.spin_t3.setRange(-10.0, 10.0); self.spin_t3.setValue(0.0); self.spin_t3.setMinimumWidth(65)
        self.spin_r3 = QDoubleSpinBox(); self.spin_r3.setRange(0.0, 10.0); self.spin_r3.setValue(3.1); self.spin_r3.setMinimumWidth(65)
        
        # Add Interlayer Hopping Controls
        self.spin_t4 = QDoubleSpinBox(); self.spin_t4.setRange(-10.0, 10.0); self.spin_t4.setValue(-0.3); self.spin_t4.setMinimumWidth(65)
        self.spin_r4 = QDoubleSpinBox(); self.spin_r4.setRange(0.0, 15.0); self.spin_r4.setValue(4.5); self.spin_r4.setMinimumWidth(65)
        
        
        # Create compact rows for each shell: [ t_value | max_dist ]
        shells = [
            (self.spin_t1, self.spin_r1), 
            (self.spin_t2, self.spin_r2), 
            (self.spin_t3, self.spin_r3),
            (self.spin_t4, self.spin_r4)  # <-- Added t4 to the layout loop
        ]
        for i, (t_spin, r_spin) in enumerate(shells, start=1):
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
        
        # --- NEW: Connect spin box for live plotting updates ---
        self.spin_iso.valueChanged.connect(self.update_2d_plot)

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
            formula = self.engine.crystal_structure.composition.reduced_formula
            hopping = self.engine.get_default_hopping(formula)
            
            self.active_hopping_keys = list(hopping.keys())
            vals = list(hopping.values())
            
            # --- UPDATED: Pre-fill the UI SpinBoxes safely including t4 ---
            if len(vals) > 0: self.spin_t1.setValue(vals[0])
            if len(vals) > 1: self.spin_t2.setValue(vals[1])
            if len(vals) > 2: self.spin_t3.setValue(vals[2])
            if len(vals) > 3: self.spin_t4.setValue(vals[3]) # <-- Map 4th value to spin_t4
            else: self.spin_t4.setValue(0.0) # Reset to 0 if material has no interlayer default
            
            # --- NEW: Populate Projection Dropdown ---
            self.combo_projection.clear()
            self.combo_projection.addItem("None (Standard Lines)")
            
            # Pull explicit orbital strings directly from the engine's basis rules
            if self.engine.crystal_structure:
                for site in self.engine.crystal_structure:
                    elem = site.species_string
                    for orb_str in self.engine._get_orbital_basis(elem):
                        # Convert Chinook's strings into proper physics labels
                        if orb_str.endswith("0"):
                            orb_name = "s"
                        elif orb_str[1] == "1":
                            orb_name = "p" + orb_str[2:]
                        elif orb_str[1] == "2":
                            # Prettify the weird Chinook abbreviations for d-orbitals
                            if orb_str.endswith("ZR"):
                                orb_name = "dz2"
                            elif orb_str.endswith("XY"):
                                orb_name = "dx2-y2"
                            else:
                                orb_name = "d" + orb_str[2:]
                        else:
                            orb_name = "unknown"
                        label = f"{elem}_{orb_name}"
                        
                        # Prevent duplicates in the dropdown
                        existing_items = [self.combo_projection.itemText(i) for i in range(self.combo_projection.count())]
                        if label not in existing_items:
                            self.combo_projection.addItem(label)
            
            QMessageBox.information(self, "Success", f"Loaded '{target}' ({formula}) into DFT engine.")
        else:
            QMessageBox.critical(self, "Error", "Failed to load structure.")

    def calculate_bands(self):
        is_2d = hasattr(self, 'combo_k_mode') and self.combo_k_mode.currentIndex() == 1
        
        try:
            # 1. Generate k-points
            if not is_2d:
                template_name = self.combo_k_template.currentText()
                
                # Route to the correct path generator
                if "Auto-Detect" in template_name:
                    k_points, k_labels = self.engine.get_auto_kpath()
                elif "Arbitrary" in template_name:
                    k_points, k_labels = self.engine.get_custom_kpath(self.line_k_coords.text(), self.line_k_labels.text())
                else:
                    # Fallback to old basic templates
                    temp_key = template_name.split()[0].lower()
                    lattice_a, lattice_b = 3.0, 3.0
                    if hasattr(self.engine.crystal_structure, 'lattice'):
                        lattice_a = self.engine.crystal_structure.lattice.a
                        lattice_b = self.engine.crystal_structure.lattice.b
                    k_points, k_labels = self.engine.get_kpath_template(temp_key, a=lattice_a, b=lattice_b)

                # --- RESTORED FIX: Convert fractional k-points to Cartesian ---
                if hasattr(self.engine.crystal_structure, 'lattice'):
                    recip_matrix = self.engine.crystal_structure.lattice.reciprocal_lattice.matrix
                    k_points = np.dot(k_points, recip_matrix)

                # Interpolate the path with the requested resolution
                k_vecs, k_dist, node_idx, labels = self.engine.generate_k_path(
                    k_points, k_labels, points_per_segment=self.spin_k_res.value()
                )
            else:
                self.res = self.spin_k_res.value() 
                self.kx_vals = np.linspace(-4.5, 4.5, self.res)
                self.ky_vals = np.linspace(-4.5, 4.5, self.res)
                Kx, Ky = np.meshgrid(self.kx_vals, self.ky_vals)
                k_vecs = np.column_stack([Kx.ravel(), Ky.ravel(), np.zeros_like(Kx.ravel())])
                k_dist, node_idx, labels = None, None, None

            # 2. Re-pack UI values into the custom hopping dictionary
            custom_hopping = {}
            if hasattr(self, 'active_hopping_keys'):
                keys = self.active_hopping_keys
                if len(keys) > 0: custom_hopping[keys[0]] = self.spin_t1.value()
                if len(keys) > 1: custom_hopping[keys[1]] = self.spin_t2.value()
                if len(keys) > 2: custom_hopping[keys[2]] = self.spin_t3.value()
                if len(keys) > 3: custom_hopping[keys[3]] = self.spin_t4.value()

            # 3. Route to solver using custom physical shells or Wannier90
            is_soc = self.chk_soc.isChecked()
            soc_val = self.spin_soc_val.value()
            
            # Pass the active Wannier90 filepath if one has been loaded by the user
            w90_file = getattr(self, 'active_w90_file', None)

            # Gather cutoffs from UI SpinBoxes
            ui_cutoffs = [
                self.spin_r1.value(),
                self.spin_r2.value(),
                self.spin_r3.value(),
                self.spin_r4.value()
            ]
            
            eigenvalues, eigenvectors, orb_labels = self.engine.solve_bands(
                k_vecs,
                custom_hopping=custom_hopping,
                onsite_e=self.spin_onsite.value(),
                use_soc=is_soc,
                soc_strength=soc_val,
                w90_filepath=w90_file,
                cutoffs=ui_cutoffs,
                tb_mode=self.combo_tb_mode.currentText() # <--- NEW PARAMETER
            )
            
            # Update title text based on mode
            soc_title_tag = " (with SOC)" if is_soc and not w90_file else ""
            mode_tag = "Wannier90" if w90_file else "Chinook"
            
            title = f"{mode_tag} 2D Mesh" if is_2d else f"{mode_tag} Bands ({template_name.split()[0]} Path){soc_title_tag}"
            
        except Exception as e:
            QMessageBox.warning(self, "Calculation Error", str(e))
            return

        # --- NEW: Extract Basis Coordinates for ARPES Matrix Elements ---
        basis_coords = []
        if hasattr(self.engine, 'crystal_structure') and self.engine.crystal_structure is not None:
            # Extract Cartesian coordinates of the atoms in the unit cell
            # For a simple graphene pz-orbital model, the 2 atoms map perfectly to the 2 basis states.
            basis_coords = [site.coords.tolist() for site in self.engine.crystal_structure]

        # --- NEW: Ultra-Deep Hunt for Chinook objects ---
        found_basis = None
        found_h_dict = None
        found_tb_model = None
        
        for attr_name in dir(self.engine):
            if attr_name.startswith('__'): continue
            attr_val = getattr(self.engine, attr_name)
            type_name = type(attr_val).__name__
            
            # Catch the pre-built model
            if 'TB_model' in type_name:
                found_tb_model = attr_val
            
            # Catch the Hamiltonian dict (or if basis is nested inside a dict)
            elif isinstance(attr_val, dict):
                if 'ham' in attr_name.lower() or 'dict' in attr_name.lower():
                    found_h_dict = attr_val
                if 'basis' in attr_val:
                    found_basis = attr_val['basis']
                    
            # Catch the basis by checking list/tuple/array contents
            elif isinstance(attr_val, (list, tuple)) or type_name == 'ndarray':
                try:
                    if len(attr_val) > 0 and 'orbital' in type(attr_val[0]).__name__.lower():
                        found_basis = attr_val
                except Exception:
                    pass
            
            # Fallback direct name matching for basis
            if 'basis' in attr_name.lower() and found_basis is None:
                found_basis = attr_val

        # If tb_model was found but basis/h_dict wasn't, extract them directly from the model
        if found_tb_model is not None:
            if found_basis is None: found_basis = getattr(found_tb_model, 'basis', None)
            if found_h_dict is None: found_h_dict = getattr(found_tb_model, 'H_dict', None)

        # 4. Cache data for pushing
        self.active_bands_data = {
            'type': 'band_structure',
            'is_2d': is_2d,
            'k_vecs': k_vecs,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'orbital_positions': basis_coords, 
            'basis': found_basis,
            'H_dict': found_h_dict,
            'tb_model': found_tb_model,
            'title': title
        }
        
        print(f">> DEBUG: Deep Hunt | tb_model: {type(found_tb_model)} | basis: {type(found_basis)} | H_dict: {type(found_h_dict)}")
        if is_2d:
            self.active_bands_data.update({'kx': self.kx_vals, 'ky': self.ky_vals, 'grid_shape': (self.res, self.res)})
        else:
            self.active_bands_data.update({'k_dist': k_dist, 'node_idx': node_idx, 'labels': labels})
        self.btn_push_bands.setEnabled(True)

        # 5. Render Plot
        # SAFELY remove the colorbar from C++ memory before clearing the figure
        if hasattr(self, 'cbar') and self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
            
        self.figure.clear() 
        
        if not is_2d:
            self.ax = self.figure.add_subplot(111)
            num_bands = eigenvalues.shape[1]
            projection_mode = self.combo_projection.currentText()
            
            if projection_mode != "None (Standard Lines)":
                target_el = projection_mode.replace("Element: ", "")
                target_indices = [i for i, lbl in enumerate(orb_labels) if lbl.startswith(target_el)]
                
                if target_indices:
                    probs = np.abs(eigenvectors)**2
                    if probs.shape[1] == len(orb_labels):
                        weights = np.sum(probs[:, target_indices, :], axis=1) 
                    else:
                        weights = np.sum(probs[:, :, target_indices], axis=2)
                        
                    x = np.tile(k_dist, (num_bands, 1)).T.flatten()
                    y = eigenvalues.flatten()
                    c = weights.flatten()
                    
                    scatter = self.ax.scatter(x, y, c=c, cmap='coolwarm', s=8, zorder=2, vmin=0, vmax=1)
                    self.cbar = self.figure.colorbar(scatter, ax=self.ax)
                    self.cbar.set_label(f"Orbital Character (Red = {target_el})", fontsize=10)
            else:
                for b in range(num_bands):
                    self.ax.plot(k_dist, eigenvalues[:, b], color='blue', linewidth=2)
            
            self.ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=1)
            self.ax.set_xlim(0, k_dist[-1])
            self.ax.set_xticks([k_dist[i] for i in node_idx])
            self.ax.set_xticklabels(labels, fontsize=14)
            self.ax.set_ylabel("Energy (eV)", fontsize=12)
            for i in node_idx:
                self.ax.axvline(k_dist[i], color='black', linewidth=0.8, zorder=1)
                
            self.ax.set_title(title, fontsize=14)
            self.figure.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.15)
            self.canvas.draw()
            
        else:
            self.update_2d_plot()

    def update_2d_plot(self):
        """Redraws the 2D isoenergy cut instantly without recalculating bands."""
        if not hasattr(self, 'active_bands_data') or not self.active_bands_data.get('is_2d'):
            return
            
        if hasattr(self, 'cbar') and self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
            
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        
        eigenvalues = self.active_bands_data['eigenvalues']
        num_bands = eigenvalues.shape[1]
        res = self.active_bands_data['grid_shape'][0]
        
        omega = self.spin_iso.value()
        eta = 0.1 
        
        spectral_weight = np.zeros((res, res))
        for b in range(num_bands):
            band_energy = eigenvalues[:, b].reshape((res, res))
            spectral_weight += (eta / np.pi) / ((omega - band_energy)**2 + eta**2)
        
        im = self.ax.pcolormesh(self.active_bands_data['kx'], self.active_bands_data['ky'], spectral_weight, cmap='magma', shading='auto')
        
        self.ax.set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)", fontsize=12)
        self.ax.set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)", fontsize=12)
        
        title = self.active_bands_data.get('title', "2D Mesh")
        self.ax.set_title(f"{title}\nIsoenergy Cut at {omega:.2f} eV", fontsize=14)
        self.ax.set_aspect('equal') 
        
        self.cbar = self.figure.colorbar(im, ax=self.ax)
        self.cbar.set_label("Spectral Weight (A.U.)", fontsize=10)
        
        self.figure.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
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
    
    def load_w90_file(self):
        """Opens a file dialog to load the Wannier90 hopping data."""
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Wannier90 HR File', '', "Data files (*.dat);;All files (*.*)")
        if fname:
            self.active_w90_file = fname
            filename_short = fname.split('/')[-1]
            self.lbl_w90_status.setText(f"Status: Using {filename_short}")
            self.lbl_w90_status.setStyleSheet("color: blue; font-weight: bold;")
            
            # Disable manual t1, t2, t3 boxes so the user knows they are overridden
            self.spin_t1.setEnabled(False)
            self.spin_t2.setEnabled(False)
            self.spin_t3.setEnabled(False)

    def closeEvent(self, event):
        print("close suite DFT Suite")


# Standalone runner for independent testing
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DFTSuite()
    window.show()
    sys.exit(app.exec())