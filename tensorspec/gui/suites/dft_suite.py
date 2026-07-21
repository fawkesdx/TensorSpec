import sys
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QGroupBox, QComboBox, QMessageBox, 
                               QInputDialog, QSplitter, QScrollArea, QFileDialog)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import the core math router and workspace
from tensorspec.core.dft_engine import DFTEngineRouter
from tensorspec.core.workspace import global_workspace

# Import the isolated UI components
from tensorspec.gui.components.dft_panels import TightBindingPanel
# Make sure to create this file/class based on our earlier QE Generator step!
from tensorspec.gui.components.qe_generator_panel import QEGeneratorPanel

class DFTSuite(QWidget):
    """
    Main UI Coordinator for the DFT Suite.
    Manages layout and plots, delegating physics to DFTEngineRouter 
    and UI inputs to modular panels.
    """
    def __init__(self, parent=None):
        print("open suite DFT")
        super().__init__(parent)
        self.setWindowTitle("TensorSpec - DFT & Tight Binding Suite")
        self.resize(900, 700)
        
        # Initialize Core Math Engine Router
        self.engine = DFTEngineRouter()
        
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # --- Left Panel: Controls ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        # (setFixedWidth removed to allow the QSplitter to resize horizontally)
        
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
        control_layout.addWidget(ws_group)

        # 2. Modular QE Generator Panel
        self.qe_panel = QEGeneratorPanel(self.engine)
        control_layout.addWidget(self.qe_panel)
        
        # 3. Modular Tight Binding Panel
        self.tb_panel = TightBindingPanel()
        control_layout.addWidget(self.tb_panel)
        
        # Action Buttons
        self.btn_calculate = QPushButton("⚙️ Calculate Band Structure")
        self.btn_calculate.setStyleSheet("background-color: #2b5c8f; color: white; font-weight: bold; padding: 8px;")
        
        self.btn_push_bands = QPushButton("📤 Push Bands to Workspace")
        self.btn_push_bands.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold; padding: 8px;")
        self.btn_push_bands.setEnabled(False) # Disabled until calculation is done
        
        # --- NEW: Ab-Initio Plotter Button ---
        self.btn_load_qe_bands = QPushButton("📊 Load QE Ab-Initio Bands (XML)")
        self.btn_load_qe_bands.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 8px;")
        
        control_layout.addWidget(self.btn_calculate)
        control_layout.addWidget(self.btn_push_bands)
        control_layout.addWidget(self.btn_load_qe_bands)
        control_layout.addStretch()
        
        # --- Right Panel: Band Structure Canvas ---
        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Wrap the control panel in a scroll area to fix the vertical lock
        scroll_area = QScrollArea()
        scroll_area.setWidget(control_panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(320) # Prevents the panel from being crushed completely

        # Create the Draggable Splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(scroll_area) # Add the scroll area instead of the raw panel
        main_splitter.addWidget(self.canvas)
        
        # Set initial width ratio (380px for the left panel, the rest for the canvas)
        main_splitter.setSizes([380, 520]) 
        
        main_layout.addWidget(main_splitter)

    def _connect_signals(self):
        self.btn_calculate.clicked.connect(self.calculate_bands)
        self.btn_ws_refresh.clicked.connect(self.refresh_workspace_list)
        self.btn_ws_load.clicked.connect(self.load_workspace_structure)
        self.btn_push_bands.clicked.connect(self.push_bands_to_workspace)
        
        # --- NEW: Connect Ab-Initio Plotter ---
        self.btn_load_qe_bands.clicked.connect(self.load_qe_xml_bands)
        
        # Connect spin box for live plotting updates using the panel reference
        self.tb_panel.spin_iso.valueChanged.connect(self.update_2d_plot)

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
            
            # Pre-fill the UI SpinBoxes inside the panel
            if len(vals) > 0: self.tb_panel.spin_t1.setValue(vals[0])
            if len(vals) > 1: self.tb_panel.spin_t2.setValue(vals[1])
            if len(vals) > 2: self.tb_panel.spin_t3.setValue(vals[2])
            if len(vals) > 3: self.tb_panel.spin_t4.setValue(vals[3]) 
            else: self.tb_panel.spin_t4.setValue(0.0) 
            
            # Populate Projection Dropdown
            self.tb_panel.combo_projection.clear()
            self.tb_panel.combo_projection.addItem("None (Standard Lines)")
            
            num_wann_bands = 0
            
            if self.engine.crystal_structure:
                for site in self.engine.crystal_structure:
                    el = site.specie
                    
                    # --- NEW: Automatically calculate required Wannier Bands ---
                    if el.is_transition_metal or el.number > 30:
                        num_wann_bands += 9 # s(1) + p(3) + d(5)
                    else:
                        num_wann_bands += 4 # s(1) + p(3)
                    # -----------------------------------------------------------
                    
                    elem = site.species_string
                    for orb_str in self.engine._get_orbital_basis(elem):
                        if orb_str.endswith("0"): orb_name = "s"
                        elif orb_str[1] == "1": orb_name = "p" + orb_str[2:]
                        elif orb_str[1] == "2":
                            if orb_str.endswith("ZR"): orb_name = "dz2"
                            elif orb_str.endswith("XY"): orb_name = "dx2-y2"
                            else: orb_name = "d" + orb_str[2:]
                        else: orb_name = "unknown"
                        
                        label = f"{elem}_{orb_name}"
                        existing_items = [self.tb_panel.combo_projection.itemText(i) for i in range(self.tb_panel.combo_projection.count())]
                        if label not in existing_items:
                            self.tb_panel.combo_projection.addItem(label)
            
            # --- NEW: Smart UI Popup Message ---
            msg = f"Loaded '{target}' ({formula}) into DFT engine.\n\n"
            msg += f"Based on the atomic composition, you must set:\n"
            msg += f"Number of Bands (nbnd) = {num_wann_bands}"
            
            # If your QE panel's spinbox is named spin_nbnd, this will automatically set it!
            try:
                if hasattr(self.qe_panel, 'spin_nbnd'):
                    self.qe_panel.spin_nbnd.setValue(num_wann_bands)
                    msg += "\n\n(The UI value has been updated for you automatically!)"
            except Exception:
                pass
                
            QMessageBox.information(self, "Structure Loaded", msg)
            
        else:
            QMessageBox.critical(self, "Error", "Failed to load structure.")
    def calculate_bands(self):
        is_2d = self.tb_panel.combo_k_mode.currentIndex() == 1
        
        try:
            # 1. Generate k-points
            if not is_2d:
                template_name = self.tb_panel.combo_k_template.currentText()
                
                if "Auto-Detect" in template_name:
                    k_points, k_labels = self.engine.get_auto_kpath()
                elif "Arbitrary" in template_name:
                    k_points, k_labels = self.engine.get_custom_kpath(self.tb_panel.line_k_coords.text(), self.tb_panel.line_k_labels.text())
                else:
                    temp_key = template_name.split()[0].lower()
                    lattice_a, lattice_b = 3.0, 3.0
                    if hasattr(self.engine.crystal_structure, 'lattice'):
                        lattice_a = self.engine.crystal_structure.lattice.a
                        lattice_b = self.engine.crystal_structure.lattice.b
                    k_points, k_labels = self.engine.get_kpath_template(temp_key, a=lattice_a, b=lattice_b)

                if hasattr(self.engine.crystal_structure, 'lattice'):
                    recip_matrix = self.engine.crystal_structure.lattice.reciprocal_lattice.matrix
                    k_points = np.dot(k_points, recip_matrix)

                k_vecs, k_dist, node_idx, labels = self.engine.generate_k_path(
                    k_points, k_labels, points_per_segment=self.tb_panel.spin_k_res.value()
                )
            else:
                self.res = self.tb_panel.spin_k_res.value() 
                self.kx_vals = np.linspace(-4.5, 4.5, self.res)
                self.ky_vals = np.linspace(-4.5, 4.5, self.res)
                Kx, Ky = np.meshgrid(self.kx_vals, self.ky_vals)
                k_vecs = np.column_stack([Kx.ravel(), Ky.ravel(), np.zeros_like(Kx.ravel())])
                k_dist, node_idx, labels = None, None, None

            # 2. Re-pack UI values into the custom hopping dictionary
            custom_hopping = {}
            if hasattr(self, 'active_hopping_keys'):
                keys = self.active_hopping_keys
                if len(keys) > 0: custom_hopping[keys[0]] = self.tb_panel.spin_t1.value()
                if len(keys) > 1: custom_hopping[keys[1]] = self.tb_panel.spin_t2.value()
                if len(keys) > 2: custom_hopping[keys[2]] = self.tb_panel.spin_t3.value()
                if len(keys) > 3: custom_hopping[keys[3]] = self.tb_panel.spin_t4.value()

            # 3. Route to solver using custom physical shells or Wannier90
            is_soc = self.tb_panel.chk_soc.isChecked()
            soc_val = self.tb_panel.spin_soc_val.value()
            w90_file = self.tb_panel.active_w90_file

            ui_cutoffs = [
                self.tb_panel.spin_r1.value(),
                self.tb_panel.spin_r2.value(),
                self.tb_panel.spin_r3.value(),
                self.tb_panel.spin_r4.value()
            ]
            
            eigenvalues, eigenvectors, orb_labels = self.engine.solve_bands(
                k_vecs,
                custom_hopping=custom_hopping,
                onsite_e=self.tb_panel.spin_onsite.value(),
                use_soc=is_soc,
                soc_strength=soc_val,
                w90_filepath=w90_file,
                cutoffs=ui_cutoffs,
                tb_mode=self.tb_panel.combo_tb_mode.currentText()
            )
            # --- UNIVERSAL FERMI ENERGY SHIFT ---
            fermi_energy = 0.0
            if w90_file:
                import os
                work_dir = os.path.dirname(w90_file)
                # Hunt for the Fermi energy in the QE output logs
                for out_name in ["nscf.out", "scf.out"]:
                    out_path = os.path.join(work_dir, out_name)
                    if os.path.exists(out_path):
                        with open(out_path, 'r') as f:
                            for line in f:
                                if "the Fermi energy is" in line:
                                    # Extracts the number from "the Fermi energy is     6.4404 ev"
                                    fermi_energy = float(line.split()[4])
                        if fermi_energy != 0.0:
                            break
                            
            eigenvalues = eigenvalues - fermi_energy
            
            soc_title_tag = " (with SOC)" if is_soc and not w90_file else ""
            mode_tag = "Wannier90" if w90_file else "Chinook"
            title = f"{mode_tag} 2D Mesh" if is_2d else f"{mode_tag} Bands ({template_name.split()[0]} Path){soc_title_tag}"
            
        except Exception as e:
            QMessageBox.warning(self, "Calculation Error", str(e))
            return

        # Extract Basis Coordinates for ARPES
        basis_coords = []
        if self.engine.crystal_structure is not None:
            basis_coords = [site.coords.tolist() for site in self.engine.crystal_structure]

        # Ultra-Deep Hunt for Chinook objects
        found_basis, found_h_dict, found_tb_model = None, None, None
        
        # Access the underlying chinook engine inside the router
        chinook_engine = self.engine.chinook
        
        for attr_name in dir(chinook_engine):
            if attr_name.startswith('__'): continue
            attr_val = getattr(chinook_engine, attr_name)
            type_name = type(attr_val).__name__
            
            if 'TB_model' in type_name:
                found_tb_model = attr_val
            elif isinstance(attr_val, dict):
                if 'ham' in attr_name.lower() or 'dict' in attr_name.lower():
                    found_h_dict = attr_val
                if 'basis' in attr_val:
                    found_basis = attr_val['basis']
            elif isinstance(attr_val, (list, tuple)) or type_name == 'ndarray':
                try:
                    if len(attr_val) > 0 and 'orbital' in type(attr_val[0]).__name__.lower():
                        found_basis = attr_val
                except Exception:
                    pass
            if 'basis' in attr_name.lower() and found_basis is None:
                found_basis = attr_val

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
            'fermi_energy': fermi_energy,  # <--- ADD THIS EXACT LINE
            'title': title
        }
        
        if is_2d:
            self.active_bands_data.update({'kx': self.kx_vals, 'ky': self.ky_vals, 'grid_shape': (self.res, self.res)})
        else:
            self.active_bands_data.update({'k_dist': k_dist, 'node_idx': node_idx, 'labels': labels})
        self.btn_push_bands.setEnabled(True)

        # 5. Render Plot
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
            projection_mode = self.tb_panel.combo_projection.currentText()
            
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
            
            # --- NATIVE W90 OVERLAY PARSER ---
            if not is_2d and w90_file and self.tb_panel.chk_overlay_w90.isChecked():
                import os
                band_dat_path = w90_file.replace("_hr.dat", "_band.dat")
                if os.path.exists(band_dat_path):
                    w90_x, w90_y = [], []
                    label_added = False
                    
                    with open(band_dat_path, 'r') as f:
                        for line in f:
                            parts = line.split()
                            if len(parts) == 2:
                                w90_x.append(float(parts[0]))
                                # Shift W90 energy by the exact same VBM
                                w90_y.append(float(parts[1]) - fermi_energy)
                            elif len(parts) == 0 and len(w90_x) > 0:
                                lbl = "Wannier90 Native" if not label_added else None
                                # Scale the W90 x-axis to perfectly align with PyMatgen
                                w90_x_arr = np.array(w90_x)
                                if w90_x_arr[-1] > 0:
                                    w90_x_arr = w90_x_arr * (k_dist[-1] / w90_x_arr[-1])
                                self.ax.plot(w90_x_arr, w90_y, color='red', linestyle='--', linewidth=1.5, zorder=0, label=lbl)
                                label_added = True
                                w90_x, w90_y = [], []
                                
                    if len(w90_x) > 0:
                        lbl = "Wannier90 Native" if not label_added else None
                        w90_x_arr = np.array(w90_x)
                        if w90_x_arr[-1] > 0:
                            w90_x_arr = w90_x_arr * (k_dist[-1] / w90_x_arr[-1])
                        self.ax.plot(w90_x_arr, w90_y, color='red', linestyle='--', linewidth=1.5, zorder=0, label=lbl)
                        
                    self.ax.legend(loc='upper right')
                else:
                    print(f"Native band file not found at: {band_dat_path}")
            
            self.ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=1)
            self.ax.set_xlim(0, k_dist[-1])
            
            # --- Apply Y-Axis Zoom ---
            e_min = self.tb_panel.spin_emin.value()
            e_max = self.tb_panel.spin_emax.value()
            self.ax.set_ylim(e_min, e_max)
            
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
        
        omega = self.tb_panel.spin_iso.value()
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
            
        dim_str = "2D" if self.active_bands_data.get('is_2d') else "1D"
        default_name = f"TB_Bands_{dim_str}"
        
        name, ok = QInputDialog.getText(self, "Save to Workspace", 
                                        "Enter variable name for workspace:", 
                                        text=default_name)
        
        if not ok or not name.strip():
            return
            
        name = name.strip()
        global_workspace._data[name] = self.active_bands_data
        
        dim_str_display = "2D Mesh" if self.active_bands_data.get('is_2d') else "1D Path"
        QMessageBox.information(self, "Success", f"Band structure '{name}' ({dim_str_display}) pushed to Workspace!\nYou can now load it in the ARPES Suite.")

    def load_qe_xml_bands(self):
        """Loads and overlays raw Quantum ESPRESSO bands onto the current plot."""
        fname, _ = QFileDialog.getOpenFileName(self, 'Open QE XML File', '', "XML files (*.xml);;All files (*.*)")
        if not fname:
            return

        try:
            qe_k_dist, qe_eigenvals, qe_fermi = self.engine.parse_qe_xml(fname)

            # Scale the QE x-axis to perfectly align with the PyMatgen tight-binding axis
            if hasattr(self, 'active_bands_data') and not self.active_bands_data.get('is_2d'):
                tb_k_dist = self.active_bands_data.get('k_dist')
                if tb_k_dist is not None and len(tb_k_dist) > 0 and qe_k_dist[-1] > 0:
                    qe_k_dist = qe_k_dist * (tb_k_dist[-1] / qe_k_dist[-1])

            num_bands = qe_eigenvals.shape[1]
            label_added = False
            
            # Plot the solid black DFT bands behind the current bands (zorder=0)
            for b in range(num_bands):
                lbl = "Ab-Initio (QE)" if not label_added else None
                self.ax.plot(qe_k_dist, qe_eigenvals[:, b] - qe_fermi, color='black', linewidth=1.5, zorder=0, label=lbl)
                label_added = True

            self.ax.legend(loc='upper right')
            self.canvas.draw()
            QMessageBox.information(self, "Success", f"Successfully loaded and overlaid {num_bands} Ab-Initio bands from QE.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse QE XML:\n{str(e)}")
    
    def closeEvent(self, event):
        print("close suite DFT Suite")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DFTSuite()
    window.show()
    sys.exit(app.exec())