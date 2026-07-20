import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
                               QDoubleSpinBox, QPushButton, QGroupBox, QLabel, 
                               QSpinBox, QLineEdit, QComboBox, QCheckBox, QFileDialog)

class TightBindingPanel(QWidget):
    """
    Isolated UI Component containing all inputs for the Tight Binding engine.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_w90_file = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        tb_group = QGroupBox("Tight Binding Parameters")
        tb_form = QFormLayout(tb_group)
        
        # --- Dimension Toggle ---
        self.combo_k_mode = QComboBox()
        self.combo_k_mode.addItems(["1D High-Symmetry Path", "2D k-Mesh (kx, ky)"])
        tb_form.addRow("k-Space Grid:", self.combo_k_mode)

        # --- ARPES Isoenergy Control ---
        self.spin_iso = QDoubleSpinBox()
        self.spin_iso.setRange(-20.0, 20.0)
        self.spin_iso.setValue(0.0)
        self.spin_iso.setSingleStep(0.1)
        self.spin_iso.setEnabled(False) 
        tb_form.addRow("2D Isoenergy Cut (eV):", self.spin_iso)
        
        self.combo_k_mode.currentTextChanged.connect(
            lambda text: self.spin_iso.setEnabled("2D" in text)
        )
        
        # --- K-Path Template Toggle ---
        self.combo_k_template = QComboBox()
        self.combo_k_template.addItems([
            "Auto-Detect BZ Path (PyMatgen)",
            "Arbitrary Custom Path",
            "Hexagonal (Template)", 
            "Rectangular / Orthorhombic (Template)", 
            "Square / Tetragonal (Template)"
        ])
        tb_form.addRow("1D Path Template:", self.combo_k_template)

        # --- Spin-Orbit Coupling Controls ---
        soc_layout = QHBoxLayout()
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

        # --- Fat Band Projection Controls ---
        self.combo_projection = QComboBox()
        self.combo_projection.addItem("None (Standard Lines)")
        tb_form.addRow("Fat Band Target:", self.combo_projection)
        
        self.spin_soc_val.setEnabled(False)
        self.chk_soc.stateChanged.connect(lambda state: self.spin_soc_val.setEnabled(bool(state)))

        # --- Wannier90 Importer ---
        w90_group = QGroupBox("Ab Initio Import (Wannier90)")
        w90_layout = QVBoxLayout(w90_group)
        
        self.btn_load_w90 = QPushButton("📂 Load wannier90_hr.dat")
        self.btn_load_w90.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold;")
        self.lbl_w90_status = QLabel("Status: Using Manual Slater-Koster parameters.")
        self.lbl_w90_status.setStyleSheet("color: gray; font-size: 10px;")
        
        # --- NEW OVERLAY CHECKBOX ---
        self.chk_overlay_w90 = QCheckBox("Overlay Native W90 Bands (Red Dashed)")
        self.chk_overlay_w90.setChecked(True)
        self.chk_overlay_w90.setEnabled(False) # Disabled until a file is loaded
        
        w90_layout.addWidget(self.btn_load_w90)
        w90_layout.addWidget(self.chk_overlay_w90)
        w90_layout.addWidget(self.lbl_w90_status)
        tb_form.addRow(w90_group)
        
        self.btn_load_w90.clicked.connect(self.load_w90_file)
        
        # --- Custom Arbitrary Path Inputs ---
        self.custom_k_widget = QWidget()
        custom_layout = QFormLayout(self.custom_k_widget)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        
        self.line_k_coords = QLineEdit("0,0,0; 0.5,0,0; 0.5,0.5,0; 0,0,0")
        self.line_k_labels = QLineEdit("G; X; M; G")
        
        # --- Hamiltonian Mode Toggle ---
        self.combo_tb_mode = QComboBox()
        self.combo_tb_mode.addItems(["Simple Scalar (Isotropic)", "Slater-Koster (Rigorous)"])
        tb_form.addRow("Hamiltonian Mode:", self.combo_tb_mode)

        custom_layout.addRow("Coords (frac):", self.line_k_coords)
        custom_layout.addRow("Labels:", self.line_k_labels)
        tb_form.addRow(self.custom_k_widget)
        
        self.custom_k_widget.setVisible(False)
        self.combo_k_template.currentTextChanged.connect(
            lambda text: self.custom_k_widget.setVisible("Arbitrary" in text)
        )
        
        # --- Resolution Control ---
        self.spin_k_res = QSpinBox()
        self.spin_k_res.setRange(10, 2000)
        self.spin_k_res.setValue(100)
        self.spin_k_res.setSingleStep(50)
        tb_form.addRow("Points per Segment:", self.spin_k_res)
        
        self.spin_onsite = QDoubleSpinBox()
        self.spin_onsite.setRange(-10.0, 10.0); self.spin_onsite.setValue(0.0); self.spin_onsite.setSingleStep(0.1)
        tb_form.addRow("On-site E (eV):", self.spin_onsite)

        # --- NEW: Energy Zoom Controls ---
        self.spin_emin = QDoubleSpinBox()
        self.spin_emin.setRange(-50.0, 50.0); self.spin_emin.setValue(-6.0); self.spin_emin.setSingleStep(0.5)
        self.spin_emax = QDoubleSpinBox()
        self.spin_emax.setRange(-50.0, 50.0); self.spin_emax.setValue(6.0); self.spin_emax.setSingleStep(0.5)
        
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Min:"))
        range_layout.addWidget(self.spin_emin)
        range_layout.addWidget(QLabel("Max:"))
        range_layout.addWidget(self.spin_emax)
        tb_form.addRow("Y-Axis Zoom (eV):", range_layout)

        # Hopping Shells
        self.spin_t1 = QDoubleSpinBox(); self.spin_t1.setRange(-10.0, 10.0); self.spin_t1.setValue(2.7); self.spin_t1.setMinimumWidth(65)
        self.spin_r1 = QDoubleSpinBox(); self.spin_r1.setRange(0.0, 10.0); self.spin_r1.setValue(1.6); self.spin_r1.setMinimumWidth(65)
        
        self.spin_t2 = QDoubleSpinBox(); self.spin_t2.setRange(-10.0, 10.0); self.spin_t2.setValue(0.0); self.spin_t2.setMinimumWidth(65)
        self.spin_r2 = QDoubleSpinBox(); self.spin_r2.setRange(0.0, 10.0); self.spin_r2.setValue(2.6); self.spin_r2.setMinimumWidth(65)
        
        self.spin_t3 = QDoubleSpinBox(); self.spin_t3.setRange(-10.0, 10.0); self.spin_t3.setValue(0.0); self.spin_t3.setMinimumWidth(65)
        self.spin_r3 = QDoubleSpinBox(); self.spin_r3.setRange(0.0, 10.0); self.spin_r3.setValue(3.1); self.spin_r3.setMinimumWidth(65)
        
        self.spin_t4 = QDoubleSpinBox(); self.spin_t4.setRange(-10.0, 10.0); self.spin_t4.setValue(-0.3); self.spin_t4.setMinimumWidth(65)
        self.spin_r4 = QDoubleSpinBox(); self.spin_r4.setRange(0.0, 15.0); self.spin_r4.setValue(4.5); self.spin_r4.setMinimumWidth(65)
        
        shells = [
            (self.spin_t1, self.spin_r1), 
            (self.spin_t2, self.spin_r2), 
            (self.spin_t3, self.spin_r3),
            (self.spin_t4, self.spin_r4)
        ]
        for i, (t_spin, r_spin) in enumerate(shells, start=1):
            row = QHBoxLayout()
            row.addWidget(t_spin)
            row.addWidget(QLabel("Max Å:"))
            row.addWidget(r_spin)
            tb_form.addRow(f"Hopping t{i}:", row)

        main_layout.addWidget(tb_group)

    def load_w90_file(self):
        """Opens a file dialog to load the Wannier90 hopping data."""
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Wannier90 HR File', '', "Data files (*.dat);;All files (*.*)")
        if fname:
            self.active_w90_file = fname
            filename_short = fname.split('/')[-1]
            self.lbl_w90_status.setText(f"Status: Using {filename_short}")
            self.lbl_w90_status.setStyleSheet("color: blue; font-weight: bold;")
            
            self.spin_t1.setEnabled(False)
            self.spin_t2.setEnabled(False)
            self.spin_t3.setEnabled(False)

            self.chk_overlay_w90.setEnabled(True)