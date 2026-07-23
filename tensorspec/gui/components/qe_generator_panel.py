import os
import subprocess
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QWidget, QFormLayout, QGroupBox, QComboBox, 
                               QSpinBox, QHBoxLayout, QLineEdit, QPushButton, 
                               QMessageBox, QPlainTextEdit, QVBoxLayout, QCheckBox, QLabel)

from tensorspec.core.dft.qe_generator import QEInputGenerator

class QERunnerThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, script_content, out_dir):
        super().__init__()
        self.script_content = script_content
        self.out_dir = out_dir

    def run(self):
        script_path = os.path.join(self.out_dir, "run_pipeline.sh")
        try:
            # 1. Save the GUI text box content as a bash script
            with open(script_path, "w") as f:
                f.write(self.script_content)
            
            os.chmod(script_path, 0o755)
            
            # 2. Execute the bash script inside the target directory
            process = subprocess.Popen(
                ["/bin/bash", "run_pipeline.sh"],
                cwd=self.out_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in process.stdout:
                self.log_signal.emit(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                self.finished_signal.emit(True, "Pipeline finished successfully!")
            else:
                self.finished_signal.emit(False, f"Process failed with exit code {process.returncode}.")
                
        except Exception as e:
            self.finished_signal.emit(False, f"An error occurred: {str(e)}")

class QEGeneratorPanel(QWidget):
    def __init__(self, engine_reference, parent=None):
        super().__init__(parent)
        self.engine = engine_reference  
        self.is_viewing = True 
        self._setup_ui()

    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        qe_group = QGroupBox("Quantum Espresso & Wannier90 Pipeline")
        qe_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        qe_form = QFormLayout(qe_group)

        # Basic Parameters
        self.spin_ecut = QSpinBox()
        self.spin_ecut.setRange(20, 200); self.spin_ecut.setValue(60)
        qe_form.addRow("Wavefunction Cutoff (Ry):", self.spin_ecut)
        
        self.chk_soc = QCheckBox("Inject Spin-Orbit Coupling Math")
        qe_form.addRow("Relativistic (SOC):", self.chk_soc)
        
        self.spin_nbnd = QSpinBox()
        self.spin_nbnd.setRange(1, 500); self.spin_nbnd.setValue(12)
        qe_form.addRow("Number of Bands (nbnd):", self.spin_nbnd)
        
        # 1. Wannier Hybridization Toggle
        self.combo_wannier_mode = QComboBox()
        self.combo_wannier_mode.addItems([
            "Atomic Projections (Chinook ARPES)", 
            "Maximally Localized (WannierTools)"
        ])
        qe_form.addRow("Wannier Mode:", self.combo_wannier_mode)

        kmesh_layout = QHBoxLayout()
        self.spin_kx = QSpinBox(); self.spin_kx.setRange(1, 20); self.spin_kx.setValue(6)
        self.spin_ky = QSpinBox(); self.spin_ky.setRange(1, 20); self.spin_ky.setValue(6)
        self.spin_kz = QSpinBox(); self.spin_kz.setRange(1, 20); self.spin_kz.setValue(6)
        kmesh_layout.addWidget(self.spin_kx); kmesh_layout.addWidget(self.spin_ky); kmesh_layout.addWidget(self.spin_kz)
        qe_form.addRow("k-Mesh Grid:", kmesh_layout)

        self.line_outdir = QLineEdit("./qe_workspace")
        qe_form.addRow("Output Directory:", self.line_outdir)

        # Parallel setup
        parallel_layout = QHBoxLayout()
        self.chk_mpi = QCheckBox("Use MPI")
        self.chk_mpi.setChecked(True)
        self.spin_cores = QSpinBox()
        self.spin_cores.setRange(1, 256) 
        self.spin_cores.setValue(16) 
        parallel_layout.addWidget(self.chk_mpi); parallel_layout.addWidget(self.spin_cores); parallel_layout.addWidget(QLabel("Cores")); parallel_layout.addStretch() 
        qe_form.addRow("Parallel Execution:", parallel_layout)

        # --- MACHINE AGNOSTIC EXECUTABLES ---
        self.line_pw_cmd = QLineEdit("pw.x")
        qe_form.addRow("pw.x Command:", self.line_pw_cmd)
        
        self.line_wan_cmd = QLineEdit("wannier90.x")
        qe_form.addRow("Wannier90 Command:", self.line_wan_cmd)

        self.line_pw2wan_cmd = QLineEdit("pw2wannier90.x")
        qe_form.addRow("pw2wan.x Command:", self.line_pw2wan_cmd)

        self.btn_gen_qe = QPushButton("📄 Generate Input Files")
        self.btn_gen_qe.setStyleSheet("background-color: #e0a800; color: black; font-weight: bold;")
        qe_form.addRow(self.btn_gen_qe)
        
        # Text-Based Pipeline Script Editor
        self.script_editor = QPlainTextEdit()
        self.script_editor.setPlaceholderText("Click 'Generate Input Files' to auto-populate this bash script...")
        self.script_editor.setStyleSheet("background-color: #2b2b2b; color: #f8f8f2; font-family: monospace;")
        self.script_editor.setMaximumHeight(150)
        qe_form.addRow("Pipeline Script:", self.script_editor)

        self.btn_run_qe = QPushButton("🚀 Run Script Locally")
        self.btn_run_qe.setStyleSheet("""
            QPushButton { background-color: #c82333; color: white; font-weight: bold; padding: 5px; }
            QPushButton:disabled { background-color: #5c161b; color: #888888; }
        """)
        qe_form.addRow(self.btn_run_qe)

        self.main_layout.addWidget(qe_group)

        # Live Console Elements
        self.toggle_view_btn = QPushButton("⏸ Pause Viewer (Keep Running)")
        self.toggle_view_btn.setCheckable(True)
        self.toggle_view_btn.clicked.connect(self.toggle_viewer)
        self.toggle_view_btn.hide()
        
        self.log_display = QPlainTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace;")
        self.log_display.setMaximumHeight(200)
        self.log_display.hide()

        self.main_layout.addWidget(self.toggle_view_btn)
        self.main_layout.addWidget(self.log_display)

        self.btn_gen_qe.clicked.connect(self.generate_qe_files)
        self.btn_run_qe.clicked.connect(self.run_qe_script)
        self.chk_mpi.stateChanged.connect(lambda: self.spin_cores.setEnabled(self.chk_mpi.isChecked()))

    def generate_qe_files(self):
        if not self.engine.crystal_structure:
            QMessageBox.warning(self, "Warning", "Please load a structure from the workspace first.")
            return

        out_dir = self.line_outdir.text()
        kmesh = (self.spin_kx.value(), self.spin_ky.value(), self.spin_kz.value())
        ecut = float(self.spin_ecut.value())
        nbnd = self.spin_nbnd.value()

        qe_gen = QEInputGenerator(self.engine.crystal_structure)
        
        try:
            # Grab the SOC state directly from this QE panel!
            is_soc_enabled = self.chk_soc.isChecked()

            # Generate all 4 configuration files, passing the SOC state
            qe_gen.write_scf_input(out_dir, ecutwfc=ecut, kmesh=kmesh, use_soc=is_soc_enabled)
            qe_gen.write_nscf_input(out_dir, ecutwfc=ecut, kmesh=kmesh, nbnd=nbnd, use_soc=is_soc_enabled)
            
            # Pass the MLWF mode toggle 
            is_mlwf = (self.combo_wannier_mode.currentIndex() == 1)
            qe_gen.write_wannier90_input(out_dir, kmesh=kmesh, num_wann=nbnd, use_soc=is_soc_enabled, mlwf_mode=is_mlwf)
            
            qe_gen.write_pw2wan_input(out_dir)
            
            # Extract Commands
            pw_exec = self.line_pw_cmd.text().strip()
            wan_exec = self.line_wan_cmd.text().strip()
            pw2wan_exec = self.line_pw2wan_cmd.text().strip()
            
            mpi_cmd = f"mpirun -np {self.spin_cores.value()} " if self.chk_mpi.isChecked() else ""
            
            # Auto-populate portable bash script with exact commands
            script_text = (
                "#!/bin/bash\n"
                "set -e\n"
                "export OMP_NUM_THREADS=1\n\n"
                "# Edit this script to run specific parts of the pipeline\n"
                f"{mpi_cmd}{pw_exec} -ndiag 1 -in scf.in | tee scf.out\n"
                f"{mpi_cmd}{pw_exec} -ndiag 1 -in nscf.in | tee nscf.out\n"
                f"{wan_exec} -pp wannier90\n"
                f"{mpi_cmd}{pw2wan_exec} -in pw2wan.in | tee pw2wan.out\n"
                f"{wan_exec} wannier90\n"
            )
            
            self.script_editor.setPlainText(script_text)
            
            QMessageBox.information(self, "Success", f"Inputs generated in {out_dir}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate files:\n{str(e)}")

    def run_qe_script(self):
        script_content = self.script_editor.toPlainText().strip()
        out_dir = self.line_outdir.text()

        if not script_content:
            QMessageBox.warning(self, "Warning", "Script is empty. Please click 'Generate Input Files' first.")
            return

        os.makedirs(out_dir, exist_ok=True)
        
        self.btn_run_qe.setEnabled(False)
        self.log_display.clear()
        self.log_display.show()
        self.toggle_view_btn.show()
        self.toggle_view_btn.setChecked(False)
        self.toggle_view_btn.setText("⏸ Pause Viewer (Keep Running)")
        self.is_viewing = True
        
        self.qe_thread = QERunnerThread(script_content, out_dir)
        self.qe_thread.log_signal.connect(self.update_log)
        self.qe_thread.finished_signal.connect(self.calculation_finished)
        self.qe_thread.start()

    def update_log(self, text):
        if self.is_viewing:
            self.log_display.appendPlainText(text)
            scrollbar = self.log_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def toggle_viewer(self):
        if self.toggle_view_btn.isChecked():
            self.is_viewing = False
            self.toggle_view_btn.setText("▶ Resume Viewer")
            self.log_display.appendPlainText("\n--- VIEWER PAUSED ---\n")
        else:
            self.is_viewing = True
            self.toggle_view_btn.setText("⏸ Pause Viewer (Keep Running)")

    def calculation_finished(self, success, message):
        self.btn_run_qe.setEnabled(True)
        if success:
            self.log_display.appendPlainText(f"\n--- SUCCESS: {message} ---")
        else:
            self.log_display.appendPlainText(f"\n--- ERROR: {message} ---")