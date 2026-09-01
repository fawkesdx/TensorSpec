import os
import subprocess
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QWidget, QFormLayout, QGroupBox, QComboBox, 
                               QSpinBox, QHBoxLayout, QLineEdit, QPushButton, 
                               QMessageBox, QPlainTextEdit, QVBoxLayout, QCheckBox, QLabel)

from tensorspec.core.dft.qe_generator import QEInputGenerator
from tensorspec.core.compute import cluster_paths as cp


class QERunnerThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, script_content, out_dir, cluster=None, mpi_ranks=1):
        super().__init__()
        self.script_content = script_content
        self.out_dir = out_dir
        self.cluster = cluster
        self.mpi_ranks = max(1, int(mpi_ranks))

    def run(self):
        import os, subprocess, paramiko
        is_win = os.name == 'nt'
        script_name = "run_pipeline.ps1" if is_win else "run_pipeline.sh"
        script_path = os.path.join(self.out_dir, script_name)
        
        try:
            # 1. Save the GUI text box content as the appropriate script type
            with open(script_path, "w") as f:
                f.write(self.script_content)
                
            if self.cluster and self.cluster != "local":
                # --- REMOTE EXECUTION ---
                self.log_signal.emit(f"Connecting to {self.cluster['host']}...")
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                cp.ssh_connect(ssh, self.cluster, timeout=15)
                
                remote_dir = cp.job_dir(self.cluster, "qe")
                
                sftp = ssh.open_sftp()
                try:
                    sftp.mkdir(cp.heavy_root(self.cluster))
                except OSError:
                    pass
                try:
                    sftp.mkdir(remote_dir)
                except OSError:
                    pass
                
                # Upload only necessary input files to avoid uploading 10GB+ of old data!
                self.log_signal.emit(f"Uploading input files to {remote_dir}...")
                allowed_exts = ['.in', '.win', '.sh', '.ps1']
                for file_name in os.listdir(self.out_dir):
                    local_f = os.path.join(self.out_dir, file_name)
                    if os.path.isfile(local_f) and any(file_name.endswith(ext) for ext in allowed_exts):
                        sftp.put(local_f, f"{remote_dir}/{file_name}")
                
                # Make sure pseudo directory exists and is uploaded
                pseudo_dir_local = os.path.join(self.out_dir, "pseudo")
                if os.path.isdir(pseudo_dir_local):
                    try:
                        sftp.mkdir(f"{remote_dir}/pseudo")
                    except OSError:
                        pass
                    for p_file in os.listdir(pseudo_dir_local):
                        p_local = os.path.join(pseudo_dir_local, p_file)
                        if os.path.isfile(p_local):
                            sftp.put(p_local, f"{remote_dir}/pseudo/{p_file}")

                if cp.is_slurm(self.cluster):
                    sbatch_content = cp.build_qe_slurm_batch(
                        self.cluster,
                        remote_dir=remote_dir,
                        script_name=script_name,
                        mpi_ranks=self.mpi_ranks,
                    )
                    with sftp.file(f"{remote_dir}/job.sbatch", "w") as f:
                        f.write(sbatch_content)
                    cmd = f"cd {remote_dir} && sbatch job.sbatch"
                    msg_type = "via SLURM Queue"
                    self.log_signal.emit(
                        f"SLURM: account={cp.slurm_account(self.cluster)} "
                        f"qos={cp.slurm_qos(self.cluster)} "
                        f"constraint={cp.slurm_constraint(self.cluster)} "
                        f"time={cp.slurm_walltime(self.cluster)} "
                        f"ntasks={self.mpi_ranks}"
                    )
                else:
                    cmd = (
                        f"bash -lc 'cd {remote_dir} && "
                        f"{cp.shell_thread_limits(one_line=True)} && "
                        f"{cp.qe_env_exports(self.cluster, one_line=True)} && "
                        f"nohup bash {script_name} > sys.out.full 2>&1 &'"
                    )
                    msg_type = "via Background Daemon"

                sftp.close()
                    
                self.log_signal.emit(f"Dispatching execution {msg_type}...")
                stdin, stdout, stderr = ssh.exec_command(cmd)
                out = stdout.read().decode()
                err = stderr.read().decode()
                
                ssh.close()
                
                self.log_signal.emit(out.strip())
                if err.strip():
                    self.log_signal.emit(f"STDERR: {err.strip()}")
                if cp.is_slurm(self.cluster) and "error:" in (out + err).lower():
                    self.finished_signal.emit(
                        False,
                        "Slurm submit failed — check qos/time limits (debug max 30 min).",
                    )
                    return
                    
                self.finished_signal.emit(True, f"Remote Dispatch Complete! Logs at: {remote_dir}/sys.out.full")
                return

            # --- LOCAL EXECUTION ---
            if not is_win:
                os.chmod(script_path, 0o755)
                cmd = ["/bin/bash", script_name]
            else:
                cmd = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", script_name]
            
            process = subprocess.Popen(
                cmd,
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

        # --- PWscf backend: CPU MPI (default) or QE-native CUDA ---
        self.combo_pw_backend = QComboBox()
        self.combo_pw_backend.addItem("CPU (MPI pw.x)", "cpu")
        self.combo_pw_backend.addItem("GPU (CUDA pw.x)", "gpu")
        self.combo_pw_backend.setToolTip(
            "GPU uses Quantum ESPRESSO use_gpu=.true. (not GrizzlyME). "
            "Requires pw.x built with CUDA (remote-cluster qe conda env). "
            "Mac/local conda pw.x is usually CPU-only."
        )
        self.combo_gpu_device = QComboBox()
        self.combo_gpu_device.addItem("GPU 0", "0")
        self.combo_gpu_device.addItem("GPU 1", "1")
        self.combo_gpu_device.addItem("Both GPUs (0+1)", "0,1")
        self.combo_gpu_device.setEnabled(False)
        self.combo_gpu_device.setToolTip(
            "CUDA_VISIBLE_DEVICES for pw.x. MPI ranks should match GPU count."
        )
        self.combo_pw_backend.currentIndexChanged.connect(self._sync_pw_backend_ui)
        self.combo_gpu_device.currentIndexChanged.connect(self._sync_pw_backend_ui)
        qe_form.addRow("PWscf backend:", self.combo_pw_backend)
        qe_form.addRow("CUDA device(s):", self.combo_gpu_device)

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

        # Compute Target Dropdown
        self.combo_cluster = QComboBox()
        self.combo_cluster.addItem("💻 Local (This Machine)", "local")
        self.populate_clusters()
        qe_form.addRow("Compute Target:", self.combo_cluster)



        self.btn_run_qe = QPushButton("🚀 Run Script")
        self.btn_run_qe.setStyleSheet("""
            QPushButton { background-color: #c82333; color: white; font-weight: bold; padding: 5px; }
            QPushButton:disabled { background-color: #5c161b; color: #888888; }
        """)
        qe_form.addRow(self.btn_run_qe)
        
        self.btn_fetch_wan = QPushButton("📥 Fetch Remote Outputs")
        self.btn_fetch_wan.setStyleSheet("""
            QPushButton { background-color: #17a2b8; color: white; font-weight: bold; padding: 5px; }
            QPushButton:disabled { background-color: #0c5460; color: #888888; }
        """)
        qe_form.addRow(self.btn_fetch_wan)
        self.btn_fetch_wan.clicked.connect(self.fetch_remote_outputs)


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


    def _sync_pw_backend_ui(self, _index=None):
        use_gpu = self.combo_pw_backend.currentData() == "gpu"
        self.combo_gpu_device.setEnabled(use_gpu)
        if use_gpu and self.chk_mpi.isChecked():
            dev = self.combo_gpu_device.currentData() or "0"
            n_ranks = 2 if "," in str(dev) else 1
            if self.spin_cores.value() != n_ranks:
                self.spin_cores.setValue(n_ranks)

    def _pw_backend_flags(self):
        """Return (use_gpu, cuda_devices, mpi_ranks) for script / input generation."""
        use_gpu = self.combo_pw_backend.currentData() == "gpu"
        if not use_gpu:
            ranks = self.spin_cores.value() if self.chk_mpi.isChecked() else 1
            return False, "", ranks
        dev = str(self.combo_gpu_device.currentData() or "0")
        ranks = dev.count(",") + 1
        return True, dev, ranks

    def _pipeline_env_header(self, use_gpu: bool, cuda_devices: str) -> str:
        if not use_gpu:
            return "export OMP_NUM_THREADS=1\n\n"
        return (
            "export OMP_NUM_THREADS=1\n"
            f"export CUDA_VISIBLE_DEVICES={cuda_devices}\n"
            "# QE GPU: pw.x must be CUDA build; MPI ranks ≈ number of GPUs\n\n"
        )

    def populate_clusters(self):
        import json, os
        config_file = os.path.expanduser('~/.tensorspec_clusters.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    clusters = json.load(f)
                for c in clusters:
                    self.combo_cluster.addItem(f"🚀 Remote: {c.get('name', c.get('host'))}", c)
            except Exception as e:
                print(f"Error reading cluster config: {e}")

    def get_selected_cluster(self):
        selected_data = self.combo_cluster.currentData()
        if isinstance(selected_data, dict):
            return selected_data
        return None


    def fetch_remote_outputs(self):
        cluster = self.get_selected_cluster()
        if not cluster or cluster == "local":
            QMessageBox.information(self, "Info", "Select a remote cluster to fetch from.")
            return
            
        out_dir = self.line_outdir.text()
        os.makedirs(out_dir, exist_ok=True)
        
        try:
            import paramiko
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cp.ssh_connect(ssh, cluster, timeout=15)
            
            sftp = ssh.open_sftp()
            remote_dir = cp.job_dir(cluster, "qe")
            
            # Fetch only the logs and final Wannier90 tight-binding model to save bandwidth!
            # Do NOT download .mmn, .amn, .eig, .chk, or .save directories as they are massive (10+ GB).
            allowed_fetch = ['.out', '.gnu', '.kpt', '.dat', '.xyz', 'sys.out.full', '.wout']
            
            for file_attr in sftp.listdir_attr(remote_dir):
                remote_f = f"{remote_dir}/{file_attr.filename}"
                local_f = os.path.join(out_dir, file_attr.filename)
                import stat
                if stat.S_ISREG(file_attr.st_mode):
                    if any(file_attr.filename.endswith(ext) for ext in allowed_fetch):
                        sftp.get(remote_f, local_f)
            
            sftp.close()
            ssh.close()
            QMessageBox.information(self, "Success", f"Fetched all remote outputs into '{out_dir}'.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch remote outputs:\n{str(e)}")

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
            use_gpu, cuda_devices, mpi_ranks = self._pw_backend_flags()
            if use_gpu and is_soc_enabled:
                QMessageBox.warning(
                    self,
                    "GPU + SOC",
                    "QE GPU + noncollinear SOC can be slow or unsupported on some builds. "
                    "Verify on your cluster before long runs.",
                )

            # Generate all 4 configuration files, passing the SOC state
            qe_gen.write_scf_input(
                out_dir, ecutwfc=ecut, kmesh=kmesh, use_soc=is_soc_enabled, use_gpu=use_gpu
            )
            qe_gen.write_nscf_input(
                out_dir,
                ecutwfc=ecut,
                kmesh=kmesh,
                nbnd=nbnd,
                use_soc=is_soc_enabled,
                use_gpu=use_gpu,
            )
            
            # Pass the MLWF mode toggle 
            is_mlwf = (self.combo_wannier_mode.currentIndex() == 1)
            qe_gen.write_wannier90_input(out_dir, kmesh=kmesh, num_wann=nbnd, use_soc=is_soc_enabled, mlwf_mode=is_mlwf)
            
            qe_gen.write_pw2wan_input(out_dir)
            
            # Extract Commands
            pw_exec = self.line_pw_cmd.text().strip()
            wan_exec = self.line_wan_cmd.text().strip()
            pw2wan_exec = self.line_pw2wan_cmd.text().strip()
            
            mpi_cmd = cp.mpi_launch_prefix(
                self.get_selected_cluster(),
                mpi_ranks,
                use_mpi=self.chk_mpi.isChecked(),
            )
            env_header = self._pipeline_env_header(use_gpu, cuda_devices)
            
            # Auto-populate portable script with exact commands based on Operating System
            if os.name == 'nt':
                # Windows native PowerShell formatting
                script_text = (
                    "$env:OMP_NUM_THREADS=1\n"
                    + (
                        f"$env:CUDA_VISIBLE_DEVICES='{cuda_devices}'\n"
                        if use_gpu
                        else ""
                    )
                    + "\n"
                    "# ==================================================================\n"
                    "# ADVANCED: HSE HYBRID FUNCTIONAL SWITCH\n"
                    "# By default, this script runs a standard PBE calculation.\n"
                    "# To run HSE, remove the '#' from the two replacement commands below.\n"
                    "# ==================================================================\n"
                    f"# (Get-Content scf.in) -replace '&SYSTEM', \"&SYSTEM`n    input_dft = 'hse',`n    nqx1 = {kmesh[0]}, nqx2 = {kmesh[1]}, nqx3 = {kmesh[2]},\" | Set-Content scf.in\n"
                    f"# (Get-Content nscf.in) -replace '&SYSTEM', \"&SYSTEM`n    input_dft = 'hse',`n    nqx1 = {kmesh[0]}, nqx2 = {kmesh[1]}, nqx3 = {kmesh[2]},\" | Set-Content nscf.in\n\n"
                    "mkdir -p out tmp\n" 
                    "export TMPDIR=$(pwd)/tmp\n" 
                    "# Edit this script to run specific parts of the pipeline\n"
                    f"{mpi_cmd}{pw_exec} -in scf.in | Tee-Object -FilePath scf.out\n"
                    f"{mpi_cmd}{pw_exec} -in nscf.in | Tee-Object -FilePath nscf.out\n"
                    f"{wan_exec} -pp wannier90\n"
                    f"{mpi_cmd}{pw2wan_exec} -in pw2wan.in | Tee-Object -FilePath pw2wan.out\n"
                    f"{wan_exec} wannier90\n"
                )
            else:
                # Mac / Linux native Bash formatting using a robust Python one-liner!
                script_text = (
                    "#!/bin/bash\n"
                    "set -e\n"
                    + env_header
                    + "# ==================================================================\n"
                    + "# ADVANCED: HSE HYBRID FUNCTIONAL SWITCH\n"
                    + "# By default, this script runs a standard PBE calculation.\n"
                    + "# To run HSE, just uncomment (remove the '#') from the python command below.\n"
                    + "# ==================================================================\n"
                    + f"# python -c \"for f in ['scf.in','nscf.in']: d=open(f).read(); open(f,'w').write(d.replace('&SYSTEM','&SYSTEM\\n    input_dft=\\'hse\\',\\n    nqx1={kmesh[0]}, nqx2={kmesh[1]}, nqx3={kmesh[2]},'))\"\n\n"
                    + "mkdir -p out tmp\n"
                    + "export TMPDIR=$(pwd)/tmp\n"
                    + "# Edit this script to run specific parts of the pipeline\n"
                    + f"{mpi_cmd}{pw_exec} -in scf.in | tee scf.out\n"
                    + f"{mpi_cmd}{pw_exec} -in nscf.in | tee nscf.out\n"
                    + f"{wan_exec} -pp wannier90\n"
                    + f"{mpi_cmd}{pw2wan_exec} -in pw2wan.in | tee pw2wan.out\n"
                    + f"{wan_exec} wannier90\n"
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
        
        cluster = self.get_selected_cluster()
        mpi_ranks = self.spin_cores.value() if self.chk_mpi.isChecked() else 1
        self.qe_thread = QERunnerThread(script_content, out_dir, cluster, mpi_ranks=mpi_ranks)
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