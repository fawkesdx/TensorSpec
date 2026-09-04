import os
import subprocess
import stat as statmod
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QWidget, QFormLayout, QGroupBox, QComboBox, 
                               QSpinBox, QHBoxLayout, QLineEdit, QPushButton, 
                               QMessageBox, QPlainTextEdit, QVBoxLayout, QCheckBox, QLabel,
                               QProgressDialog, QApplication)

from tensorspec.core.dft.qe_generator import QEInputGenerator
from tensorspec.core.compute import cluster_paths as cp
from tensorspec.gui.services.nersc_auth import refresh_sshproxy_login


class QEFetchThread(QThread):
    """Download ARPES-useful QE/Wannier outputs from remote job dir."""

    progress = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, cluster, out_dir):
        super().__init__()
        self.cluster = cluster
        self.out_dir = out_dir

    def run(self):
        try:
            import paramiko

            self.progress.emit("Connecting...")
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cp.ssh_connect(ssh, self.cluster, timeout=20)
            sftp = ssh.open_sftp()
            remote_dir = cp.job_dir(self.cluster, "qe")
            os.makedirs(self.out_dir, exist_ok=True)

            attrs = sftp.listdir_attr(remote_dir)
            candidates = [
                a
                for a in attrs
                if statmod.S_ISREG(a.st_mode) and cp.is_arpes_fetch_candidate(a.filename)
            ]
            if not candidates:
                sftp.close()
                ssh.close()
                self.finished_signal.emit(
                    False, f"No ARPES package files found in {remote_dir}"
                )
                return

            fetched = []
            for i, file_attr in enumerate(candidates, 1):
                name = file_attr.filename
                size_mb = (file_attr.st_size or 0) / (1024 * 1024)
                self.progress.emit(
                    f"[{i}/{len(candidates)}] {name} ({size_mb:.1f} MB)..."
                )
                sftp.get(f"{remote_dir}/{name}", os.path.join(self.out_dir, name))
                fetched.append(f"{name} ({size_mb:.1f} MB)")

            sftp.close()
            ssh.close()
            summary = "\n".join(fetched)
            self.finished_signal.emit(
                True,
                f"Fetched {len(fetched)} file(s) into '{self.out_dir}':\n\n{summary}",
            )
        except Exception as exc:
            self.finished_signal.emit(False, str(exc))


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

        # Parallel setup — separate ranks for pw.x vs pw2wannier90 (FFT limit on Wannier step)
        parallel_layout = QHBoxLayout()
        self.chk_mpi = QCheckBox("Use MPI")
        self.chk_mpi.setChecked(True)
        self.spin_pw_cores = QSpinBox()
        self.spin_pw_cores.setRange(1, 256)
        self.spin_pw_cores.setValue(128)
        self.spin_pw_cores.setToolTip("MPI ranks for pw.x (scf + nscf). Perlmutter: up to 128 on one CPU node.")
        self.spin_wannier_cores = QSpinBox()
        self.spin_wannier_cores.setRange(1, 256)
        self.spin_wannier_cores.setValue(36)
        self.spin_wannier_cores.setToolTip(
            "MPI ranks for pw2wannier90.x only. Must not exceed smallest FFT dimension "
            "(often ≤36 for this cell). wannier90.x stays serial."
        )
        parallel_layout.addWidget(self.chk_mpi)
        parallel_layout.addWidget(QLabel("pw.x"))
        parallel_layout.addWidget(self.spin_pw_cores)
        parallel_layout.addWidget(QLabel("pw2wan"))
        parallel_layout.addWidget(self.spin_wannier_cores)
        parallel_layout.addStretch()
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

        fetch_row = QHBoxLayout()
        self.btn_nersc_login = QPushButton("🔑 Refresh NERSC Login")
        self.btn_nersc_login.setToolTip(
            "Run sshproxy for NERSC MFA (Perlmutter). Hidden for Daemon / local."
        )
        self.btn_nersc_login.clicked.connect(self.refresh_nersc_login)
        self.btn_fetch_wan = QPushButton("📥 Fetch ARPES Package")
        self.btn_fetch_wan.setStyleSheet("""
            QPushButton { background-color: #17a2b8; color: white; font-weight: bold; padding: 5px; }
            QPushButton:disabled { background-color: #0c5460; color: #888888; }
        """)
        self.btn_fetch_wan.setToolTip(
            "Download wannier90_hr.dat + logs/inputs for ARPES. "
            "Skips .mmn/.amn/wavefunctions."
        )
        self.btn_fetch_wan.clicked.connect(self.fetch_remote_outputs)
        fetch_row.addWidget(self.btn_nersc_login)
        fetch_row.addWidget(self.btn_fetch_wan)
        qe_form.addRow(fetch_row)

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
        self.chk_mpi.stateChanged.connect(self._sync_mpi_spinboxes)
        self.combo_cluster.currentIndexChanged.connect(self._on_cluster_changed)
        self._sync_mpi_spinboxes()
        self._on_cluster_changed()

    def _on_cluster_changed(self, _index=None):
        self._adapt_script_to_cluster()
        cluster = self.get_selected_cluster()
        self.btn_nersc_login.setVisible(cp.uses_sshproxy(cluster))

    def refresh_nersc_login(self):
        cluster = self.get_selected_cluster()
        if not cluster:
            QMessageBox.information(self, "Info", "Select Perlmutter (NERSC) as Compute Target.")
            return
        refresh_sshproxy_login(self, cluster)


    def _sync_mpi_spinboxes(self, _state=None):
        enabled = self.chk_mpi.isChecked()
        self.spin_pw_cores.setEnabled(enabled)
        self.spin_wannier_cores.setEnabled(enabled)

    def _sync_pw_backend_ui(self, _index=None):
        use_gpu = self.combo_pw_backend.currentData() == "gpu"
        self.combo_gpu_device.setEnabled(use_gpu)
        if use_gpu and self.chk_mpi.isChecked():
            dev = self.combo_gpu_device.currentData() or "0"
            n_ranks = 2 if "," in str(dev) else 1
            if self.spin_pw_cores.value() != n_ranks:
                self.spin_pw_cores.setValue(n_ranks)
            if self.spin_wannier_cores.value() != n_ranks:
                self.spin_wannier_cores.setValue(n_ranks)

    def _mpi_rank_settings(self):
        """Return (pw_ranks, wannier_ranks, slurm_ntasks) for script / Slurm generation."""
        if not self.chk_mpi.isChecked():
            return 1, 1, 1
        pw_ranks = max(1, int(self.spin_pw_cores.value()))
        wan_ranks = max(1, int(self.spin_wannier_cores.value()))
        return pw_ranks, wan_ranks, max(pw_ranks, wan_ranks)

    def _pw_backend_flags(self):
        """Return (use_gpu, cuda_devices, pw_ranks, wannier_ranks, slurm_ntasks)."""
        use_gpu = self.combo_pw_backend.currentData() == "gpu"
        pw_ranks, wan_ranks, slurm_ntasks = self._mpi_rank_settings()
        if not use_gpu:
            return False, "", pw_ranks, wan_ranks, slurm_ntasks
        dev = str(self.combo_gpu_device.currentData() or "0")
        gpu_ranks = dev.count(",") + 1
        return True, dev, gpu_ranks, gpu_ranks, gpu_ranks

    def _pipeline_env_header(self, use_gpu: bool, cuda_devices: str) -> str:
        if not use_gpu:
            return "export OMP_NUM_THREADS=1\n\n"
        return (
            "export OMP_NUM_THREADS=1\n"
            f"export CUDA_VISIBLE_DEVICES={cuda_devices}\n"
            "# QE GPU: pw.x must be CUDA build; MPI ranks ≈ number of GPUs\n\n"
        )

    def _adapt_script_to_cluster(self, _index=None):
        """Rewrite MPI launcher in Pipeline Script for current Compute Target."""
        text = self.script_editor.toPlainText()
        if not text.strip():
            return
        adapted = cp.adapt_pipeline_mpi_launcher(text, self.get_selected_cluster())
        if adapted != text:
            self.script_editor.setPlainText(adapted)

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
        if not isinstance(selected_data, dict):
            return None
        # Re-read ~/.tensorspec_clusters.json so edits apply without restarting GUI.
        config_file = os.path.expanduser("~/.tensorspec_clusters.json")
        if os.path.exists(config_file):
            try:
                import json

                with open(config_file, encoding="utf-8") as f:
                    clusters = json.load(f)
                key_name = selected_data.get("name")
                key_host = selected_data.get("host")
                for cluster in clusters:
                    if key_name and cluster.get("name") == key_name:
                        return cluster
                    if key_host and cluster.get("host") == key_host:
                        return cluster
            except Exception as exc:
                print(f"WARN: could not reload cluster config: {exc}", flush=True)
        return selected_data


    def fetch_remote_outputs(self):
        cluster = self.get_selected_cluster()
        if not cluster or cluster == "local":
            QMessageBox.information(self, "Info", "Select a remote cluster to fetch from.")
            return

        out_dir = self.line_outdir.text()
        os.makedirs(out_dir, exist_ok=True)

        if cp.uses_sshproxy(cluster):
            reply = QMessageBox.question(
                self,
                "NERSC Auth",
                "Fetch needs a valid NERSC key.\n\n"
                "Refresh NERSC Login first if you have not today.\n\n"
                "Continue fetch now?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self.btn_fetch_wan.setEnabled(False)
        self._fetch_progress = QProgressDialog(
            "Fetching ARPES package from remote...", None, 0, 0, self
        )
        self._fetch_progress.setWindowTitle("Fetch ARPES Package")
        self._fetch_progress.setMinimumDuration(0)
        self._fetch_progress.setCancelButton(None)
        self._fetch_progress.show()

        self._fetch_thread = QEFetchThread(cluster, out_dir)
        self._fetch_thread.progress.connect(self._on_fetch_progress)
        self._fetch_thread.finished_signal.connect(self._on_fetch_finished)
        self._fetch_thread.start()

    def _on_fetch_progress(self, text):
        if getattr(self, "_fetch_progress", None):
            self._fetch_progress.setLabelText(text)
        QApplication.processEvents()

    def _on_fetch_finished(self, success, message):
        self.btn_fetch_wan.setEnabled(True)
        if getattr(self, "_fetch_progress", None):
            self._fetch_progress.close()
            self._fetch_progress = None
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Fetch Failed", message)
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
            use_gpu, cuda_devices, pw_ranks, wan_ranks, _slurm_ntasks = self._pw_backend_flags()
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
            
            cluster = self.get_selected_cluster()
            pw_mpi_cmd = cp.mpi_launch_prefix(
                cluster,
                pw_ranks,
                use_mpi=self.chk_mpi.isChecked(),
            )
            wan_mpi_cmd = cp.mpi_launch_prefix(
                cluster,
                wan_ranks,
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
                    f"{pw_mpi_cmd}{pw_exec} -in scf.in | Tee-Object -FilePath scf.out\n"
                    f"{pw_mpi_cmd}{pw_exec} -in nscf.in | Tee-Object -FilePath nscf.out\n"
                    f"{wan_exec} -pp wannier90\n"
                    f"{wan_mpi_cmd}{pw2wan_exec} -in pw2wan.in | Tee-Object -FilePath pw2wan.out\n"
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
                    + f"# pw.x ranks={pw_ranks}; pw2wannier90 ranks={wan_ranks}\n"
                    + f"{pw_mpi_cmd}{pw_exec} -in scf.in | tee scf.out\n"
                    + f"{pw_mpi_cmd}{pw_exec} -in nscf.in | tee nscf.out\n"
                    + f"{wan_exec} -pp wannier90\n"
                    + f"{wan_mpi_cmd}{pw2wan_exec} -in pw2wan.in | tee pw2wan.out\n"
                    + f"{wan_exec} wannier90\n"
                )
            
            self.script_editor.setPlainText(script_text)
            
            QMessageBox.information(self, "Success", f"Inputs generated in {out_dir}.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate files:\n{str(e)}")

    def run_qe_script(self):
        out_dir = self.line_outdir.text()
        cluster = self.get_selected_cluster()
        # Always adapt launcher to current target (SLURM→srun, else mpirun).
        script_content = cp.adapt_pipeline_mpi_launcher(
            self.script_editor.toPlainText(), cluster
        ).strip()
        if script_content != self.script_editor.toPlainText().strip():
            self.script_editor.setPlainText(script_content)

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
        
        _pw_ranks, _wan_ranks, slurm_ntasks = self._mpi_rank_settings()
        launcher = "srun" if cp.is_slurm(cluster) else "mpirun"
        self.log_display.appendPlainText(
            f"MPI launcher for this run: {launcher} (target={'SLURM' if cp.is_slurm(cluster) else 'Daemon/local'})"
        )
        self.qe_thread = QERunnerThread(
            script_content, out_dir, cluster, mpi_ranks=slurm_ntasks
        )
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