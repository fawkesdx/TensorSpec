from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
                               QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, 
                               QLabel, QMessageBox, QLineEdit)

from PySide6.QtCore import Signal

from tensorspec.gui.cluster_utils import (
    cluster_display_name,
    find_cluster_by_name,
    is_remote_target,
    populate_compute_target_combo,
    selected_cluster,
)


def _connect_cluster(cluster):
    import paramiko

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pwd = cluster.get("password", "") or None
    ssh.connect(
        cluster["host"],
        port=cluster.get("port", 22),
        username=cluster["user"],
        password=pwd,
        timeout=30,
    )
    return ssh


class SPRKKRDftPanel(QWidget):
    job_started = Signal()
    
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group = QGroupBox("SPRKKR SCF Configuration")
        form = QFormLayout(group)
        
        self.combo_target = QComboBox()
        populate_compute_target_combo(self.combo_target)
        self.combo_target.setToolTip(
            "Remote targets come from Compute Manager (~/.tensorspec_clusters.json)."
        )
        form.addRow("Compute Target:", self.combo_target)
        
        self.spin_lmax = QSpinBox()
        self.spin_lmax.setRange(2, 6)
        self.spin_lmax.setValue(3)
        self.spin_lmax.setToolTip("Angular momentum cutoff (lmax). 3=f-electrons.")
        form.addRow("LMAX Cutoff:", self.spin_lmax)
        
        self.spin_nktab = QSpinBox()
        self.spin_nktab.setRange(10, 1000)
        self.spin_nktab.setValue(250)
        form.addRow("k-points (NKTAB):", self.spin_nktab)
        
        self.spin_ne = QSpinBox()
        self.spin_ne.setRange(10, 500)
        self.spin_ne.setValue(30)
        form.addRow("Energy Grid (NE):", self.spin_ne)
        
        self.combo_rel = QComboBox()
        self.combo_rel.addItems(["Fully Relativistic (Dirac)", "Scalar Relativistic", "Non-Relativistic"])
        form.addRow("Relativity Mode:", self.combo_rel)
        
        self.btn_run = QPushButton("🚀 Run SPRKKR SCF")
        self.btn_run.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 5px;")
        self.btn_run.clicked.connect(self.run_scf)
        form.addRow(self.btn_run)
        
        self.btn_save_vault = QPushButton("💾 Save Remote Vault Checkpoint")
        self.btn_save_vault.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold; padding: 5px;")
        self.btn_save_vault.clicked.connect(self.save_remote_vault)
        form.addRow(self.btn_save_vault)
        
        layout.addWidget(group)
        layout.addStretch()

    def save_remote_vault(self):
        from PySide6.QtWidgets import QInputDialog
        from tensorspec.core.workspace import global_workspace
        
        if not is_remote_target(self.combo_target):
            QMessageBox.warning(
                self,
                "Not Supported",
                "Save Vault is only available for remote cluster jobs.",
            )
            return

        cluster = selected_cluster(self.combo_target)
        if not cluster:
            QMessageBox.warning(
                self,
                "Error",
                "No remote cluster configured. Add one in Compute Manager.",
            )
            return
            
        vault_name, ok = QInputDialog.getText(
            self,
            "Save Remote Vault",
            "Enter a permanent name for this checkpoint (e.g., Cu_SCF_Converged):",
        )
        if not ok or not vault_name:
            return
            
        try:
            user = cluster["user"]
            ssh = _connect_cluster(cluster)
            
            scratch_dir = f"/mnt/data/{user}/tensorspec_heavy/sprkkr_gui_run"
            perm_dir = f"/mnt/data/{user}/tensorspec_heavy/vaults/{vault_name}"
            
            cmd = f"mkdir -p /mnt/data/{user}/tensorspec_heavy/vaults && cp -r {scratch_dir} {perm_dir}"
            stdin, stdout, stderr = ssh.exec_command(cmd)
            exit_status = stdout.channel.recv_exit_status()
            ssh.close()
            
            if exit_status == 0:
                global_workspace.push_remote_run(
                    name=vault_name,
                    cluster_name=cluster_display_name(cluster),
                    engine="SPRKKR",
                    remote_path=perm_dir,
                )
                QMessageBox.information(
                    self,
                    "Success",
                    f"Remote folder copied and saved to Workspace as '{vault_name}'.\n\n"
                    "You can now load this in the ARPES suite.",
                )
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to copy directory on cluster.\n{stderr.read().decode()}",
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save vault:\n{str(e)}")

    def run_scf(self):
        struct = getattr(self.engine, "crystal_structure", None)
        if struct is None:
            QMessageBox.warning(self, "Error", "No structure loaded! Please load a structure first.")
            return
            
        try:
            from tensorspec.core.dft.sprkkr_generator import SPRKKRInputGenerator
            gen = SPRKKRInputGenerator(struct)
            import os
            out_dir = "scratch/sprkkr_gui_run"
            os.makedirs(out_dir, exist_ok=True)
            gen.write_scf_input(
                out_dir, 
                lmax=self.spin_lmax.value(), 
                nktab=self.spin_nktab.value(), 
                ne=self.spin_ne.value(),
                rel_mode=self.combo_rel.currentText(),
                niter=10
            )
            
            if is_remote_target(self.combo_target):
                cluster = selected_cluster(self.combo_target)
                if not cluster:
                    QMessageBox.warning(
                        self,
                        "Error",
                        "No remote cluster configured. Add one in Compute Manager.",
                    )
                    return

                user = cluster["user"]
                ssh = _connect_cluster(cluster)
                
                remote_dir = f"/mnt/data/{user}/tensorspec_heavy/sprkkr_gui_run"
                ssh.exec_command(f"mkdir -p {remote_dir}")
                import time; time.sleep(1)
                
                sftp = ssh.open_sftp()
                sftp.put(f"{out_dir}/scf.inp", f"{remote_dir}/scf.inp")
                sftp.put(f"{out_dir}/scf.pot", f"{remote_dir}/scf.pot")
                sftp.close()
                
                cmd = (
                    f"bash -c 'cd {remote_dir} && export TMPDIR=/mnt/data/{user}/tmp && "
                    f"mkdir -p $TMPDIR && "
                    f"export PATH=\"/home/{user}/miniconda3/envs/qe/bin:$PATH\" && "
                    f"export LD_LIBRARY_PATH=\"/home/{user}/miniconda3/envs/qe/lib:$LD_LIBRARY_PATH\" && "
                    f"nohup /mnt/data/{user}/tensorspec_heavy/SPRKKR/bin/kkrscf9.7 "
                    f"< scf.inp > scf.out.full 2>&1 &'"
                )
                ssh.exec_command(cmd)
                ssh.close()
                
                self.job_started.emit()
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"SPRKKR SCF submitted to {cluster_display_name(cluster)} "
                    f"({cluster['host']})!\n\nLogs: {remote_dir}/scf.out.full\n\n"
                    "When it finishes, click 'Save Remote Vault Checkpoint'.",
                )
            else:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Generated input files in {out_dir}. (Local executable not found, skipped run).",
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run SPRKKR:\n{str(e)}")

class SPRKKRArpesPanel(QWidget):
    job_started = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group = QGroupBox("SPRKKR Spectroscopy (kkrspec/kkrgen)")
        form = QFormLayout(group)
        
        vault_layout = QHBoxLayout()
        self.combo_vault = QComboBox()
        self.btn_refresh = QPushButton("🔄")
        self.btn_refresh.setMaximumWidth(40)
        self.btn_refresh.clicked.connect(self.refresh_vaults)
        vault_layout.addWidget(self.combo_vault)
        vault_layout.addWidget(self.btn_refresh)
        form.addRow("Load Vault:", vault_layout)
        
        self.combo_target = QComboBox()
        populate_compute_target_combo(self.combo_target)
        self.combo_target.setToolTip(
            "Remote targets come from Compute Manager (~/.tensorspec_clusters.json)."
        )
        form.addRow("Compute Target:", self.combo_target)
        
        self.combo_task = QComboBox()
        self.combo_task.addItems(["Bloch Spectral Function (BSF)", "One-Step Photoemission (ARPES)"])
        form.addRow("Task Type:", self.combo_task)
        
        self.spin_ne = QSpinBox()
        self.spin_ne.setRange(50, 1000)
        self.spin_ne.setValue(300)
        form.addRow("Energy Points:", self.spin_ne)
        
        self.spin_ephot = QDoubleSpinBox()
        self.spin_ephot.setRange(5.0, 2000.0)
        self.spin_ephot.setValue(21.2)
        self.spin_ephot.setSuffix(" eV")
        form.addRow("Photon Energy:", self.spin_ephot)
        
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(0.0, 1000.0)
        self.spin_temp.setValue(10.0)
        self.spin_temp.setSuffix(" K")
        form.addRow("Temperature:", self.spin_temp)
        
        self.spin_workf = QDoubleSpinBox()
        self.spin_workf.setRange(0.0, 10.0)
        self.spin_workf.setValue(4.5)
        self.spin_workf.setSuffix(" eV")
        form.addRow("Work Function:", self.spin_workf)
        
        self.combo_polar = QComboBox()
        self.combo_polar.addItems(["p (Linear P)", "s (Linear S)", "sp (Circular +)", "sm (Circular -)"])
        form.addRow("Polarization:", self.combo_polar)
        
        hkl_layout = QHBoxLayout()
        self.spin_h = QSpinBox(); self.spin_h.setRange(-5,5); self.spin_h.setValue(0)
        self.spin_k = QSpinBox(); self.spin_k.setRange(-5,5); self.spin_k.setValue(0)
        self.spin_l = QSpinBox(); self.spin_l.setRange(-5,5); self.spin_l.setValue(1)
        hkl_layout.addWidget(self.spin_h); hkl_layout.addWidget(self.spin_k); hkl_layout.addWidget(self.spin_l)
        form.addRow("Surface Miller (hkl):", hkl_layout)
        
        self.btn_run = QPushButton("🚀 Run SPRKKR Spectroscopy")
        self.btn_run.setStyleSheet("background-color: #17a2b8; color: white; font-weight: bold; padding: 5px;")
        self.btn_run.clicked.connect(self.run_arpes)
        form.addRow(self.btn_run)
        
        layout.addWidget(group)
        layout.addStretch()
        self.refresh_vaults()
        
    def refresh_vaults(self):
        from tensorspec.core.workspace import global_workspace
        self.combo_vault.clear()
        runs = global_workspace.list_remote_runs(engine="SPRKKR")
        if runs:
            self.combo_vault.addItems(runs)
        else:
            self.combo_vault.addItem("No SPRKKR Vaults Found")

    def run_arpes(self):
        try:
            import os
            vault_name = self.combo_vault.currentText()
            if not vault_name or vault_name == "No SPRKKR Vaults Found":
                QMessageBox.warning(self, "Error", "No valid SPRKKR Vault selected! Run an SCF job first.")
                return
                
            from tensorspec.core.workspace import global_workspace
            vault = global_workspace.get(vault_name)
            remote_dir = vault.get('remote_path')
            vault_cluster_name = vault.get('cluster_name', vault.get('cluster', ''))
            
            task_str = "BSF" if "BSF" in self.combo_task.currentText() else "ARPES"
            polar = self.combo_polar.currentText().split()[0]
            
            from tensorspec.core.dft.sprkkr_generator import SPRKKRInputGenerator
            gen = SPRKKRInputGenerator(None)
            
            out_dir = "scratch/sprkkr_gui_run"
            gen.write_arpes_input(
                out_dir,
                task=task_str,
                ne=self.spin_ne.value(),
                ephot=self.spin_ephot.value(),
                temp=self.spin_temp.value(),
                workf=self.spin_workf.value(),
                polar=polar,
                hkl=(self.spin_h.value(), self.spin_k.value(), self.spin_l.value())
            )
            
            if is_remote_target(self.combo_target):
                cluster = (
                    find_cluster_by_name(vault_cluster_name)
                    or selected_cluster(self.combo_target)
                )
                if not cluster:
                    QMessageBox.warning(
                        self,
                        "Error",
                        "No remote cluster configured. Add one in Compute Manager.",
                    )
                    return

                user = cluster["user"]
                ssh = _connect_cluster(cluster)
                
                sftp = ssh.open_sftp()
                sftp.put(f"{out_dir}/sys.inp", f"{remote_dir}/sys.inp")
                sftp.close()
                
                binary = "kkrspec9.7" if task_str == "ARPES" else "kkrgen9.7"
                
                cmd = (
                    f"bash -c 'cd {remote_dir} && export TMPDIR=/mnt/data/{user}/tmp && "
                    f"mkdir -p $TMPDIR && "
                    f"export PATH=\"/home/{user}/miniconda3/envs/qe/bin:$PATH\" && "
                    f"export LD_LIBRARY_PATH=\"/home/{user}/miniconda3/envs/qe/lib:$LD_LIBRARY_PATH\" && "
                    f"nohup /mnt/data/{user}/tensorspec_heavy/SPRKKR/bin/{binary} "
                    f"< sys.inp > sys.out.full 2>&1 &'"
                )
                ssh.exec_command(cmd)
                ssh.close()
                self.job_started.emit()
                QMessageBox.information(
                    self,
                    "Success",
                    f"SPRKKR {task_str} submitted to {cluster_display_name(cluster)} "
                    f"({cluster['host']})!\n\nLogs: {remote_dir}/sys.out.full\n\n"
                    "Check 'Calculation Live Logs' to watch progress.",
                )
            else:
                QMessageBox.information(
                    self,
                    "Success",
                    "ARPES calculation finished successfully (Local Mock)!\n\n"
                    "You can now load 'scratch/copper_arpes_mock.npz' in the Data Viewer.",
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run ARPES:\n{str(e)}")
