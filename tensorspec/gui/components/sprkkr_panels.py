from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
                               QSpinBox, QDoubleSpinBox, QComboBox, QPushButton,
                               QLabel, QMessageBox, QLineEdit, QTextEdit, QCheckBox)

from PySide6.QtCore import Signal, QThread

import time
from pathlib import Path

from tensorspec.gui.services.cluster_utils import (
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


class ScfRunnerThread(QThread):
    """Runs core.dft.sprkkr.run_scf off the GUI thread. wait=True + poll_cb."""
    progress = Signal(object)
    finished_signal = Signal(bool, object, str)

    def __init__(self, structure, params, workdir, launcher, nproc, parent=None, remote_workdir=None):
        super().__init__(parent)
        self.structure = structure
        self.params = params
        self.workdir = workdir
        self.launcher = launcher
        self.nproc = nproc
        self.remote_workdir = remote_workdir
        self.log_path = str(Path(workdir) / f"{params.dataset}_SCF.out")

    def run(self):
        from tensorspec.core.dft.sprkkr import run_scf
        try:
            result = run_scf(
                self.structure, self.params, self.workdir, self.launcher,
                nproc=self.nproc, wait=True,
                poll_cb=lambda st: self.progress.emit(st),
                remote_workdir=self.remote_workdir,
            )
            self.finished_signal.emit(True, result, "")
        except Exception as e:
            self.finished_signal.emit(False, None, str(e))


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

        from tensorspec.core.dft.sprkkr import load_settings, local_binaries_present
        settings = load_settings()

        self.edit_bin_dir = QLineEdit(settings.bin_dir)
        self.edit_bin_dir.setToolTip("Directory holding kkrscf9.7 / kkrspec9.7 binaries.")
        form.addRow("Local SPR-KKR bin dir:", self.edit_bin_dir)

        self.lbl_bin_status = QLabel()
        form.addRow("", self.lbl_bin_status)

        def _refresh_bin_status(_text=None):
            bins = local_binaries_present(self.edit_bin_dir.text().strip() or settings.bin_dir)
            self.lbl_bin_status.setText(
                f"kkrscf: {'OK' if bins.get('kkrscf') else 'missing'}   "
                f"kkrspec: {'OK' if bins.get('kkrspec') else 'missing'}"
            )
        self.edit_bin_dir.textChanged.connect(_refresh_bin_status)
        _refresh_bin_status()

        self.spin_nproc = QSpinBox()
        self.spin_nproc.setRange(1, 256)
        self.spin_nproc.setValue(settings.nproc)
        form.addRow("nproc:", self.spin_nproc)

        self.spin_niter = QSpinBox()
        self.spin_niter.setRange(1, 2000)
        self.spin_niter.setValue(200)
        form.addRow("SCF NITER:", self.spin_niter)

        self.spin_mix = QDoubleSpinBox()
        self.spin_mix.setRange(0.01, 1.0)
        self.spin_mix.setSingleStep(0.05)
        self.spin_mix.setValue(0.2)
        form.addRow("SCF MIX:", self.spin_mix)

        self.combo_vxc = QComboBox()
        self.combo_vxc.addItems(["VWN", "PBE"])
        form.addRow("VXC:", self.combo_vxc)

        self.chk_nonmag = QCheckBox("Non-magnetic")
        form.addRow("", self.chk_nonmag)

        self.chk_primitive = QCheckBox("Reduce to primitive cell")
        self.chk_primitive.setChecked(True)
        form.addRow("", self.chk_primitive)

        self.lbl_scf_status = QLabel("idle")
        form.addRow("SCF Status:", self.lbl_scf_status)

        self.txt_scf_tail = QTextEdit()
        self.txt_scf_tail.setReadOnly(True)
        self.txt_scf_tail.setMaximumHeight(100)
        form.addRow("Log tail:", self.txt_scf_tail)

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
            from tensorspec.core.dft.sprkkr import (
                ScfParams, LocalLauncher, RemoteLauncher, load_settings, local_binaries_present,
            )

            structure = struct
            # cif_lattice meta below uses THIS (pre-primitive-reduction) structure --
            # hkl the user types means "in this frame", not a possibly-reoriented
            # primitive cell (design doc §0 / resolve_surface_geometry).
            self._scf_orig_structure = struct
            if self.chk_primitive.isChecked() and len(structure) > 1:
                structure = structure.get_primitive_structure()

            params = ScfParams(
                nl=self.spin_lmax.value(),
                ne=self.spin_ne.value(),
                nktab=self.spin_nktab.value(),
                niter=self.spin_niter.value(),
                mix=self.spin_mix.value(),
                vxc=self.combo_vxc.currentText(),
                nonmag=self.chk_nonmag.isChecked(),
                dataset="scf",
            )

            ts = time.strftime("%Y%m%d_%H%M%S")
            remote = is_remote_target(self.combo_target)
            nproc = self.spin_nproc.value()
            cluster = None
            remote_workdir = None

            if remote:
                cluster = selected_cluster(self.combo_target)
                if not cluster:
                    QMessageBox.warning(
                        self,
                        "Error",
                        "No remote cluster configured. Add one in Compute Manager.",
                    )
                    return
                from tensorspec.core.compute import cluster_paths
                remote_workdir = f"{cluster_paths.job_dir(cluster, 'sprkkr')}/scf_{ts}"
                workdir = f"scratch/sprkkr_gui_run/scf_{ts}"
                launcher = RemoteLauncher(cluster)
            else:
                bin_dir = self.edit_bin_dir.text().strip() or load_settings().bin_dir
                bins = local_binaries_present(bin_dir)
                if not bins.get("kkrscf"):
                    QMessageBox.warning(
                        self,
                        "Missing Binary",
                        f"No kkrscf9.7 found in {bin_dir}. Set 'Local SPR-KKR bin dir' first.",
                    )
                    return
                workdir = f"scratch/sprkkr_gui_run/scf_{ts}"
                launcher = LocalLauncher(bin_dir)

            self._scf_structure = structure
            self._scf_params = params
            self._scf_workdir = workdir
            self._scf_remote = remote
            self._scf_cluster = cluster
            self._scf_name = f"{getattr(structure, 'formula', 'structure').replace(' ', '')}_{ts}"

            self.lbl_scf_status.setText("starting...")
            self.txt_scf_tail.clear()

            self.thread_scf = ScfRunnerThread(
                structure, params, workdir, launcher, nproc, remote_workdir=remote_workdir
            )
            self.thread_scf.progress.connect(self._on_scf_progress)
            self.thread_scf.finished_signal.connect(self._on_scf_finished)
            self.thread_scf.start()
            self.job_started.emit()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run SPRKKR:\n{str(e)}")

    def _on_scf_progress(self, status):
        self.lbl_scf_status.setText(
            f"iter {status.iterations}  err {status.last_err}  "
            f"EF {status.ef_ry} Ry  converged={status.converged}"
        )
        log_path = getattr(self.thread_scf, "log_path", None)
        if log_path and Path(log_path).exists():
            lines = Path(log_path).read_text(errors="ignore").splitlines()
            self.txt_scf_tail.setPlainText("\n".join(lines[-6:]))

    def _on_scf_finished(self, ok, result, err):
        if not ok:
            self.lbl_scf_status.setText("failed")
            QMessageBox.critical(self, "Error", f"SPRKKR SCF failed:\n{err}")
            return

        self.lbl_scf_status.setText(
            f"converged={result.status.converged}  EF={result.status.ef_ry} Ry"
        )

        if not self._scf_remote:
            from tensorspec.core.dft.sprkkr import Vault, pot_key, load_settings
            from tensorspec.core.workspace import global_workspace

            settings = load_settings()
            vault = Vault(settings.vault_root)
            key = pot_key(self._scf_structure, self._scf_params)
            orig_structure = getattr(self, "_scf_orig_structure", None) or self._scf_structure
            try:
                cif_lattice = orig_structure.lattice.matrix.tolist()
            except Exception:
                cif_lattice = None
            meta = {
                "ef_ry": result.status.ef_ry,
                "workdir": result.workdir,
                "cif_lattice": cif_lattice,
                "formula": getattr(orig_structure, "formula", None),
                "nonmag": self._scf_params.nonmag,
            }
            entry = vault.register(
                key=key,
                name=self._scf_name,
                pot_path=result.pot_path,
                meta=meta,
            )
            global_workspace.push_remote_run(
                name=self._scf_name,
                cluster_name="local",
                engine="SPRKKR",
                remote_path=entry.pot_path,
                meta=meta,
            )

        QMessageBox.information(
            self,
            "SPRKKR SCF Done",
            f"Converged: {result.status.converged}\n"
            f"EF = {result.status.ef_ry} Ry\n"
            f"Potential: {result.pot_path}",
        )


class ArpesRunnerThread(QThread):
    """Runs KKRWrapper().run_simulation off the GUI thread."""
    finished_signal = Signal(bool, object, str)

    def __init__(self, kwargs, parent=None):
        super().__init__(parent)
        self.kwargs = kwargs

    def run(self):
        from tensorspec.core.arpes.one_step.kkr_wrapper import KKRWrapper
        try:
            out = KKRWrapper().run_simulation({}, self.kwargs)
            self.finished_signal.emit(True, out, "")
        except Exception as e:
            self.finished_signal.emit(False, None, str(e))


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
        vault_name = self.combo_vault.currentText()
        if not vault_name or vault_name == "No SPRKKR Vaults Found":
            QMessageBox.warning(self, "Error", "No valid SPRKKR Vault selected! Run an SCF job first.")
            return

        if "BSF" in self.combo_task.currentText():
            QMessageBox.information(
                self, "Not Wired Yet", "BSF via SPR-KKR not wired yet (later gate)."
            )
            return

        try:
            from tensorspec.core.workspace import global_workspace
            from tensorspec.core.dft.sprkkr import load_settings, LocalLauncher, RemoteLauncher

            vault = global_workspace.get(vault_name)
            if not vault:
                QMessageBox.warning(self, "Error", "Vault entry not found.")
                return
            remote_path = vault.get('remote_path')
            vault_cluster_name = vault.get('cluster_name', vault.get('cluster', ''))

            if vault_cluster_name == "local":
                pot_path = remote_path
            else:
                pot_path = f"{remote_path}/scf.pot_new"

            polar_map = {
                "p (Linear P)": "P",
                "s (Linear S)": "S",
                "sp (Circular +)": "C+",
                "sm (Circular -)": "C-",
            }
            polarization = polar_map.get(self.combo_polar.currentText(), "P")
            hkl = (self.spin_h.value(), self.spin_k.value(), self.spin_l.value())

            settings = load_settings()
            ts = time.strftime("%Y%m%d_%H%M%S")

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
                from tensorspec.core.compute import cluster_paths
                launcher = RemoteLauncher(cluster)
                # workdir = LOCAL staging; remote_workdir = run dir on cluster.
                remote_workdir = f"{cluster_paths.job_dir(cluster, 'sprkkr')}/arpes_{ts}"
                workdir = f"scratch/sprkkr_gui_run/arpes_{ts}"
            else:
                launcher = LocalLauncher(settings.bin_dir)
                remote_workdir = None
                workdir = f"scratch/sprkkr_gui_run/arpes_{ts}"

            kwargs = {
                "photon_energy": self.spin_ephot.value(),
                "polarization": polarization,
                "work_function": self.spin_workf.value(),
                "hkl": hkl,
                "k_bounds": {"X": [-20.0, 20.0, 21], "Y": [0.0, 0.0, 1]},
                "e_steps": self.spin_ne.value(),
                "pot_path": pot_path,
                "launcher": launcher,
                "nproc": settings.nproc,
                "workdir": workdir,
                "remote_workdir": remote_workdir,
            }

            self._arpes_vault_name = vault_name
            self._arpes_ts = ts

            self.thread_arpes = ArpesRunnerThread(kwargs)
            self.thread_arpes.finished_signal.connect(self._on_arpes_finished)
            self.thread_arpes.start()
            self.job_started.emit()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run ARPES:\n{str(e)}")

    def _on_arpes_finished(self, ok, result, err):
        if not ok:
            QMessageBox.critical(self, "Error", f"Failed to run ARPES:\n{err}")
            return

        from tensorspec.core.workspace import global_workspace

        name = f"{self._arpes_vault_name}_arpes_{self._arpes_ts}"
        global_workspace.push_spectroscopy_data(name, result["tensor"])
        QMessageBox.information(
            self,
            "SPRKKR ARPES Done",
            f"ARPES simulation finished.\nPushed to Workspace as '{name}'.",
        )
