"""Upload TB band jobs to a remote cluster and fetch tb_bands_result.npz."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

RUNNER_MODULE = Path(__file__).with_name("tb_remote_runner.py")
JOB_NAME = "tb_job.json"
RESULT_NAME = "tb_bands_result.npz"
HR_NAME = "wannier90_hr.dat"


def remote_run_dir(cluster: Dict[str, Any]) -> str:
    user = cluster.get("user") or "user"
    return f"/mnt/data/{user}/tensorspec_heavy/tb_gui_run"


def _connect(cluster: Dict[str, Any]):
    import paramiko

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pwd = cluster.get("password") or None
    ssh.connect(
        cluster["host"],
        port=int(cluster.get("port", 22)),
        username=cluster["user"],
        password=pwd,
        timeout=30,
    )
    return ssh


def _mkdir_p(sftp, path: str) -> None:
    parts = path.strip("/").split("/")
    cur = ""
    for part in parts:
        cur = f"{cur}/{part}"
        try:
            sftp.mkdir(cur)
        except OSError:
            pass


def build_job_payload(
    structure,
    k_vecs: np.ndarray,
    *,
    fermi_energy: float = 0.0,
    need_eigenvectors: bool = True,
    diag_engine: str = "chinook",
    diag_device: str = "cpu",
    use_soc: bool = False,
    soc_strength: float = 0.5,
    onsite_e: float = 0.0,
    orbital_shifts: Optional[dict] = None,
    custom_hopping: Optional[dict] = None,
    cutoffs: Optional[list] = None,
    tb_mode: str = "Simple Scalar",
    w90_basename: Optional[str] = None,
) -> dict:
    return {
        "structure": structure.as_dict(),
        "k_vecs": np.asarray(k_vecs, dtype=float).tolist(),
        "fermi_energy": float(fermi_energy),
        "need_eigenvectors": bool(need_eigenvectors),
        "diag_engine": str(diag_engine),
        "diag_device": str(diag_device),
        "use_soc": bool(use_soc),
        "soc_strength": float(soc_strength),
        "onsite_e": float(onsite_e),
        "orbital_shifts": orbital_shifts or {},
        "custom_hopping": custom_hopping or {},
        "cutoffs": cutoffs,
        "tb_mode": tb_mode,
        "w90_basename": w90_basename,
    }


def remote_python_bin(cluster: Dict[str, Any]) -> str:
    user = cluster.get("user") or "user"
    return f"/home/{user}/TensorSpec/TensorSpec_env/bin/python"


def run_remote_tb_bands(
    cluster: Dict[str, Any],
    job: dict,
    w90_filepath: Optional[str],
    *,
    python_bin: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    poll_s: float = 3.0,
    timeout_s: float = 7200.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], list, float]:
    """Run job on cluster; return (eigenvalues, eigenvectors, orb_labels, fermi_energy)."""
    log = log_fn or (lambda _msg: None)
    remote_dir = remote_run_dir(cluster)
    user = cluster.get("user") or "user"
    python_bin = python_bin or remote_python_bin(cluster)
    tensorspec_home = f"/home/{user}/TensorSpec"

    ssh = _connect(cluster)
    try:
        sftp = ssh.open_sftp()
        _mkdir_p(sftp, remote_dir)

        log(f"Uploading TB job to {remote_dir} ...")
        with sftp.file(f"{remote_dir}/{JOB_NAME}", "w") as rf:
            rf.write(json.dumps(job))

        with open(RUNNER_MODULE, "r") as f:
            runner_src = f.read()
        with sftp.file(f"{remote_dir}/tb_remote_runner.py", "w") as rf:
            rf.write(runner_src)

        if w90_filepath and os.path.isfile(w90_filepath):
            work_dir = os.path.dirname(os.path.abspath(w90_filepath))
            with sftp.file(f"{remote_dir}/{HR_NAME}", "wb") as rf:
                with open(w90_filepath, "rb") as lf:
                    rf.write(lf.read())
            for aux in ("wannier90.wout", "scf.out", "nscf.out"):
                local_aux = os.path.join(work_dir, aux)
                if os.path.isfile(local_aux):
                    with sftp.file(f"{remote_dir}/{aux}", "wb") as rf:
                        with open(local_aux, "rb") as lf:
                            rf.write(lf.read())
            # Any other .wout in folder (named wannier90.win run)
            for name in os.listdir(work_dir):
                if name.endswith(".wout") and name not in ("wannier90.wout",):
                    local_aux = os.path.join(work_dir, name)
                    if os.path.isfile(local_aux):
                        with sftp.file(f"{remote_dir}/{name}", "wb") as rf:
                            with open(local_aux, "rb") as lf:
                                rf.write(lf.read())

        sftp.close()

        env_exports = f"export PYTHONPATH={tensorspec_home}:$PYTHONPATH"
        if str(job.get("diag_device", "cpu")).lower() == "cuda":
            # Prefer the less-loaded GPU when multiple devices exist.
            env_exports += " && export CUDA_VISIBLE_DEVICES=1"

        cmd = (
            f"bash -c 'cd {remote_dir} && {env_exports} && "
            f"rm -f {RESULT_NAME} tb_remote.log && "
            f"{python_bin} tb_remote_runner.py --job {JOB_NAME} "
            f"--out {RESULT_NAME} > tb_remote.log 2>&1'"
        )
        log("Running remote band diagonalization (auto-download when done)...")
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout_s)
        exit_code = stdout.channel.recv_exit_status()
        out_tail = stdout.read().decode(errors="replace")[-2000:]
        err_tail = stderr.read().decode(errors="replace")[-1000:]
        if exit_code != 0:
            try:
                _, tail_stdout, _ = ssh.exec_command(
                    f"tail -40 {remote_dir}/tb_remote.log", timeout=15
                )
                log_tail = tail_stdout.read().decode(errors="replace")
            except Exception:
                log_tail = out_tail + err_tail
            raise RuntimeError(
                f"Remote TB runner failed (exit {exit_code}).\n{log_tail}"
            )

        local_npz = os.path.join(
            os.path.expanduser("~"), ".tensorspec_cache", RESULT_NAME
        )
        os.makedirs(os.path.dirname(local_npz), exist_ok=True)
        sftp = ssh.open_sftp()
        sftp.get(f"{remote_dir}/{RESULT_NAME}", local_npz)
        sftp.close()
        log(f"Downloaded {RESULT_NAME}")

        data = np.load(local_npz, allow_pickle=True)
        eigenvalues = np.asarray(data["eigenvalues"], dtype=float)
        eigenvectors = None
        if "eigenvectors" in data:
            eigenvectors = np.asarray(data["eigenvectors"])
        orb_labels = [str(x) for x in data["orb_labels"].tolist()]
        fermi_energy = float(data.get("fermi_energy", job.get("fermi_energy", 0.0)))
        return eigenvalues, eigenvectors, orb_labels, fermi_energy
    finally:
        ssh.close()
