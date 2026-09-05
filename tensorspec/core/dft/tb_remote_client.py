"""Upload TB band jobs to a remote cluster and fetch tb_bands_result.npz."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from tensorspec.core.compute.cluster_paths import heavy_root, job_dir, python_bin as cluster_python_bin, repo_root
from tensorspec.core.dft.w90_tb_cache import (
    REMOTE_CACHE_NAME,
    cache_key,
    local_cache_path,
)

RUNNER_MODULE = Path(__file__).with_name("tb_remote_runner.py")
JOB_NAME = "tb_job.json"
RESULT_NAME = "tb_bands_result.npz"
HR_NAME = "wannier90_hr.dat"
PID_NAME = "tb_remote.pid"


class TBCancelled(Exception):
    """User cancelled a hybrid TB band job."""


def kill_remote_tb_job(cluster: Dict[str, Any], log_fn: Optional[Callable[[str], None]] = None) -> None:
    """Stop tb_remote_runner on cluster (best-effort)."""
    log = log_fn or (lambda _msg: None)
    remote_dir = remote_run_dir(cluster)
    ssh = _connect(cluster)
    try:
        cmd = (
            f"bash -c '"
            f"if [ -f {remote_dir}/{PID_NAME} ]; then kill $(cat {remote_dir}/{PID_NAME}) 2>/dev/null; fi; "
            f"pkill -f \"{remote_dir}/tb_remote_runner.py\" 2>/dev/null; "
            f"rm -f {remote_dir}/{PID_NAME}; "
            f"echo killed'"
        )
        _, stdout, _ = ssh.exec_command(cmd, timeout=20)
        log(stdout.read().decode(errors="replace").strip() or "Remote TB job stop requested.")
    finally:
        ssh.close()


def remote_run_dir(cluster: Dict[str, Any]) -> str:
    return job_dir(cluster, "tb")


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
    w90_cache_key: Optional[str] = None,
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
        "w90_cache_basename": REMOTE_CACHE_NAME if w90_cache_key else None,
        "w90_cache_key": w90_cache_key,
    }


def _remote_file_size(sftp, path: str) -> Optional[int]:
    try:
        return int(sftp.stat(path).st_size)
    except OSError:
        return None


def _upload_if_changed(sftp, local_path: str, remote_path: str, log: Callable[[str], None]) -> bool:
    """Upload when missing or size differs. Returns True if uploaded."""
    local_size = os.path.getsize(local_path)
    remote_size = _remote_file_size(sftp, remote_path)
    if remote_size == local_size:
        log(f"Skip upload (unchanged): {os.path.basename(local_path)}")
        return False
    with sftp.file(remote_path, "wb") as rf:
        with open(local_path, "rb") as lf:
            rf.write(lf.read())
    log(f"Uploaded {os.path.basename(local_path)} ({local_size // (1024 * 1024)} MB)")
    return True


def remote_python_bin(cluster: Dict[str, Any]) -> str:
    return cluster_python_bin(cluster)


def run_remote_tb_bands(
    cluster: Dict[str, Any],
    job: dict,
    w90_filepath: Optional[str],
    *,
    python_bin: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    poll_s: float = 3.0,
    timeout_s: float = 7200.0,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], list, float]:
    """Run job on cluster; return (eigenvalues, eigenvectors, orb_labels, fermi_energy)."""
    log = log_fn or (lambda _msg: None)
    remote_dir = remote_run_dir(cluster)
    python_bin = python_bin or remote_python_bin(cluster)
    tensorspec_home = repo_root(cluster)
    t_total = time.perf_counter()

    ssh = _connect(cluster)
    try:
        sftp = ssh.open_sftp()
        _mkdir_p(sftp, remote_dir)

        t_up = time.perf_counter()
        log(f"Uploading TB job to {remote_dir} ...")
        with sftp.file(f"{remote_dir}/{JOB_NAME}", "w") as rf:
            rf.write(json.dumps(job))

        with open(RUNNER_MODULE, "r") as f:
            runner_src = f.read()
        with sftp.file(f"{remote_dir}/tb_remote_runner.py", "w") as rf:
            rf.write(runner_src)

        w90_cache_key = job.get("w90_cache_key")
        cache_uploaded = False
        if w90_filepath and os.path.isfile(w90_filepath):
            if w90_cache_key:
                local_cache = local_cache_path(w90_cache_key)
                if local_cache.is_file():
                    remote_cache = f"{remote_dir}/{REMOTE_CACHE_NAME}"
                    cache_uploaded = _upload_if_changed(
                        sftp, str(local_cache), remote_cache, log
                    )

            if not cache_uploaded:
                remote_hr = f"{remote_dir}/{HR_NAME}"
                _upload_if_changed(sftp, w90_filepath, remote_hr, log)
                work_dir = os.path.dirname(os.path.abspath(w90_filepath))
                for aux in ("wannier90.wout", "scf.out", "nscf.out"):
                    local_aux = os.path.join(work_dir, aux)
                    if os.path.isfile(local_aux):
                        _upload_if_changed(sftp, local_aux, f"{remote_dir}/{aux}", log)
                for name in os.listdir(work_dir):
                    if name.endswith(".wout") and name not in ("wannier90.wout",):
                        local_aux = os.path.join(work_dir, name)
                        if os.path.isfile(local_aux):
                            _upload_if_changed(
                                sftp, local_aux, f"{remote_dir}/{name}", log
                            )

        log(f"Upload phase: {time.perf_counter() - t_up:.1f}s")
        sftp.close()

        env_exports = f"export PYTHONPATH={tensorspec_home}:$PYTHONPATH"
        if str(job.get("diag_device", "cpu")).lower() == "cuda":
            # Prefer the less-loaded GPU when multiple devices exist.
            env_exports += " && export CUDA_VISIBLE_DEVICES=1"

        start_cmd = (
            f"bash -c 'cd {remote_dir} && {env_exports} && "
            f"rm -f {RESULT_NAME} tb_remote.log {PID_NAME} && "
            f"nohup {python_bin} tb_remote_runner.py --job {JOB_NAME} "
            f"--out {RESULT_NAME} > tb_remote.log 2>&1 & echo $! > {PID_NAME}'"
        )
        log("Starting remote band diagonalization (poll + cancelable)...")
        t_run = time.perf_counter()
        _, start_out, start_err = ssh.exec_command(start_cmd, timeout=60)
        if start_out.channel.recv_exit_status() != 0:
            err = start_err.read().decode(errors="replace")
            raise RuntimeError(f"Failed to start remote TB runner.\n{err}")

        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if cancel_check and cancel_check():
                kill_remote_tb_job(cluster, log_fn=log)
                raise TBCancelled("Hybrid TB band run cancelled.")

            sftp = ssh.open_sftp()
            try:
                if _remote_file_size(sftp, f"{remote_dir}/{RESULT_NAME}") is not None:
                    break
            finally:
                sftp.close()

            _, poll_out, _ = ssh.exec_command(
                f"bash -c 'test -f {remote_dir}/{PID_NAME} && kill -0 $(cat {remote_dir}/{PID_NAME}) 2>/dev/null && echo running || echo stopped'",
                timeout=15,
            )
            poll_out.channel.recv_exit_status()
            state = poll_out.read().decode(errors="replace").strip()
            if state == "stopped":
                _, tail_out, _ = ssh.exec_command(
                    f"tail -30 {remote_dir}/tb_remote.log", timeout=15
                )
                tail_out.channel.recv_exit_status()
                log_tail = tail_out.read().decode(errors="replace")
                raise RuntimeError(
                    f"Remote TB runner exited before writing {RESULT_NAME}.\n{log_tail}"
                )
            time.sleep(poll_s)
        else:
            kill_remote_tb_job(cluster, log_fn=log)
            raise TimeoutError(f"Remote TB bands timed out after {timeout_s:.0f}s")

        log(f"Remote compute: {time.perf_counter() - t_run:.1f}s")

        t_dl = time.perf_counter()
        local_npz = os.path.join(
            os.path.expanduser("~"), ".tensorspec_cache", RESULT_NAME
        )
        os.makedirs(os.path.dirname(local_npz), exist_ok=True)
        sftp = ssh.open_sftp()
        sftp.get(f"{remote_dir}/{RESULT_NAME}", local_npz)
        if w90_filepath and w90_cache_key:
            try:
                remote_cache = f"{remote_dir}/{REMOTE_CACHE_NAME}"
                if _remote_file_size(sftp, remote_cache) is not None:
                    local_cache = local_cache_path(w90_cache_key)
                    local_cache.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(remote_cache, str(local_cache))
                    log("Synced remote W90 TB cache locally")
            except OSError:
                pass
        sftp.close()
        log(f"Download: {time.perf_counter() - t_dl:.1f}s")

        data = np.load(local_npz, allow_pickle=True)
        eigenvalues = np.asarray(data["eigenvalues"], dtype=float)
        eigenvectors = None
        if "eigenvectors" in data:
            eigenvectors = np.asarray(data["eigenvectors"])
        orb_labels = [str(x) for x in data["orb_labels"].tolist()]
        fermi_energy = float(data.get("fermi_energy", job.get("fermi_energy", 0.0)))
        log(f"Total remote TB wall: {time.perf_counter() - t_total:.1f}s")
        return eigenvalues, eigenvectors, orb_labels, fermi_energy
    finally:
        ssh.close()
