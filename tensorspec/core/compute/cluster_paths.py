"""Resolve remote filesystem paths from ~/.tensorspec_clusters.json entries.

Every GUI remote dispatch (ARPES, QE, SPR-KKR, live monitor) should use these
helpers instead of hard-coded /mnt/data or /home/{user}/TensorSpec paths.

Cluster JSON (optional ``paths`` block — omit for LBL-style defaults):

    {
      "name": "my-gpu",
      "host": "gpu.example.edu",
      "user": "alice",
      "mode": "Daemon",
      "paths": {
        "heavy_root": "/scratch/alice/tensorspec_heavy",
        "tmp_dir": "/scratch/alice/tmp",
        "repo_root": "/home/alice/TensorSpec",
        "python": "/home/alice/TensorSpec/TensorSpec_env/bin/python",
        "sprkkr_bin": "/opt/sprkkr/bin"
      },
      "slurm": {
        "account": "m1234",
        "qos": "regular_0",
        "constraint": "cpu",
        "walltime": "06:00:00",
        "nodes": 1
      },
      "qe_module": "espresso/7.5-libxc-7.0.0-cpu",
      "ssh_key": "~/.ssh/cluster_key"
    }

Legacy top-level keys ``heavy_root``, ``tmp_dir``, ``repo_root``, ``python`` are
also accepted for backward compatibility.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

JOB_SUBDIRS: Dict[str, str] = {
    "chinook": "chinook_gui_run",
    "sprkkr": "sprkkr_gui_run",
    "qe": "qe_gui_run",
    "tb": "tb_gui_run",
}


def _user(cluster: Mapping[str, Any]) -> str:
    return str(cluster.get("user") or "user")


def _paths_block(cluster: Mapping[str, Any]) -> Dict[str, Any]:
    block = cluster.get("paths")
    return block if isinstance(block, dict) else {}


def _lookup(cluster: Mapping[str, Any], paths_key: str, *legacy_keys: str) -> Optional[str]:
    block = _paths_block(cluster)
    if paths_key in block and block[paths_key]:
        return str(block[paths_key]).rstrip("/")
    for key in legacy_keys:
        if key in cluster and cluster[key]:
            return str(cluster[key]).rstrip("/")
    return None


def heavy_root(cluster: Mapping[str, Any]) -> str:
    """Scratch root for all TensorSpec remote job directories."""
    return _lookup(cluster, "heavy_root", "heavy_root") or (
        f"/mnt/data/{_user(cluster)}/tensorspec_heavy"
    )


def tmp_dir(cluster: Mapping[str, Any]) -> str:
    """Remote TMPDIR for large temp files."""
    return _lookup(cluster, "tmp_dir", "tmp_dir") or f"/mnt/data/{_user(cluster)}/tmp"


def repo_root(cluster: Mapping[str, Any]) -> str:
    """Checkout of TensorSpec on the remote host (for venv / optional clone)."""
    return _lookup(cluster, "repo_root", "repo_root") or f"/home/{_user(cluster)}/TensorSpec"


def python_bin(cluster: Mapping[str, Any]) -> str:
    """Python interpreter used to launch remote runners."""
    explicit = _lookup(cluster, "python", "python")
    if explicit:
        return explicit
    return f"{repo_root(cluster)}/TensorSpec_env/bin/python"


def sprkkr_bin_dir(cluster: Mapping[str, Any]) -> str:
    """Directory containing kkrspec / kkrscf binaries."""
    return _lookup(cluster, "sprkkr_bin", "sprkkr_bin") or f"{heavy_root(cluster)}/SPRKKR/bin"


def job_dir(cluster: Mapping[str, Any], job: str) -> str:
    """Absolute run directory for a job family (chinook, sprkkr, qe, tb)."""
    sub = JOB_SUBDIRS.get(job, job)
    return f"{heavy_root(cluster)}/{sub}"


def sprkkr_binary(cluster: Mapping[str, Any], name: str) -> str:
    return f"{sprkkr_bin_dir(cluster)}/{name}"


def vault_dir(cluster: Mapping[str, Any], vault_name: str) -> str:
    return f"{heavy_root(cluster)}/vaults/{vault_name}"


def mkdir_p_bash(cluster: Mapping[str, Any], *extra_dirs: str) -> str:
    dirs = [
        heavy_root(cluster),
        tmp_dir(cluster),
        repo_root(cluster),
    ]
    for sub in JOB_SUBDIRS.values():
        dirs.append(f"{heavy_root(cluster)}/{sub}")
    dirs.extend(extra_dirs)
    # preserve order, drop dupes
    seen = set()
    lines = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            lines.append(f'mkdir -p "{d}"')
    return "\n".join(lines)


def shell_export_tmp(cluster: Mapping[str, Any], *, one_line: bool = False) -> str:
    t = tmp_dir(cluster)
    if one_line:
        return f'export TMPDIR="{t}" && mkdir -p "$TMPDIR"'
    return f'export TMPDIR="{t}"\nmkdir -p "$TMPDIR"'


def shell_thread_limits(*, one_line: bool = False) -> str:
    parts = (
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
    )
    return " && ".join(parts) if one_line else "\n".join(parts)


def qe_env_exports(cluster: Mapping[str, Any], *, one_line: bool = False) -> str:
    """Optional QE / SPR-KKR conda prefix (paths.qe_env)."""
    prefix = _lookup(cluster, "qe_env", "qe_env")
    user = _user(cluster)
    if not prefix:
        prefix = f"/home/{user}/miniconda3/envs/qe"
    lines = (
        f'export PATH="{prefix}/bin:$PATH"',
        f'export LD_LIBRARY_PATH="{prefix}/lib:$LD_LIBRARY_PATH"',
    )
    return " && ".join(lines) if one_line else "\n".join(lines)


def _slurm_block(cluster: Mapping[str, Any]) -> Dict[str, Any]:
    block = cluster.get("slurm")
    if isinstance(block, dict):
        return block
    paths = _paths_block(cluster)
    nested = paths.get("slurm")
    return nested if isinstance(nested, dict) else {}


def is_slurm(cluster: Optional[Mapping[str, Any]]) -> bool:
    if not cluster:
        return False
    return str(cluster.get("mode", "")).upper() == "SLURM"


def slurm_account(cluster: Mapping[str, Any]) -> str:
    return str(_slurm_block(cluster).get("account") or _lookup(cluster, "slurm_account", "slurm_account") or "")


def slurm_qos(cluster: Mapping[str, Any]) -> str:
    return str(
        _slurm_block(cluster).get("qos")
        or _lookup(cluster, "slurm_qos", "slurm_qos")
        or "regular"
    )


def slurm_constraint(cluster: Mapping[str, Any]) -> str:
    return str(
        _slurm_block(cluster).get("constraint")
        or _lookup(cluster, "slurm_constraint", "slurm_constraint")
        or "cpu"
    )


def slurm_walltime(cluster: Mapping[str, Any]) -> str:
    return str(
        _slurm_block(cluster).get("walltime")
        or _lookup(cluster, "slurm_walltime", "slurm_walltime")
        or "02:00:00"
    )


def slurm_nodes(cluster: Mapping[str, Any]) -> int:
    raw = _slurm_block(cluster).get("nodes")
    if raw is not None:
        return max(1, int(raw))
    legacy = _lookup(cluster, "slurm_nodes", "slurm_nodes")
    if legacy:
        return max(1, int(legacy))
    return 1


def qe_module(cluster: Mapping[str, Any]) -> str:
    return str(_lookup(cluster, "qe_module", "qe_module") or "")


def qe_module_load(cluster: Mapping[str, Any], *, one_line: bool = False) -> str:
    mod = qe_module(cluster).strip()
    if not mod:
        return ""
    line = f"module load {mod}"
    return line if one_line else f"{line}\n"


def ssh_key_path(cluster: Mapping[str, Any]) -> Optional[str]:
    key = _lookup(cluster, "ssh_key", "ssh_key")
    if not key:
        return None
    return os.path.expanduser(key)


def mpi_launch_prefix(
    cluster: Optional[Mapping[str, Any]],
    ranks: int,
    *,
    use_mpi: bool = True,
) -> str:
    if not use_mpi or ranks < 1:
        return ""
    if is_slurm(cluster):
        return f"srun -n {ranks} --cpu-bind=cores "
    return f"mpirun --use-hwthread-cpus --oversubscribe -np {ranks} "


def slurm_sbatch_header(
    cluster: Mapping[str, Any],
    *,
    mpi_ranks: int,
    job_name: str = "ts_qe",
) -> str:
    account = slurm_account(cluster)
    if not account:
        raise ValueError("SLURM cluster missing slurm.account (or paths.slurm.account)")
    lines = [
        "#!/bin/bash",
        f"#SBATCH -A {account}",
        f"#SBATCH -C {slurm_constraint(cluster)}",
        f"#SBATCH -q {slurm_qos(cluster)}",
        f"#SBATCH -N {slurm_nodes(cluster)}",
        f"#SBATCH -t {slurm_walltime(cluster)}",
        f"#SBATCH -J {job_name}",
        "#SBATCH -o sys.out.full",
        "#SBATCH -e sys.out.full",
        f"#SBATCH --ntasks={max(1, mpi_ranks)}",
        "#SBATCH --cpus-per-task=1",
    ]
    return "\n".join(lines)


def build_qe_slurm_batch(
    cluster: Mapping[str, Any],
    *,
    remote_dir: str,
    script_name: str,
    mpi_ranks: int,
) -> str:
    body = [
        slurm_sbatch_header(cluster, mpi_ranks=mpi_ranks),
        "",
        f"cd {remote_dir}",
        shell_export_tmp(cluster),
        shell_thread_limits(),
        qe_module_load(cluster).rstrip(),
        qe_env_exports(cluster),
        "",
        f"bash {script_name}",
    ]
    return "\n".join(part for part in body if part) + "\n"


def ssh_connect(client, cluster: Mapping[str, Any], *, timeout: int = 10) -> None:
    """Paramiko connect using password and/or paths.ssh_key; agent keys if no password."""
    import paramiko

    if not isinstance(client, paramiko.SSHClient):
        raise TypeError("client must be paramiko.SSHClient")
    pwd = cluster.get("password") or None
    key = ssh_key_path(cluster)
    kwargs: Dict[str, Any] = {
        "hostname": cluster["host"],
        "port": int(cluster.get("port", 22)),
        "username": cluster["user"],
        "timeout": timeout,
        "allow_agent": True,
        "look_for_keys": True,
    }
    if pwd:
        kwargs["password"] = pwd
    if key:
        kwargs["key_filename"] = key
        kwargs["allow_agent"] = False
        kwargs["look_for_keys"] = False
    client.connect(**kwargs)


def provision_script(cluster: Mapping[str, Any]) -> str:
    """Bash run by Compute Manager → Auto-Setup Remote Environment."""
    py = python_bin(cluster)
    repo = repo_root(cluster)
    return f"""{mkdir_p_bash(cluster)}

if [ ! -f "{py}" ]; then
    python3 -m venv "{repo}/TensorSpec_env" || virtualenv "{repo}/TensorSpec_env"
fi

"{py}" -m pip install --quiet numpy scipy matplotlib
echo "SETUP_COMPLETE"
"""
