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
      "ssh_key": "~/.ssh/cluster_key",
      "trust_new_host_key": false
    }

``trust_new_host_key`` (default false): accept and save an SSH host key that is
not yet in ~/.ssh/known_hosts without asking (headless use only). See the
"SSH host-key validation" section below.

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


def uses_sshproxy(cluster: Optional[Mapping[str, Any]]) -> bool:
    """True for clusters that log in through an ``sshproxy`` (MFA) short-lived key.

    Opt-in only: set ``"auth": "sshproxy"`` in the cluster JSON (and usually
    ``"ssh_key"`` for the key file sshproxy writes). No host-name detection.
    """
    if not cluster:
        return False
    auth = str(
        cluster.get("auth")
        or _lookup(cluster, "auth", "auth")
        or ""
    ).lower()
    if auth == "sshproxy":
        return True
    return False


# Flat files useful for local ARPES / Chinook / Grizzly after a QE+Wannier run.
# Skips wavefunctions and large Wannier intermediates (.mmn/.amn/.eig/.chk).
_ARPES_FETCH_EXACT = frozenset({
    "sys.out.full",
    "fermi_energy.txt",
    "relax.out",
    "relaxed_structure.cif",
    "structure_template.cif",
    "tensorspec_relax_meta.json",
})
_ARPES_FETCH_SUFFIXES = (
    "_hr.dat",
    ".out",
    ".wout",
    ".win",
    ".in",
    ".xyz",
    ".gnu",
    ".kpt",
)
_ARPES_FETCH_SKIP_SUFFIXES = (".mmn", ".amn", ".eig", ".chk", ".wfc", ".save")


def is_arpes_fetch_candidate(filename: str) -> bool:
    """Return True if ``filename`` should be downloaded for ARPES follow-up work."""
    name = os.path.basename(filename)
    lower = name.lower()
    if any(lower.endswith(s) for s in _ARPES_FETCH_SKIP_SUFFIXES):
        return False
    if lower in _ARPES_FETCH_EXACT:
        return True
    return any(lower.endswith(s) for s in _ARPES_FETCH_SUFFIXES)


def sshproxy_command(cluster: Mapping[str, Any]) -> list[str]:
    """Build ``sshproxy`` argv for this cluster's configured key path."""
    import shutil

    if not uses_sshproxy(cluster):
        raise ValueError("Cluster does not use sshproxy auth (set auth: sshproxy)")
    exe = shutil.which("sshproxy")
    if not exe:
        raise FileNotFoundError(
            "sshproxy not found in PATH. Install your site's sshproxy client, then retry."
        )
    user = str(cluster.get("user") or "").strip()
    if not user:
        raise ValueError("Cluster missing user for sshproxy")
    key = ssh_key_path(cluster) or os.path.expanduser("~/.ssh/sshproxy")
    key_dir = os.path.dirname(key) or os.path.expanduser("~/.ssh")
    key_name = os.path.basename(key) or "sshproxy"
    return [exe, "-u", user, "-o", key_name, "-k", key_dir]


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


def load_private_key(path: str):
    """Load PEM/OpenSSH private key for Paramiko (sshproxy keys are typically RSA PEM)."""
    import paramiko

    key_path = os.path.expanduser(path)
    if not os.path.isfile(key_path):
        raise FileNotFoundError(f"SSH key not found: {key_path}")

    loaders = (
        paramiko.RSAKey.from_private_key_file,
        paramiko.Ed25519Key.from_private_key_file,
        paramiko.ECDSAKey.from_private_key_file,
    )
    errors: list[str] = []
    for loader in loaders:
        try:
            return loader(key_path)
        except Exception as exc:
            errors.append(str(exc))
    raise paramiko.ssh_exception.SSHException(
        f"Could not load SSH private key {key_path}: {'; '.join(errors)}"
    )


def local_mpi_prefix(ranks: int, *, windows: bool) -> str:
    """Non-SLURM launcher. Windows local is ``mpiexec -n``; otherwise ``mpirun -np``.

    No hwthread/oversubscribe — those break QE FFT groups on conda / Homebrew Open MPI.
    """
    if windows:
        return f"mpiexec -n {ranks} "
    return f"mpirun -np {ranks} "


def _windows_local(cluster: Optional[Mapping[str, Any]]) -> bool:
    """True only when this GUI machine runs the job. A cluster dict is a remote host."""
    return not cluster and os.name == "nt"


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
    return local_mpi_prefix(ranks, windows=_windows_local(cluster))


def adapt_pipeline_mpi_launcher(
    script: str,
    cluster: Optional[Mapping[str, Any]],
) -> str:
    """Rewrite MPI launch prefixes to match the selected compute target.

    SLURM (HPC) → ``srun -n N --cpu-bind=cores``
    Windows local → ``mpiexec -n N``
    macOS / Linux local → ``mpirun -np N`` (no hwthread/oversubscribe)

    Rank counts are preserved. Safe to call at Generate and again at Run so
    switching Compute Target after Generate still uploads the right launcher.
    """
    import re

    if not script:
        return script
    want_srun = is_slurm(cluster)
    pattern = (
        r"(?:"
        r"srun\s+-n\s+(\d+)(?:\s+--cpu-bind=cores)?\s+"
        r"|"
        r"mpirun(?:\s+--use-hwthread-cpus)?(?:\s+--oversubscribe)?\s+-np\s+(\d+)\s+"
        r"|"
        r"mpiexec\s+-n\s+(\d+)\s+"
        r")"
    )

    def _replace(match: re.Match) -> str:
        ranks = match.group(1) or match.group(2) or match.group(3)
        if want_srun:
            return f"srun -n {ranks} --cpu-bind=cores "
        return local_mpi_prefix(int(ranks), windows=_windows_local(cluster))

    return re.sub(pattern, _replace, script)


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
        kwargs["pkey"] = load_private_key(key)
        kwargs["allow_agent"] = False
        kwargs["look_for_keys"] = False
    client.connect(**kwargs)


# ---------------------------------------------------------------------------
# SSH host-key validation
#
# Every paramiko client in TensorSpec must come from ``new_ssh_client`` so the
# server's host key is checked against ~/.ssh/known_hosts (OpenSSH format).
# A host that is not there raises ``UnknownHostKeyError`` instead of being
# silently trusted (CodeQL py/paramiko-missing-host-key-validation).  GUI code
# catches that error once, shows host + SHA256 fingerprint and, on Yes, calls
# ``trust_host_key`` and retries (tensorspec.gui.services.host_key_prompt).
# Headless use may opt in per cluster with ``"trust_new_host_key": true``.
# ---------------------------------------------------------------------------


class UnknownHostKeyError(Exception):
    """The server presented a host key that is not in known_hosts.

    ``hostname`` is paramiko's known_hosts name (``host`` or ``[host]:port``),
    ``key`` the paramiko ``PKey``, ``fingerprint`` its ``SHA256:...`` digest.
    """

    def __init__(self, hostname: str, key, *, known_hosts: Optional[str] = None):
        self.hostname = hostname
        self.key = key
        self.key_type = key.get_name()
        self.fingerprint = host_key_fingerprint(key)
        self.known_hosts = known_hosts or known_hosts_path()
        super().__init__(
            f"Unknown SSH host key for {hostname}: {self.key_type} {self.fingerprint}. "
            f"The host is not in {self.known_hosts}. Verify the fingerprint with the "
            "cluster documentation, then accept it via Compute Manager > Test "
            'Connection, or set "trust_new_host_key": true in the cluster entry '
            "for headless use."
        )


def host_key_fingerprint(key) -> str:
    """OpenSSH-style ``SHA256:<base64, no padding>`` fingerprint of a paramiko key."""
    import base64
    import hashlib

    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def known_hosts_path() -> str:
    return os.path.expanduser("~/.ssh/known_hosts")


SYSTEM_KNOWN_HOSTS = "/etc/ssh/ssh_known_hosts"


def host_key_name(cluster: Mapping[str, Any]) -> str:
    """known_hosts name paramiko uses for this cluster (``[host]:port`` off port 22)."""
    host = str(cluster["host"])
    port = int(cluster.get("port", 22))
    return host if port == 22 else f"[{host}]:{port}"


def trusts_new_host_key(cluster: Optional[Mapping[str, Any]]) -> bool:
    """Opt-in: cluster JSON ``"trust_new_host_key": true`` accepts and saves unknown keys."""
    return bool(cluster and cluster.get("trust_new_host_key"))


def load_known_hosts(path: Optional[str] = None):
    """``paramiko.HostKeys`` loaded from ``path`` (default ~/.ssh/known_hosts); empty if absent.

    Tolerant replacement for ``HostKeys.load`` / ``SSHClient.load_system_host_keys``,
    which abort the whole file on one line they cannot decode. ``@cert-authority``
    and ``@revoked`` markers (common at HPC centres) and malformed lines are
    skipped; every ordinary entry, hashed or plain, is kept.
    """
    import paramiko
    from paramiko.hostkeys import HostKeyEntry

    keys = paramiko.HostKeys()
    path = path or known_hosts_path()
    if not os.path.isfile(path):
        return keys
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line[0] in "#@":
                continue
            try:
                entry = HostKeyEntry.from_line(line, lineno)
            except Exception:
                continue
            if entry is None:
                continue
            for name in entry.hostnames:
                keys.add(name, entry.key.get_name(), entry.key)
    return keys


def lookup_host_key(cluster: Mapping[str, Any], *, path: Optional[str] = None):
    """Known host keys for ``cluster`` (``{key_type: key}``) or None if not in known_hosts."""
    return load_known_hosts(path).lookup(host_key_name(cluster))


def trust_host_key(hostname: str, key, *, path: Optional[str] = None, client=None) -> None:
    """Append ``hostname key`` to known_hosts and (optionally) to ``client``'s in-memory keys.

    Appends one OpenSSH-format line rather than rewriting the file with
    ``HostKeys.save`` so that entries paramiko cannot parse (``@cert-authority``,
    comments, unknown key types) are preserved.
    """
    from paramiko.hostkeys import HostKeyEntry

    path = path or known_hosts_path()
    line = HostKeyEntry([hostname], key).to_line()
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, mode=0o700, exist_ok=True)
    existed = os.path.isfile(path)
    needs_newline = False
    if existed and os.path.getsize(path) > 0:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            needs_newline = fh.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as fh:
        if needs_newline:
            fh.write("\n")
        fh.write(line)
    if not existed:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    if client is not None:
        client.get_host_keys().add(hostname, key.get_name(), key)


class RejectUnknownHostKeyPolicy:
    """paramiko missing-host-key policy: raise ``UnknownHostKeyError``.

    With ``trust_new_host_key=True`` (cluster opt-in) the key is appended to
    known_hosts and accepted instead -- trust on first use, verified afterwards.
    Duck-typed like ``paramiko.MissingHostKeyPolicy`` so paramiko is not
    imported at module load.
    """

    def __init__(self, *, trust_new_host_key: bool = False, known_hosts: Optional[str] = None):
        self.trust_new_host_key = bool(trust_new_host_key)
        self.known_hosts = known_hosts or known_hosts_path()

    def missing_host_key(self, client, hostname: str, key) -> None:
        if not self.trust_new_host_key:
            raise UnknownHostKeyError(hostname, key, known_hosts=self.known_hosts)
        trust_host_key(hostname, key, path=self.known_hosts, client=client)


def new_ssh_client(
    cluster: Optional[Mapping[str, Any]] = None,
    *,
    known_hosts: Optional[str] = None,
):
    """``paramiko.SSHClient`` with known_hosts loaded and unknown hosts rejected.

    Replaces bare ``paramiko.SSHClient()`` + auto-accept policies everywhere. Pass the
    cluster dict so its ``trust_new_host_key`` opt-in is honoured. ``known_hosts``
    overrides the file (tests); otherwise /etc/ssh/ssh_known_hosts and
    ~/.ssh/known_hosts are loaded (tolerantly, see ``load_known_hosts``).
    """
    import paramiko

    client = paramiko.SSHClient()
    path = known_hosts or known_hosts_path()
    sources = [path] if known_hosts else [SYSTEM_KNOWN_HOSTS, path]
    target = client.get_host_keys()
    for source in sources:
        loaded = load_known_hosts(source)
        for hostname in loaded.keys():
            for key_type, key in loaded[hostname].items():
                target.add(hostname, key_type, key)
    client.set_missing_host_key_policy(
        RejectUnknownHostKeyPolicy(
            trust_new_host_key=trusts_new_host_key(cluster), known_hosts=path
        )
    )
    return client


def fetch_host_key(cluster: Mapping[str, Any], *, timeout: float = 10):
    """Server host key from a key exchange only (no authentication).

    Used by the GUI to show host + fingerprint before a worker thread connects.
    """
    import socket

    import paramiko

    sock = socket.create_connection(
        (str(cluster["host"]), int(cluster.get("port", 22))), timeout=timeout
    )
    transport = paramiko.Transport(sock)
    try:
        transport.start_client(timeout=timeout)
        return transport.get_remote_server_key()
    finally:
        transport.close()


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
