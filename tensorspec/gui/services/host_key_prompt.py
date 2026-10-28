"""First-contact SSH host-key confirmation for the GUI.

``cluster_paths.new_ssh_client`` rejects hosts missing from ~/.ssh/known_hosts
with ``UnknownHostKeyError``. Main-thread GUI code connects through
``connect_cluster`` (catches that error once, asks, trusts, retries). Code that
connects inside a worker ``QThread`` must not open dialogs, so it calls
``ensure_host_key_trusted`` on the main thread *before* starting the thread.
Both paths show the same dialog: host, key type and SHA256 fingerprint.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from PySide6.QtWidgets import QMessageBox, QWidget

from tensorspec.core.compute import cluster_paths as cp


def confirm_host_key(parent: Optional[QWidget], hostname: str, key) -> bool:
    """Ask the user to accept ``key`` for ``hostname``; on Yes append to known_hosts."""
    reply = QMessageBox.question(
        parent,
        "Unknown SSH Host Key",
        f"The authenticity of host '{hostname}' can't be established.\n\n"
        f"{key.get_name()} key fingerprint:\n{cp.host_key_fingerprint(key)}\n\n"
        "Compare it with the fingerprint published by the cluster (or run\n"
        "ssh-keyscan on a trusted machine). Only continue if it matches.\n\n"
        f"Add this key to {cp.known_hosts_path()} and connect?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    if reply != QMessageBox.Yes:
        return False
    cp.trust_host_key(hostname, key)
    return True


def confirm_unknown_host_key(parent: Optional[QWidget], exc: cp.UnknownHostKeyError) -> bool:
    """Dialog for an ``UnknownHostKeyError`` caught from a connect attempt."""
    return confirm_host_key(parent, exc.hostname, exc.key)


def ensure_host_key_trusted(
    parent: Optional[QWidget], cluster: Mapping[str, Any], *, timeout: float = 10
) -> bool:
    """True once ``cluster``'s host key is in known_hosts (or the cluster opts in).

    Known host: no network, no dialog. Unknown host: key exchange only (no
    authentication), then the confirmation dialog. Returns False if the user
    declines or the host cannot be reached (an error box is shown).
    Call on the main thread before starting a worker thread that connects.
    """
    if cp.trusts_new_host_key(cluster) or cp.lookup_host_key(cluster):
        return True
    try:
        key = cp.fetch_host_key(cluster, timeout=timeout)
    except Exception as exc:
        QMessageBox.critical(
            parent,
            "SSH Host Key",
            f"Could not read the SSH host key of {cluster.get('host')}:\n{exc}",
        )
        return False
    return confirm_host_key(parent, cp.host_key_name(cluster), key)


def connect_cluster(
    parent: Optional[QWidget],
    cluster: Mapping[str, Any],
    *,
    timeout: int = 10,
    connect=None,
):
    """Open a paramiko client to ``cluster``, asking once if its host key is unknown.

    ``connect(client)`` overrides ``cluster_paths.ssh_connect`` (e.g. for custom
    banner timeouts). Re-raises ``UnknownHostKeyError`` if the user declines.
    Main thread only (shows a dialog).
    """
    do_connect = connect or (lambda ssh: cp.ssh_connect(ssh, cluster, timeout=timeout))
    ssh = cp.new_ssh_client(cluster)
    try:
        do_connect(ssh)
        return ssh
    except cp.UnknownHostKeyError as exc:
        ssh.close()
        if not confirm_unknown_host_key(parent, exc):
            raise
    ssh = cp.new_ssh_client(cluster)
    do_connect(ssh)
    return ssh
