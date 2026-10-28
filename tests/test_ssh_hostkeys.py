"""SSH host-key validation (cluster_paths.new_ssh_client and friends).

A tiny in-process paramiko server on 127.0.0.1 presents a freshly generated
host key, so the real ``SSHClient.connect`` host-key check runs:

* host in known_hosts            -> connection proceeds (to auth)
* host not in known_hosts        -> UnknownHostKeyError, nothing written
* cluster["trust_new_host_key"]  -> accepted and appended to known_hosts

Placeholder hosts only (cluster.example.org); no real network.
"""

from __future__ import annotations

import os
import socket
import threading
from types import SimpleNamespace

import paramiko
import pytest
from paramiko.hostkeys import HostKeyEntry

from tensorspec.core.compute import cluster_paths as cp


# --------------------------------------------------------------------------- server


class _AcceptAll(paramiko.ServerInterface):
    def check_auth_password(self, username, password):
        return paramiko.AUTH_SUCCESSFUL

    def get_allowed_auths(self, username):
        return "password"

    def check_channel_request(self, kind, chanid):
        return paramiko.OPEN_SUCCEEDED


@pytest.fixture
def ssh_server():
    host_key = paramiko.ECDSAKey.generate()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(5)
    sock.settimeout(0.2)
    port = sock.getsockname()[1]
    stop = threading.Event()
    transports = []

    def serve():
        while not stop.is_set():
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = paramiko.Transport(conn)
            t.add_server_key(host_key)
            try:
                t.start_server(event=threading.Event(), server=_AcceptAll())
            except Exception:
                pass
            transports.append(t)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield SimpleNamespace(host="127.0.0.1", port=port, key=host_key)
    stop.set()
    thread.join(2)
    for t in transports:
        t.close()
    sock.close()


def _cluster(server, **extra):
    c = {"name": "test", "host": server.host, "port": server.port, "user": "alice"}
    c.update(extra)
    return c


def _connect(client, server):
    client.connect(
        server.host,
        port=server.port,
        username="alice",
        password="not-a-real-password",
        allow_agent=False,
        look_for_keys=False,
        timeout=5,
        banner_timeout=5,
        auth_timeout=5,
    )


# --------------------------------------------------------------------------- helpers


def test_fingerprint_matches_paramiko_and_openssh_format():
    key = paramiko.ECDSAKey.generate()
    fp = cp.host_key_fingerprint(key)
    assert fp.startswith("SHA256:")
    assert "=" not in fp
    assert fp == key.fingerprint


def test_host_key_name_uses_bracket_port_off_22():
    assert cp.host_key_name({"host": "cluster.example.org"}) == "cluster.example.org"
    assert cp.host_key_name({"host": "cluster.example.org", "port": 22}) == "cluster.example.org"
    assert cp.host_key_name({"host": "cluster.example.org", "port": 2222}) == "[cluster.example.org]:2222"


def test_trust_host_key_appends_and_preserves_unparseable_lines(tmp_path):
    kh = tmp_path / "known_hosts"
    kh.write_text("# comment\n@cert-authority *.example.org ssh-ed25519 AAAA bogus\nold.example.org ssh-rsa AAAA")  # no trailing newline
    key = paramiko.ECDSAKey.generate()

    cp.trust_host_key("cluster.example.org", key, path=str(kh))

    lines = kh.read_text().splitlines()
    assert lines[0] == "# comment"
    assert lines[1].startswith("@cert-authority")
    assert lines[2] == "old.example.org ssh-rsa AAAA"
    assert lines[3] == HostKeyEntry(["cluster.example.org"], key).to_line().rstrip("\n")
    assert cp.lookup_host_key({"host": "cluster.example.org"}, path=str(kh))[key.get_name()] == key


def test_trust_host_key_creates_private_file(tmp_path):
    kh = tmp_path / ".ssh" / "known_hosts"
    key = paramiko.ECDSAKey.generate()
    cp.trust_host_key("[cluster.example.org]:2222", key, path=str(kh))
    assert kh.is_file()
    assert oct(kh.stat().st_mode & 0o777) == "0o600"
    assert cp.lookup_host_key({"host": "cluster.example.org", "port": 2222}, path=str(kh))


def test_new_ssh_client_loads_known_hosts_and_rejects_by_default(tmp_path):
    kh = tmp_path / "known_hosts"
    key = paramiko.ECDSAKey.generate()
    kh.write_text(HostKeyEntry(["cluster.example.org"], key).to_line())

    client = cp.new_ssh_client({"host": "cluster.example.org"}, known_hosts=str(kh))
    assert client.get_host_keys().lookup("cluster.example.org")[key.get_name()] == key
    policy = client._policy
    assert isinstance(policy, cp.RejectUnknownHostKeyPolicy)
    assert policy.trust_new_host_key is False

    other = paramiko.ECDSAKey.generate()
    with pytest.raises(cp.UnknownHostKeyError) as info:
        policy.missing_host_key(client, "other.example.org", other)
    err = info.value
    assert err.hostname == "other.example.org"
    assert err.key is other
    assert err.fingerprint == cp.host_key_fingerprint(other)
    assert "other.example.org" in str(err) and err.fingerprint in str(err)
    assert str(kh) in str(err)
    # nothing written
    assert kh.read_text() == HostKeyEntry(["cluster.example.org"], key).to_line()


def test_new_ssh_client_without_file_is_empty_and_rejecting(tmp_path):
    client = cp.new_ssh_client(None, known_hosts=str(tmp_path / "missing"))
    assert len(client.get_host_keys()) == 0
    with pytest.raises(cp.UnknownHostKeyError):
        client._policy.missing_host_key(client, "cluster.example.org", paramiko.ECDSAKey.generate())


# --------------------------------------------------------------------------- real connect


def test_known_host_is_accepted(ssh_server, tmp_path):
    kh = tmp_path / "known_hosts"
    kh.write_text(HostKeyEntry([cp.host_key_name(_cluster(ssh_server))], ssh_server.key).to_line())
    client = cp.new_ssh_client(_cluster(ssh_server), known_hosts=str(kh))
    try:
        _connect(client, ssh_server)  # would raise UnknownHostKeyError if the key were not trusted
        assert client.get_transport().is_authenticated()
    finally:
        client.close()


def test_unknown_host_is_rejected(ssh_server, tmp_path):
    kh = tmp_path / "known_hosts"
    kh.write_text("")
    client = cp.new_ssh_client(_cluster(ssh_server), known_hosts=str(kh))
    try:
        with pytest.raises(cp.UnknownHostKeyError) as info:
            _connect(client, ssh_server)
    finally:
        client.close()
    assert info.value.hostname == cp.host_key_name(_cluster(ssh_server))
    assert info.value.fingerprint == cp.host_key_fingerprint(ssh_server.key)
    assert kh.read_text() == ""


def test_changed_host_key_is_rejected_by_paramiko(ssh_server, tmp_path):
    kh = tmp_path / "known_hosts"
    impostor = paramiko.ECDSAKey.generate()
    kh.write_text(HostKeyEntry([cp.host_key_name(_cluster(ssh_server))], impostor).to_line())
    client = cp.new_ssh_client(_cluster(ssh_server), known_hosts=str(kh))
    try:
        with pytest.raises(paramiko.BadHostKeyException):
            _connect(client, ssh_server)
    finally:
        client.close()


def test_opt_in_trust_new_host_key_accepts_and_saves(ssh_server, tmp_path):
    kh = tmp_path / "known_hosts"
    cluster = _cluster(ssh_server, trust_new_host_key=True)
    client = cp.new_ssh_client(cluster, known_hosts=str(kh))
    assert client._policy.trust_new_host_key is True
    try:
        _connect(client, ssh_server)
        assert client.get_transport().is_authenticated()
    finally:
        client.close()
    saved = cp.lookup_host_key(cluster, path=str(kh))
    assert saved and saved[ssh_server.key.get_name()] == ssh_server.key

    # second contact is verified against the saved key, no trust needed
    client2 = cp.new_ssh_client(_cluster(ssh_server), known_hosts=str(kh))
    try:
        _connect(client2, ssh_server)
    finally:
        client2.close()


def test_fetch_host_key_returns_server_key_without_auth(ssh_server):
    key = cp.fetch_host_key(_cluster(ssh_server), timeout=5)
    assert key == ssh_server.key
    assert cp.host_key_fingerprint(key) == cp.host_key_fingerprint(ssh_server.key)


def test_ssh_connect_surfaces_unknown_host(ssh_server, tmp_path):
    """The shared ssh_connect helper (used by most call sites) raises the same error."""
    cluster = _cluster(ssh_server, password="not-a-real-password")
    client = cp.new_ssh_client(cluster, known_hosts=str(tmp_path / "known_hosts"))
    try:
        with pytest.raises(cp.UnknownHostKeyError):
            cp.ssh_connect(client, cluster, timeout=5)
    finally:
        client.close()
