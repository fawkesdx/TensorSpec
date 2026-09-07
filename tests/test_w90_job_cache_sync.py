"""Hybrid W90 cache export/sync for ARPES H_dict attach."""

from __future__ import annotations

import pickle
from pathlib import Path

from tensorspec.core.dft.w90_tb_cache import ensure_cache_key, write_job_dir_cache


def test_write_job_dir_cache_uses_client_key(tmp_path: Path):
    out = tmp_path / "w90_tb_cache.pkl"
    tb = {"type": "list", "list": [[0, 0, 1.0, 0.0, 0.0, 1.0 + 0j]], "H": []}
    tb["H"] = tb["list"]
    write_job_dir_cache(
        out,
        key="abc123clientkey",
        tb_dict=tb,
        basis_args={"atoms": ["X"], "pos": [[0, 0, 0]]},
        source="/remote/wannier90_hr.dat",
    )
    with out.open("rb") as f:
        payload = pickle.load(f)
    assert payload["key"] == "abc123clientkey"
    assert len(payload["tb_dict"]["list"]) == 1


def test_ensure_cache_key_rewrites_mismatched_key(tmp_path: Path):
    out = tmp_path / "w90_tb_cache.pkl"
    tb = {"type": "list", "list": [[1, 1, 0.0, 0.0, 0.0, 0.5 + 0j]], "H": []}
    tb["H"] = tb["list"]
    write_job_dir_cache(
        out,
        key="wrong_remote_path_key",
        tb_dict=tb,
        basis_args={"atoms": ["X"]},
    )
    assert ensure_cache_key(out, "local_client_key") is True
    with out.open("rb") as f:
        payload = pickle.load(f)
    assert payload["key"] == "local_client_key"


def test_ensure_cache_key_rejects_empty_hops(tmp_path: Path):
    out = tmp_path / "empty.pkl"
    write_job_dir_cache(
        out,
        key="k",
        tb_dict={"type": "list", "list": [], "H": []},
        basis_args={},
    )
    assert ensure_cache_key(out, "k") is False
