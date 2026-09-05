"""Disk cache for parsed wannier90_hr.dat → Chinook tb_dict (band diag hot path)."""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional, Tuple

CACHE_DIR = Path(os.path.expanduser("~/.tensorspec_cache/w90_tb"))
CACHE_VERSION = 6  # v6: basis_source + ARPES quality metadata in basis_args
REMOTE_CACHE_NAME = "w90_tb_cache.pkl"


def cache_key(
    w90_filepath: str,
    use_soc: bool,
    onsite_e: float,
    hop_tol: float = 1e-6,
) -> str:
    path = os.path.abspath(w90_filepath)
    try:
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
    except OSError:
        mtime, size = 0, 0
    raw = (
        f"v{CACHE_VERSION}|{path}|{mtime}|{size}|{use_soc}|"
        f"{float(onsite_e):.12g}|{float(hop_tol):.12g}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def local_cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.pkl"


def load_parsed_tb(
    w90_filepath: str,
    use_soc: bool,
    onsite_e: float,
    hop_tol: float = 1e-6,
) -> Optional[Tuple[dict, dict, Optional[Any]]]:
    """Return (tb_dict, basis_args, A_qe) or None."""
    key = cache_key(w90_filepath, use_soc, onsite_e, hop_tol)
    path = local_cache_path(key)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("key") != key:
            return None
        return payload["tb_dict"], payload["basis_args"], payload.get("A_qe")
    except Exception:
        return None


def save_parsed_tb(
    w90_filepath: str,
    use_soc: bool,
    onsite_e: float,
    tb_dict: dict,
    basis_args: dict,
    A_qe,
    hop_tol: float = 1e-6,
) -> Path:
    key = cache_key(w90_filepath, use_soc, onsite_e, hop_tol)
    path = local_cache_path(key)
    payload = {
        "key": key,
        "tb_dict": tb_dict,
        "basis_args": basis_args,
        "A_qe": A_qe,
        "hop_tol": float(hop_tol),
        "saved_at": time.time(),
        "source": os.path.abspath(w90_filepath),
    }
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return path


def remote_cache_payload_path() -> str:
    return REMOTE_CACHE_NAME
