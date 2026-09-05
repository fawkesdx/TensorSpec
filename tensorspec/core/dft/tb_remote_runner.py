#!/usr/bin/env python3
"""Remote tight-binding band diagonalization (runs on cluster, writes tb_bands_result.npz)."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np


def _solve_bands(engine, k_vecs, job: dict):
    """Call solve_bands; fail clearly if cluster TensorSpec lacks Grizzly API."""
    kwargs = {
        "custom_hopping": job.get("custom_hopping") or {},
        "onsite_e": float(job.get("onsite_e", 0.0)),
        "use_soc": bool(job.get("use_soc", False)),
        "soc_strength": float(job.get("soc_strength", 0.5)),
        "w90_filepath": job.get("_w90_path"),
        "cutoffs": job.get("cutoffs"),
        "tb_mode": job.get("tb_mode", "Simple Scalar"),
        "orbital_shifts": job.get("orbital_shifts"),
        "need_eigenvectors": bool(job.get("need_eigenvectors", True)),
        "diag_engine": str(job.get("diag_engine", "chinook")),
        "diag_device": str(job.get("diag_device", "cpu")),
    }
    params = inspect.signature(engine.solve_bands).parameters
    missing = [
        k
        for k in ("need_eigenvectors", "diag_engine", "diag_device")
        if k in kwargs and k not in params
    ]
    want_grizzly = kwargs.get("diag_engine") == "grizzly"
    if want_grizzly and missing:
        raise RuntimeError(
            "Remote TensorSpec is outdated (no Grizzly band diag). "
            "On cluster: cd ~/TensorSpec && git fetch && git checkout TensorSpec_GUI "
            "&& git pull origin TensorSpec_GUI"
        )
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return engine.solve_bands(k_vecs, **filtered)


def main() -> int:
    parser = argparse.ArgumentParser(description="TensorSpec remote TB band runner")
    parser.add_argument("--job", default="tb_job.json", help="Job specification JSON")
    parser.add_argument(
        "--out",
        default="tb_bands_result.npz",
        help="Output NPZ (eigenvalues, optional eigenvectors, orb_labels)",
    )
    args = parser.parse_args()

    job_path = Path(args.job)
    if not job_path.is_file():
        print(f"FATAL: missing job file {job_path}", flush=True)
        return 1

    job = json.loads(job_path.read_text())
    run_dir = job_path.parent.resolve()

    from pymatgen.core import Structure

    from tensorspec.core.dft.chinook_tb import ChinookTightBindingEngine

    engine = ChinookTightBindingEngine()
    engine.crystal_structure = Structure.from_dict(job["structure"])

    k_vecs = np.asarray(job["k_vecs"], dtype=float)
    w90_name = job.get("w90_basename")
    w90_path = str(run_dir / w90_name) if w90_name else None
    job["_w90_path"] = w90_path

    use_soc = bool(job.get("use_soc", False))
    onsite_e = float(job.get("onsite_e", 0.0))
    cache_name = job.get("w90_cache_basename")
    if w90_path and cache_name:
        cache_path = run_dir / cache_name
        if cache_path.is_file():
            import pickle

            try:
                with cache_path.open("rb") as f:
                    payload = pickle.load(f)
                source_key = engine._w90_source_key(w90_path, use_soc, onsite_e)
                engine.seed_w90_parsed(
                    source_key,
                    payload["tb_dict"],
                    payload["basis_args"],
                    payload.get("A_qe"),
                )
                n_hop = len(payload.get("tb_dict", {}).get("list", []))
                print(
                    f"W90 TB cache loaded on server ({n_hop} hops, skip hr.dat parse)",
                    flush=True,
                )
            except Exception as exc:
                print(f"WARN: W90 cache load failed, will parse hr.dat: {exc}", flush=True)

    t0 = time.perf_counter()
    eigenvalues, eigenvectors, orb_labels = _solve_bands(engine, k_vecs, job)
    fermi_energy = float(job.get("fermi_energy", 0.0))
    eigenvalues = np.asarray(eigenvalues, dtype=float) - fermi_energy

    out_path = Path(args.out)
    if eigenvectors is None:
        np.savez(
            out_path,
            eigenvalues=eigenvalues,
            orb_labels=np.array(orb_labels, dtype=object),
            fermi_energy=fermi_energy,
            backend=str(job.get("diag_engine", "chinook")),
        )
    else:
        np.savez(
            out_path,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            orb_labels=np.array(orb_labels, dtype=object),
            fermi_energy=fermi_energy,
            backend=str(job.get("diag_engine", "chinook")),
        )

    elapsed = time.perf_counter() - t0
    print(
        f"OK: {eigenvalues.shape[0]} k-pts x {eigenvalues.shape[1]} bands "
        f"in {elapsed:.2f}s -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
