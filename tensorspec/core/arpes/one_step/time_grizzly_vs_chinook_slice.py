#!/usr/bin/env python3
"""Time one θ-slice: chinook (kmesh) vs GrizzlyME (Phase 0 baseline).

Usage (on the cluster run dir, after TB + arpes_physics.json present)::

    python time_grizzly_vs_chinook_slice.py \\
      --tb_file tb_data.npz \\
      --physics_file arpes_physics.json \\
      --theta 0.0 --phi_min -15 --phi_max 15 --nphi 40 \\
      --e_min -2 --e_max 0.5 --ne 40 --hv 84 --device cuda
"""

from __future__ import annotations

import argparse
import collections
import collections.abc
import importlib.util
import json
import os
import time

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import numpy as np

import chinook.ARPES_lib as arpes_lib
import chinook.TB_lib as tb_lib
import chinook.klib as klib
import chinook.orbital as olib


def _load_kmesh_module():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "chinook_arpes_kmesh.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {path}")
    spec = importlib.util.spec_from_file_location("chinook_arpes_kmesh", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_tb(tb_file: str, e_fermi: float | None = None):
    data = np.load(tb_file, allow_pickle=True)
    indices = data["indices"]
    values = data["values"]
    basis_list = data["basis_list"]
    a_mat = data["a_mat"].tolist() if "a_mat" in data else np.eye(3).tolist()

    if np.issubdtype(indices.dtype, np.integer):
        print(
            "WARNING: indices are integer-typed — R may be truncated. Re-upload float64 TB.",
            flush=True,
        )

    explicit_hopping = []
    for i in range(len(indices)):
        explicit_hopping.append(
            [
                int(indices[i, 0]),
                int(indices[i, 1]),
                float(indices[i, 2]),
                float(indices[i, 3]),
                float(indices[i, 4]),
                complex(values[i]),
            ]
        )

    tb_dict = {
        "type": "list",
        "list": explicit_hopping,
        "H": explicit_hopping,
        "a": a_mat,
        "spin": {"bool": False, "soc": False},
    }
    bulk_basis = []
    for i, b in enumerate(basis_list):
        orb = olib.orbital(
            i, i, str(b["label"]), b["pos"], int(b.get("Z", 1)), spin=b.get("spin", 1.0)
        )
        bulk_basis.append(orb)
    model = tb_lib.TB_model(bulk_basis, tb_dict, klib.kpath(np.array([[0, 0, 0]])))

    ef = float(e_fermi) if e_fermi is not None else float(
        data["e_fermi"] if "e_fermi" in data else 0.0
    )
    if abs(ef) > 1e-12:
        _orig = model.solve_H

        def _solve_h_ef(Eonly=False, _o=_orig, _ef=ef):
            Eband, Evec = _o(Eonly=Eonly)
            return np.asarray(Eband) - _ef, Evec

        model.solve_H = _solve_h_ef
        print(f"Patched solve_H with e_fermi={ef}", flush=True)
    return model


def load_physics(physics_file: str, hv: float, workf: float, v0: float, temp: float):
    with open(physics_file, "r") as f:
        physics = json.load(f)
    physics.setdefault("hkl", [0, 0, 1])
    physics["hkl"] = tuple(int(x) for x in physics["hkl"])
    physics["hv"] = float(hv)
    physics["work_function"] = float(workf)
    physics["inner_potential"] = float(v0)
    physics["temperature"] = float(temp)
    return physics


def load_b_matrix(tb_file: str, data) -> np.ndarray:
    if "b_matrix" in data:
        return np.asarray(data["b_matrix"], dtype=float)
    a_mat = data["a_mat"].tolist() if "a_mat" in data else np.eye(3).tolist()
    A = np.asarray(a_mat, dtype=float)
    return 2 * np.pi * np.linalg.inv(A).T


def make_arpes_args(theta, phis, e_axis, hv, workf, temp, polar):
    return {
        "cube": {
            "Tx": (theta, theta, 1),
            "Ty": (phis[0], phis[-1], len(phis)),
            "E": (e_axis[0], e_axis[-1], len(e_axis)),
            "kz": 0.0,
        },
        "hv": hv,
        "W": workf,
        "pol": np.array([1, 0, 0]) if polar == "P" else np.array([0, 1, 0]),
        "T": temp,
        "resolution": {"E": 0.02, "k": 0.01},
        "SE": ["constant", 0.05],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tb_file", required=True)
    p.add_argument("--theta", type=float, default=0.0)
    p.add_argument("--phi_min", type=float, required=True)
    p.add_argument("--phi_max", type=float, required=True)
    p.add_argument("--nphi", type=int, required=True)
    p.add_argument("--e_min", type=float, default=-2.0)
    p.add_argument("--e_max", type=float, default=0.5)
    p.add_argument("--ne", type=int, default=40)
    p.add_argument("--hv", type=float, default=84.0)
    p.add_argument("--workf", type=float, default=4.5)
    p.add_argument("--temp", type=float, default=10.0)
    p.add_argument("--polar", type=str, default="P")
    p.add_argument("--device", type=str, default="cuda", choices=("auto", "cpu", "cuda"))
    p.add_argument("--physics_file", default="arpes_physics.json")
    p.add_argument("--v0", type=float, default=12.0)
    p.add_argument("--e_fermi", type=float, default=None)
    p.add_argument("--skip_chinook", action="store_true")
    p.add_argument("--skip_grizzly", action="store_true")
    args = p.parse_args()

    phis = np.linspace(args.phi_min, args.phi_max, args.nphi)
    e_axis = np.linspace(args.e_min, args.e_max, args.ne)

    print("Loading TB...", flush=True)
    t0 = time.perf_counter()
    data = np.load(args.tb_file, allow_pickle=True)
    tb = load_tb(args.tb_file, e_fermi=args.e_fermi)
    b_matrix = load_b_matrix(args.tb_file, data)
    physics = load_physics(
        args.physics_file, args.hv, args.workf, args.v0, args.temp
    )
    print(f"  TB load: {time.perf_counter() - t0:.2f}s", flush=True)

    k_bounds = {
        "X": [float(args.theta), float(args.theta), 1],
        "Y": [float(phis[0]), float(phis[-1]), len(phis)],
        "E": [float(e_axis[0]), float(e_axis[-1]), len(e_axis)],
    }

    t_ch = None
    ig_ch = None
    if not args.skip_chinook:
        kmesh = _load_kmesh_module()
        print("Chinook kmesh spectral (one θ-slice)...", flush=True)
        t0 = time.perf_counter()
        ig_ch = kmesh.run_chinook_arpes(
            tb, k_bounds, physics, b_matrix, fermi_shift=0.0
        )
        t_ch = time.perf_counter() - t0
        ig_ch = np.asarray(ig_ch)
        print(
            f"  chinook: {t_ch:.2f}s  Ig.shape={ig_ch.shape} "
            f"sum={ig_ch.sum():.6e} max={ig_ch.max():.6e}",
            flush=True,
        )

    t_gr = None
    ig_gr = None
    if not args.skip_grizzly:
        kmesh = _load_kmesh_module()
        print(f"GrizzlyME+kmesh spectral (device={args.device})...", flush=True)
        t0 = time.perf_counter()
        ig_gr = kmesh.run_grizzly_arpes(
            tb, k_bounds, physics, b_matrix, fermi_shift=0.0, device=args.device
        )
        t_gr = time.perf_counter() - t0
        ig_gr = np.asarray(ig_gr)
        print(
            f"  grizzly: {t_gr:.2f}s  Ig.shape={ig_gr.shape} "
            f"sum={ig_gr.sum():.6e} max={ig_gr.max():.6e}",
            flush=True,
        )

    if t_ch and t_gr:
        print(f"speedup (chinook/grizzly): {t_ch / t_gr:.2f}x", flush=True)

    if ig_ch is not None and ig_gr is not None:
        a = ig_ch.reshape(-1)
        b = ig_gr.reshape(-1)
        if a.shape == b.shape:
            rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
            print(f"slice rel L2 (kmesh chinook vs kmesh grizzly): {rel:.6e}", flush=True)

    for name, ig in (("chinook", ig_ch), ("grizzly", ig_gr)):
        if ig is None:
            continue
        if float(np.asarray(ig).sum()) == 0.0:
            print(f"SMOKE FAIL: {name} intensity all zeros", flush=True)
            raise SystemExit(2)
    print("SMOKE OK: nonzero intensity", flush=True)


if __name__ == "__main__":
    main()
