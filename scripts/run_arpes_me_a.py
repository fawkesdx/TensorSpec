#!/usr/bin/env python3
"""Run ARPES Option A ME in a prepared job directory (Einstein entrypoint)."""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

# Ensure repo root importable when invoked as scripts/run_arpes_me_a.py
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tensorspec.core.arpes_engine import ARPESEngineRouter
from tensorspec.core.dft import band_service
from tensorspec.core.dft_engine import DFTEngineRouter

MAX_SIM_VOXELS = 80 * 80 * 80
MAX_MESH_POINTS = 40 * 40


def _log(job_dir: Path, msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    print(line, end="")
    with (job_dir / "remote_arpes_me.log").open("a", encoding="utf-8") as fh:
        fh.write(line)


def _load_structure(job_dir: Path) -> Structure:
    cif = job_dir / "structure.cif"
    js = job_dir / "structure.json"
    if cif.is_file() and js.is_file():
        raise ValueError("provide exactly one of structure.cif or structure.json")
    if cif.is_file():
        return Structure.from_file(str(cif))
    if js.is_file():
        return Structure.from_dict(json.loads(js.read_text(encoding="utf-8")))
    raise ValueError("missing structure.cif or structure.json")


def _axis(d: dict, name: str):
    a = d[name]
    return float(a["min"]), float(a["max"]), int(a["steps"])


def _experiment_kwargs(req: dict) -> dict:
    return {
        "photon_energy": float(req.get("photon_energy", 90.0)),
        "work_function": float(req.get("work_function", 4.5)),
        "inner_potential": float(req.get("inner_potential", 15.0)),
        "temperature": float(req.get("temperature", 10.0)),
        "incidence_angle": float(req.get("incidence_angle", 55.0)),
        "polarization": req.get("polarization", "Linear Horizontal"),
        "lin_pol_angle": float(req.get("lin_pol_angle", 45.0)),
        "matrix_element_mode": req.get(
            "matrix_element_mode", "Bare Spectral Function (ME Off)"
        ),
        "manip_theta": float(req.get("manip_theta", 0.0)),
        "manip_azimuth": float(req.get("manip_azimuth", 0.0)),
        "manip_tilt": float(req.get("manip_tilt", 0.0)),
        "hkl": (int(req.get("h", 0)), int(req.get("k", 0)), int(req.get("l", 1))),
        "k_bounds": {
            "X": list(_axis(req, "kx")),
            "Y": list(_axis(req, "ky")),
            "E": list(_axis(req, "energy")),
        },
        "se_width": float(req.get("se_width", 0.01)),
        "res_E": float(req.get("res_E", 0.02)),
        "res_k": float(req.get("res_k", 0.05)),
        "slit_angle": float(req.get("slit_angle", 15.0)),
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("Usage: run_arpes_me_a.py <job_dir>", file=sys.stderr)
        return 2
    job_dir = Path(argv[0]).resolve()
    if not job_dir.is_dir():
        print(f"error: not a directory: {job_dir}", file=sys.stderr)
        return 2

    log_path = job_dir / "remote_arpes_me.log"
    log_path.write_text("", encoding="utf-8")

    try:
        req_path = job_dir / "request.json"
        if not req_path.is_file():
            raise ValueError("request.json missing")
        req = json.loads(req_path.read_text(encoding="utf-8"))
        model = str(req.get("model", "A"))
        if model != "A":
            raise ValueError(
                f"remote runner supports Option A only (got model={model!r}); B1 out of scope"
            )

        kx = _axis(req, "kx")
        ky = _axis(req, "ky")
        energy = _axis(req, "energy")
        voxels = kx[2] * ky[2] * energy[2]
        mesh_res = int(req.get("mesh_resolution", 20))
        if voxels > MAX_SIM_VOXELS:
            raise ValueError(f"detector voxels {voxels} > cap {MAX_SIM_VOXELS}")
        if mesh_res * mesh_res > MAX_MESH_POINTS:
            raise ValueError(f"mesh points {mesh_res**2} > cap {MAX_MESH_POINTS}")

        structure = _load_structure(job_dir)
        _log(job_dir, f"[arpes] structure {structure.composition.reduced_formula} "
             f"({len(structure)} sites)")

        engine = DFTEngineRouter()
        engine.load_structure(structure)
        chinook = engine.chinook
        shells = chinook.get_default_hopping(structure.composition.reduced_formula)
        shell_keys = list(shells.keys())
        hoppings = list(req.get("hoppings", [2.7, 0.0, 0.0, -0.3])[: len(shell_keys)])
        while len(hoppings) < len(shell_keys):
            hoppings.append(0.0)

        tb_mode = req.get("tb_mode", "Simple Scalar (Isotropic)")
        _log(job_dir, f"[arpes] mesh {mesh_res}x{mesh_res} tb_mode={tb_mode}")
        try:
            band_data = band_service.calculate_2d_mesh(
                chinook,
                kx_min=kx[0],
                kx_max=kx[1],
                ky_min=ky[0],
                ky_max=ky[1],
                resolution=mesh_res,
                shell_keys=shell_keys,
                hoppings=hoppings,
                cutoffs=req.get("cutoffs", [1.6, 2.6, 3.1, 4.5]),
                onsite_e=float(req.get("onsite_e", 0.0)),
                tb_mode=tb_mode,
            )
        except ImportError as e:
            _log(job_dir, f"[arpes] missing dependency: {e}")
            return 6

        _log(job_dir, f"[arpes] mesh ready ({band_data['n_bands']} bands)")
        arpes = ARPESEngineRouter()
        results = arpes.run_simulation("A", band_data, _experiment_kwargs(req))
        intensity = np.asarray(results["intensity_broadened"], dtype=float)
        kx_ax, ky_ax = results.get("k_axes", (None, None))
        e_ax = results.get("e_axis")
        if kx_ax is None:
            kx_ax = np.linspace(kx[0], kx[1], intensity.shape[0])
            ky_ax = np.linspace(ky[0], ky[1], intensity.shape[1])
            e_ax = np.linspace(energy[0], energy[1], intensity.shape[2])
        cube = np.transpose(intensity, (2, 0, 1))  # (E, kx, ky)
        np.savez_compressed(
            job_dir / "intensity.npz",
            intensity=cube,
            E=np.asarray(e_ax, dtype=float),
            kx=np.asarray(kx_ax, dtype=float),
            ky=np.asarray(ky_ax, dtype=float),
        )
        meta = {
            "model": "A",
            "shape": list(cube.shape),
            "formula": structure.composition.reduced_formula,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tb_mode": tb_mode,
            "mesh_resolution": mesh_res,
        }
        (job_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _log(job_dir, f"[arpes] wrote intensity.npz shape {list(cube.shape)}")
        return 0
    except ValueError as e:
        _log(job_dir, f"error: {e}")
        return 2
    except Exception as e:
        _log(job_dir, f"error: simulation failed: {e}")
        _log(job_dir, traceback.format_exc())
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
