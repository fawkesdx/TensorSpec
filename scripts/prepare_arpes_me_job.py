#!/usr/bin/env python3
"""Write a remote ARPES ME job directory from a CIF + optional JSON overrides."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from pymatgen.core import Structure

_DEFAULT_REQUEST = {
    "model": "A",
    "crystal_name": "crystal",
    "photon_energy": 90.0,
    "work_function": 4.5,
    "inner_potential": 15.0,
    "temperature": 10.0,
    "incidence_angle": 55.0,
    "polarization": "Linear Horizontal",
    "lin_pol_angle": 45.0,
    "matrix_element_mode": "Bare Spectral Function (ME Off)",
    "manip_theta": 0.0,
    "manip_azimuth": 0.0,
    "manip_tilt": 0.0,
    "h": 0,
    "k": 0,
    "l": 1,
    "kx": {"min": -1.0, "max": 1.0, "steps": 20},
    "ky": {"min": -1.0, "max": 1.0, "steps": 20},
    "energy": {"min": -2.0, "max": 0.5, "steps": 40},
    "se_width": 0.01,
    "res_E": 0.02,
    "res_k": 0.05,
    "slit_angle": 15.0,
    "mesh_resolution": 20,
    "hoppings": [2.7, 0.0, 0.0, -0.3],
    "cutoffs": [1.6, 2.6, 3.1, 4.5],
    "onsite_e": 0.0,
    "tb_mode": "Simple Scalar (Isotropic)",
    "store_as": "simulated_arpes",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cif", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--request", type=Path, default=None, help="JSON overrides merged onto defaults")
    args = p.parse_args()
    if not args.cif.is_file():
        print(f"error: CIF missing: {args.cif}", file=sys.stderr)
        return 2
    struct = Structure.from_file(str(args.cif))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.cif, args.out_dir / "structure.cif")
    req = dict(_DEFAULT_REQUEST)
    req["crystal_name"] = struct.composition.reduced_formula
    if args.request is not None:
        overrides = json.loads(args.request.read_text(encoding="utf-8"))
        req.update(overrides)
    req["model"] = "A"
    (args.out_dir / "request.json").write_text(json.dumps(req, indent=2), encoding="utf-8")
    print(f"wrote {args.out_dir}/structure.cif + request.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
