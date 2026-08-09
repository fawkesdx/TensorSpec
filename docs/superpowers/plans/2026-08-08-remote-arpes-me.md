# Remote ARPES ME CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Mac CLI `scripts/remote_arpes_me.sh` + Einstein entry `scripts/run_arpes_me_a.py` so Option A ME jobs run on Einstein CPU and pull `intensity.npz`.

**Architecture:** Prepare job dir on Mac (`request.json` + structure) → rsync to Einstein scratch → `TensorSpec_env` Python runs Option A (TB mesh + three-step) → pull allowlist → wipe scratch on success. Chinook-free Simple Scalar mesh when chinook missing (Einstein today).

**Tech Stack:** bash, ssh, rsync, numpy, pymatgen, TensorSpec `band_service` + `ARPESEngineRouter` Option A.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-remote-arpes-me-design.md`
- Branch: `HTML_einstein_app` only — never merge to `main`
- Option **A** only; refuse `model=B1`
- No web ARPES Queue `einstein_ssh` in this plan
- No live SSH in CI; `--dry-run` = zero network
- Scratch: `/data/sandy/arpes_me_scratch` if writable else `$HOME/arpes_me_scratch`
- Caps: reuse `MAX_SIM_VOXELS = 80*80*80`, `MAX_MESH_POINTS = 40*40` (same as `tensorspec/web/server/routers/arpes.py`)
- Exit codes: 0 ok; 2 validation; 4 sim fail; 6 missing dep (non-Scalar / SOC without chinook)
- Default host `einstein`; `TENSORSPEC_ROOT` default `$HOME/TensorSpec`

## File map

| File | Role |
|------|------|
| `tensorspec/core/dft/chinook_tb.py` | Chinook-free Simple Scalar `solve_bands` fallback |
| `scripts/run_arpes_me_a.py` | Einstein entry: load job → mesh → Option A → npz |
| `scripts/prepare_arpes_me_job.py` | Optional Mac helper: CIF + overrides → job dir |
| `scripts/remote_arpes_me.sh` | Mac CLI: rsync / SSH / pull / wipe |
| `scripts/README-remote-arpes-me.md` | Usage + policies |
| `tests/test_chinook_free_scalar_mesh.py` | Numpy Scalar mesh without chinook |
| `tests/test_run_arpes_me_a.py` | Validation / B1 reject / tiny Option A |
| `tests/test_remote_arpes_me_script.py` | Dry-run contract + allowlist strings |

---

### Task 1: Chinook-free Simple Scalar `solve_bands`

**Files:**
- Modify: `tensorspec/core/dft/chinook_tb.py` (`solve_bands` and small helper)
- Test: `tests/test_chinook_free_scalar_mesh.py`

**Interfaces:**
- Consumes: existing `export_chinook_dictionary` list mode (`"Scalar" in tb_mode` and `not use_soc`)
- Produces: `solve_bands(...)` returns `(eigenvalues, eigenvectors, orb_labels)` with shapes `(nk, nb)`, `(nk, nb, nb)` or chinook-compatible eigenvector layout, labels `list[str]` — same contract as today when chinook present
- Locked rule: if `build_lib is None` and (`"Scalar" not in tb_mode` or `use_soc` or `w90_filepath`): raise `ImportError` (caller maps to exit 6)
- Locked rule: if `build_lib is None` and Simple Scalar + no SOC: **numpy path**, do not raise

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chinook_free_scalar_mesh.py
"""Simple Scalar mesh works when chinook is absent."""
import unittest
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice, Structure

from tensorspec.core.dft import chinook_tb as ct
from tensorspec.core.dft import band_service


class TestChinookFreeScalarMesh(unittest.TestCase):
    def test_solve_bands_numpy_when_chinook_missing(self):
        lat = Lattice.cubic(5.43)
        struct = Structure(lat, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        eng = ct.ChinookTightBindingEngine()
        eng.crystal_structure = struct
        k = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float)
        with patch.object(ct, "build_lib", None), patch.object(ct, "klib", None):
            evals, evecs, labels = eng.solve_bands(
                k,
                custom_hopping={"M-M": -1.5, "M-X": -1.2},
                cutoffs=[2.5, 4.0],
                tb_mode="Simple Scalar (Isotropic)",
                use_soc=False,
            )
        self.assertEqual(evals.shape[0], 2)
        self.assertGreater(evals.shape[1], 0)
        self.assertEqual(len(labels), evals.shape[1])
        self.assertTrue(np.isfinite(evals).all())

    def test_sk_mode_raises_without_chinook(self):
        lat = Lattice.cubic(5.0)
        struct = Structure(lat, ["Si"], [[0, 0, 0]])
        eng = ct.ChinookTightBindingEngine()
        eng.crystal_structure = struct
        with patch.object(ct, "build_lib", None), patch.object(ct, "klib", None):
            with self.assertRaises(ImportError):
                eng.solve_bands(
                    np.zeros((1, 3)),
                    custom_hopping={"M-M": -1.0},
                    tb_mode="Slater-Koster (Rigorous)",
                )

    def test_calculate_2d_mesh_with_patched_missing_chinook(self):
        lat = Lattice.cubic(5.43)
        struct = Structure(lat, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        eng = ct.ChinookTightBindingEngine()
        eng.crystal_structure = struct
        hop = eng.get_default_hopping("Si")
        with patch.object(ct, "build_lib", None), patch.object(ct, "klib", None):
            mesh = band_service.calculate_2d_mesh(
                eng,
                -0.3,
                0.3,
                -0.3,
                0.3,
                resolution=4,
                shell_keys=tuple(hop.keys()),
                hoppings=tuple(hop.values()),
                tb_mode="Simple Scalar (Isotropic)",
            )
        self.assertEqual(mesh["grid_shape"], (4, 4))
        self.assertEqual(mesh["eigenvalues"].shape[0], 16)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/sandyai/Documents/GitHub/TensorSpec
./TensorSpec_env/bin/python -m pytest tests/test_chinook_free_scalar_mesh.py -v
```

Expected: FAIL — `ImportError: Chinook is not installed` (or module/path missing helper).

- [ ] **Step 3: Implement numpy Simple Scalar path**

In `chinook_tb.py`, replace the hard fail at the start of `solve_bands` with:

```python
def solve_bands(self, k_points, custom_hopping=None, onsite_e=0.0, use_soc=False,
                soc_strength=0.5, w90_filepath=None, cutoffs=None,
                tb_mode="Simple Scalar", orbital_shifts=None):
    if not self.crystal_structure:
        raise ValueError("No structure loaded.")

    chinook_missing = build_lib is None or klib is None
    scalar_ok = ("Scalar" in (tb_mode or "")) and (not use_soc) and (not w90_filepath)
    if chinook_missing and not scalar_ok:
        raise ImportError(
            "Chinook is not installed. Simple Scalar (no SOC) works without it; "
            "Slater-Koster / SOC / Wannier require: pip install chinook"
        )
    if chinook_missing and scalar_ok:
        return self._solve_bands_simple_scalar_numpy(
            k_points,
            custom_hopping=custom_hopping,
            onsite_e=onsite_e,
            cutoffs=cutoffs,
            orbital_shifts=orbital_shifts,
            tb_mode=tb_mode,
        )
    # ... existing chinook path unchanged ...
```

Add helper (same file):

```python
def _solve_bands_simple_scalar_numpy(
    self, k_points, custom_hopping=None, onsite_e=0.0, cutoffs=None,
    orbital_shifts=None, tb_mode="Simple Scalar",
):
    """Diagonalize isotropic list-mode hoppings with numpy (no chinook)."""
    shells = []
    if custom_hopping:
        distances = cutoffs if cutoffs else [1.6, 2.6, 3.1, 4.5]
        for i, (key, t_val) in enumerate(custom_hopping.items()):
            r_max = distances[i] if i < len(distances) else 10.0
            shells.append((t_val, r_max))

    tb_dict = self.export_chinook_dictionary(
        shells=shells,
        onsite_e=onsite_e,
        use_soc=False,
        soc_strength=0.0,
        tb_mode=tb_mode,
        orbital_shifts=orbital_shifts,
    )
    hops = tb_dict.get("list") or tb_dict.get("H") or []
    if not hops:
        raise RuntimeError("Simple Scalar hopping list empty.")

    n_orb = int(max(max(int(h[0]), int(h[1])) for h in hops)) + 1
    k_points = np.asarray(k_points, dtype=float)
    nk = k_points.shape[0]
    evals = np.zeros((nk, n_orb), dtype=float)
    evecs = np.zeros((nk, n_orb, n_orb), dtype=complex)

    for ik, k in enumerate(k_points):
        H = np.zeros((n_orb, n_orb), dtype=complex)
        for row in hops:
            ia, ib = int(row[0]), int(row[1])
            Rx, Ry, Rz = float(row[2]), float(row[3]), float(row[4])
            t = complex(row[5])
            phase = np.exp(1j * (k[0] * Rx + k[1] * Ry + k[2] * Rz))
            H[ia, ib] += t * phase
        # Hermitize numerical noise
        H = 0.5 * (H + H.conj().T)
        w, v = np.linalg.eigh(H)
        evals[ik] = np.real(w)
        evecs[ik] = v

    raw_labels = []
    for site in self.crystal_structure:
        for orb_str in self._get_orbital_basis(site.species_string):
            if orb_str.endswith("0"):
                orb_name = "s"
            elif orb_str[1] == "1":
                orb_name = "p" + orb_str[2:]
            elif orb_str[1] == "2":
                if orb_str.endswith("ZR"):
                    orb_name = "dz2"
                elif orb_str.endswith("XY"):
                    orb_name = "dx2-y2"
                else:
                    orb_name = "d" + orb_str[2:]
            else:
                orb_name = orb_str
            raw_labels.append(f"{site.species_string}_{orb_name}")

    self.basis = None
    self.H_dict = tb_dict
    self.tb_model = None
    return evals, evecs, raw_labels
```

Keep existing chinook branch body after the early return (do not delete `build_lib` path).

- [ ] **Step 4: Run tests to verify they pass**

```bash
./TensorSpec_env/bin/python -m pytest tests/test_chinook_free_scalar_mesh.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tensorspec/core/dft/chinook_tb.py tests/test_chinook_free_scalar_mesh.py
git commit -m "$(cat <<'EOF'
feat: chinook-free Simple Scalar TB diagonalization

Einstein TensorSpec_env lacks chinook; Option A remote ME needs numpy mesh.
EOF
)"
```

---

### Task 2: `run_arpes_me_a.py` + prepare helper + unit tests

**Files:**
- Create: `scripts/run_arpes_me_a.py`
- Create: `scripts/prepare_arpes_me_job.py`
- Test: `tests/test_run_arpes_me_a.py`

**Interfaces:**
- CLI: `python scripts/run_arpes_me_a.py <job_dir>` → exit 0/2/4/6
- Job dir: `request.json` (`model` must be `"A"`) + exactly one of `structure.cif` | `structure.json`
- Writes: `intensity.npz` keys `intensity`, `E`, `kx`, `ky`; `meta.json`; appends `remote_arpes_me.log`
- Caps: import or duplicate `MAX_SIM_VOXELS`, `MAX_MESH_POINTS` from router constants (prefer local constants matching web to avoid importing FastAPI app):

```python
MAX_SIM_VOXELS = 80 * 80 * 80
MAX_MESH_POINTS = 40 * 40
```

- Prepare CLI: `python scripts/prepare_arpes_me_job.py <cif> <out_dir> [--request overrides.json]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_run_arpes_me_a.py
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pymatgen.core import Lattice, Structure

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_arpes_me_a.py"
PREPARE = REPO / "scripts" / "prepare_arpes_me_job.py"
PY = sys.executable


def _write_si_job(job: Path, model="A", mesh=4, steps=4):
    struct = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    struct.to(filename=str(job / "structure.cif"))
    req = {
        "model": model,
        "crystal_name": "Si",
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
        "kx": {"min": -0.3, "max": 0.3, "steps": steps},
        "ky": {"min": -0.3, "max": 0.3, "steps": steps},
        "energy": {"min": -1.0, "max": 0.5, "steps": steps},
        "se_width": 0.01,
        "res_E": 0.02,
        "res_k": 0.05,
        "slit_angle": 15.0,
        "mesh_resolution": mesh,
        "hoppings": [2.7, 0.0, 0.0, -0.3],
        "cutoffs": [1.6, 2.6, 3.1, 4.5],
        "onsite_e": 0.0,
        "tb_mode": "Simple Scalar (Isotropic)",
        "store_as": "simulated_arpes",
    }
    (job / "request.json").write_text(json.dumps(req), encoding="utf-8")


class TestRunArpesMeA(unittest.TestCase):
    def test_rejects_b1(self):
        with TemporaryDirectory() as tmp:
            job = Path(tmp)
            _write_si_job(job, model="B1")
            r = subprocess.run([PY, str(SCRIPT), str(job)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn("Option A", (r.stderr + r.stdout))

    def test_missing_request_exit_2(self):
        with TemporaryDirectory() as tmp:
            job = Path(tmp)
            Structure(Lattice.cubic(5.0), ["Si"], [[0, 0, 0]]).to(
                filename=str(job / "structure.cif")
            )
            r = subprocess.run([PY, str(SCRIPT), str(job)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)

    def test_tiny_option_a_writes_npz(self):
        with TemporaryDirectory() as tmp:
            job = Path(tmp)
            _write_si_job(job, mesh=4, steps=4)
            r = subprocess.run([PY, str(SCRIPT), str(job)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, msg=r.stderr + r.stdout)
            import numpy as np

            data = np.load(job / "intensity.npz")
            self.assertIn("intensity", data.files)
            self.assertEqual(data["intensity"].ndim, 3)
            self.assertEqual(tuple(data["intensity"].shape), (4, 4, 4))
            meta = json.loads((job / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["model"], "A")
```

- [ ] **Step 2: Run tests — expect FAIL (script missing)**

```bash
./TensorSpec_env/bin/python -m pytest tests/test_run_arpes_me_a.py -v
```

- [ ] **Step 3: Implement `scripts/run_arpes_me_a.py`**

```python
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
```

- [ ] **Step 4: Implement `scripts/prepare_arpes_me_job.py`**

```python
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
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
./TensorSpec_env/bin/python -m pytest tests/test_run_arpes_me_a.py -v
```

- [ ] **Step 6: Commit**

```bash
git add scripts/run_arpes_me_a.py scripts/prepare_arpes_me_job.py tests/test_run_arpes_me_a.py
git commit -m "$(cat <<'EOF'
feat: add Option A remote ARPES ME job runner

Job dir contract + prepare helper; Einstein entry writes intensity.npz.
EOF
)"
```

---

### Task 3: `remote_arpes_me.sh` + README + contract tests

**Files:**
- Create: `scripts/remote_arpes_me.sh` (executable)
- Create: `scripts/README-remote-arpes-me.md`
- Test: `tests/test_remote_arpes_me_script.py`
- Modify: `tests/test_remote_scratch_wipe.py` — add sidecar assert for new script (optional one-liner in new test file instead)

**Interfaces:**
- CLI: `./scripts/remote_arpes_me.sh <local_job_dir> [--host einstein] [--keep-scratch] [--dry-run]`
- Mirror `scripts/remote_qe.sh` patterns: dry-run zero network; sidecar `.tensorspec_remote_scratch`; allowlist pull; wipe on success
- Allowlist: `intensity.npz` `meta.json` `remote_arpes_me.log`
- Remote run:

```bash
cd "$SCRATCH"
export PYTHONPATH="${TENSORSPEC_ROOT:-$HOME/TensorSpec}"
PY="${TENSORSPEC_ROOT:-$HOME/TensorSpec}/TensorSpec_env/bin/python"
"$PY" "$PYTHONPATH/scripts/run_arpes_me_a.py" .
```

- [ ] **Step 1: Write failing contract test**

```python
# tests/test_remote_arpes_me_script.py
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "remote_arpes_me.sh"


class TestRemoteArpesMeScript(unittest.TestCase):
    def test_script_contract_strings(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("arpes_me_scratch", text)
        self.assertIn(".tensorspec_remote_scratch", text)
        self.assertIn("intensity.npz", text)
        self.assertIn("run_arpes_me_a.py", text)
        self.assertIn("--dry-run", text)

    def test_dry_run_zero_network(self):
        with TemporaryDirectory() as tmp:
            job = Path(tmp)
            (job / "request.json").write_text('{"model":"A"}', encoding="utf-8")
            (job / "structure.cif").write_text("data_dummy\n", encoding="utf-8")
            r = subprocess.run(
                ["bash", str(SCRIPT), str(job), "--dry-run"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, msg=r.stderr + r.stdout)
            out = r.stdout + r.stderr
            self.assertIn("dry-run", out.lower())
            self.assertNotIn("ssh ", out)  # plan print only; no live ssh lines required
```

Note: dry-run may print `ssh` in the plan text — assert **exit 0** and that the script does not actually call network. Prefer asserting a `DRY=1` path that never invokes `ssh`/`rsync` (match `remote_qe.sh`).

- [ ] **Step 2: Run test — expect FAIL**

```bash
./TensorSpec_env/bin/python -m pytest tests/test_remote_arpes_me_script.py -v
```

- [ ] **Step 3: Implement `scripts/remote_arpes_me.sh`**

Mirror `scripts/remote_qe.sh` structure. Critical differences:

- Require `request.json` + (`structure.cif` XOR present — accept either cif or json; refuse if neither)
- Scratch names: `arpes_me_scratch`
- No MPI / pw.x
- Remote python as above; capture exit code from remote script (map to local 4 or 6 if remote returns those)
- Sidecar: `printf '%s\t%s\n' "$HOST" "$SCRATCH" >"$JOB_DIR/.tensorspec_remote_scratch"`
- `--dry-run`: print planned HOST, scratch assumption `$HOME/arpes_me_scratch/<job_id>`, rsync, remote cmd, allowlist; **exit 0 without ssh/rsync**

Skeleton (fill fully like remote_qe):

```bash
#!/usr/bin/env bash
set -euo pipefail
# parse --host --keep-scratch --dry-run
# validate JOB_DIR/request.json + structure
# JOB_ID=...
# ALLOWLIST=(intensity.npz meta.json remote_arpes_me.log)
# if DRY: print plan; exit 0
# resolve_scratch_root via ssh (arpes_me_scratch)
# write sidecar
# rsync -az --delete "$JOB_DIR/" "$HOST:$SCRATCH/"
# ssh run python; REMOTE_RC=$?
# pull allowlist files that exist
# if REMOTE_RC!=0: keep scratch; exit $REMOTE_RC (or 4)
# if success && !KEEP: ssh rm -rf SCRATCH
```

Map remote exit: preserve 2/4/6 when pulled from remote; connectivity failures → 1.

- [ ] **Step 4: Write `scripts/README-remote-arpes-me.md`**

Sections: Prerequisites (SSH `einstein`, `~/TensorSpec` + `TensorSpec_env`, chinook **not** required for Simple Scalar), Job dir layout, `prepare_arpes_me_job.py` example, Usage (`--dry-run`, live), Scratch policy, Pull allowlist, Exit codes, B1 out of scope, Web Queue later.

- [ ] **Step 5: `chmod +x scripts/remote_arpes_me.sh`**

- [ ] **Step 6: Run tests**

```bash
./TensorSpec_env/bin/python -m pytest tests/test_remote_arpes_me_script.py tests/test_run_arpes_me_a.py tests/test_chinook_free_scalar_mesh.py -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/remote_arpes_me.sh scripts/README-remote-arpes-me.md tests/test_remote_arpes_me_script.py
git commit -m "$(cat <<'EOF'
feat: add remote_arpes_me.sh CLI for Einstein Option A

Mac job dir rsync + pull intensity.npz; dry-run stays offline.
EOF
)"
```

---

### Task 4: Push branch + Einstein pull note (controller)

**Files:** none new (ops)

- [ ] **Step 1: Push `HTML_einstein_app`**

```bash
git push -u origin HEAD
```

- [ ] **Step 2: Einstein pull** (controller or user machine with SSH)

```bash
ssh einstein 'cd ~/TensorSpec && git fetch && git checkout HTML_einstein_app && git pull'
```

- [ ] **Step 3: Optional live smoke** (not CI)

```bash
# on Mac after prepare:
./scripts/prepare_arpes_me_job.py /path/to/Si.cif /tmp/arpes_me_si
# edit request.json mesh_resolution=4, steps=4 for speed
./scripts/remote_arpes_me.sh /tmp/arpes_me_si
ls /tmp/arpes_me_si/intensity.npz
```

Do **not** claim live smoke green without running it. If SSH blocked in session, document commands for user / retry with approval.

---

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| Job dir contract | 2 |
| `run_arpes_me_a.py` | 2 |
| Chinook-free Scalar (pref 2) | 1 |
| Exit 2/4/6 | 2 |
| `remote_arpes_me.sh` mirror QE | 3 |
| Scratch policy | 3 |
| Sidecar | 3 |
| README | 3 |
| Unit + dry-run tests | 1–3 |
| B1 refused | 2 |
| No web Queue | (non-goal) |
| Einstein deploy | 4 |

## Self-review notes

- No TBD placeholders.
- Caps duplicated as constants in runner (avoid FastAPI import on Einstein CLI).
- `custom_hopping` dict order: `get_default_hopping` returns dict; `pack_hopping` / enumerate order must match cutoffs — same as web worker.
