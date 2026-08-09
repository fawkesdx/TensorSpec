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
    def test_rejects_unknown_model(self):
        with TemporaryDirectory() as tmp:
            job = Path(tmp)
            _write_si_job(job, model="B2")
            r = subprocess.run([PY, str(SCRIPT), str(job)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)

    def test_accepts_b1_model_field(self):
        """B1 is a valid model string; may exit 6/4 if chinook missing in CI."""
        with TemporaryDirectory() as tmp:
            job = Path(tmp)
            _write_si_job(job, model="B1", mesh=4, steps=4)
            r = subprocess.run([PY, str(SCRIPT), str(job)], capture_output=True, text=True)
            self.assertNotIn("Option A only", r.stderr + r.stdout)
            self.assertIn(r.returncode, (0, 4, 6))

    def test_missing_request_exit_2(self):
        with TemporaryDirectory() as tmp:
            job = Path(tmp)
            Structure(Lattice.cubic(5.0), ["Si"], [[0, 0, 0]]).to(
                filename=str(job / "structure.cif")
            )
            r = subprocess.run([PY, str(SCRIPT), str(job)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)

    def test_missing_axis_keys_exit_2(self):
        with TemporaryDirectory() as tmp:
            job = Path(tmp)
            _write_si_job(job)
            req = json.loads((job / "request.json").read_text(encoding="utf-8"))
            del req["kx"]
            (job / "request.json").write_text(json.dumps(req), encoding="utf-8")
            r = subprocess.run([PY, str(SCRIPT), str(job)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, msg=r.stderr + r.stdout)
            self.assertIn("invalid request", (job / "remote_arpes_me.log").read_text(encoding="utf-8"))

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
