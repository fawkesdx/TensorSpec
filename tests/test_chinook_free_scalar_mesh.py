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
                kx_min=-0.3,
                kx_max=0.3,
                ky_min=-0.3,
                ky_max=0.3,
                resolution=4,
                shell_keys=tuple(hop.keys()),
                hoppings=tuple(hop.values()),
                tb_mode="Simple Scalar (Isotropic)",
            )
        self.assertEqual(mesh["grid_shape"], (4, 4))
        self.assertEqual(mesh["eigenvalues"].shape[0], 16)
