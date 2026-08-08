"""Tests for QE slab presets and suggest_slab_qe."""
import unittest

from pymatgen.core import Lattice, Structure

from tensorspec.core.dft import qe_slab
from tensorspec.core.dft.qe_pipeline import PipelineParams


class TestQeSlab(unittest.TestCase):
    def test_presets_resolve(self):
        hkl, n, vac = qe_slab.resolve_slab_params("medium_001")
        self.assertEqual(hkl, (0, 0, 1))
        self.assertEqual(n, 3)
        self.assertEqual(vac, 15.0)

    def test_custom_resolve(self):
        hkl, n, vac = qe_slab.resolve_slab_params("custom", h=1, k=1, l=0, num_layers=2, vacuum=18)
        self.assertEqual(hkl, (1, 1, 0))
        self.assertEqual(n, 2)
        self.assertEqual(vac, 18.0)

    def test_suggest_slab_qe(self):
        bulk = Structure(Lattice.cubic(4.0), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        slabby = Structure(Lattice.hexagonal(2.46, 25.0), ["C", "C"], [[0, 0, 0.5], [1 / 3, 2 / 3, 0.5]])
        self.assertFalse(qe_slab.suggest_slab_qe(bulk))
        self.assertTrue(qe_slab.suggest_slab_qe(slabby))

    def test_pipeline_slab_kmesh(self):
        p = PipelineParams(kx=8, ky=8, kz=6, slab_mode=True)
        self.assertEqual(p.kmesh, (8, 8, 1))
        p2 = PipelineParams(kx=8, ky=8, kz=6, slab_mode=False)
        self.assertEqual(p2.kmesh, (8, 8, 6))


if __name__ == "__main__":
    unittest.main()
