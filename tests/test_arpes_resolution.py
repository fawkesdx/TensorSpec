"""Analyzer ΔE and deflector Δk helpers."""
import math
import unittest

from tensorspec.core.arpes.resolution import (
    analyzer_delta_e,
    total_delta_e,
    deflector_dk,
)


class TestArpesResolution(unittest.TestCase):
    def test_analyzer_0p2mm_20eV(self):
        # (0.2/400)*20 = 0.01 eV
        self.assertAlmostEqual(analyzer_delta_e(0.2, 20.0), 0.01, places=6)

    def test_total_quadrature(self):
        self.assertAlmostEqual(total_delta_e(0.03, 0.04, 0.0), 0.05, places=6)

    def test_deflector_zero(self):
        self.assertAlmostEqual(deflector_dk(90.0, 4.5, 0.0), 0.0, places=9)

    def test_deflector_sign(self):
        dk = deflector_dk(90.0, 4.5, 15.0)
        self.assertGreater(dk, 0.0)
        self.assertAlmostEqual(deflector_dk(90.0, 4.5, -15.0), -dk, places=9)


if __name__ == "__main__":
    unittest.main()
