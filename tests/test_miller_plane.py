"""Miller cut-plane math contract (mirrors browser guide plane)."""
import unittest
import numpy as np

from tensorspec.core.miller_plane import miller_normal, plane_offset, plane_size


class TestMillerPlane(unittest.TestCase):
    def test_001_cubic_along_c(self):
        cell = np.eye(3) * 4.0  # a,b,c along axes, length 4
        n = miller_normal(cell, (0, 0, 1))
        np.testing.assert_allclose(n, [0, 0, 1], atol=1e-9)

    def test_100_cubic_along_a(self):
        cell = np.eye(3) * 4.0
        n = miller_normal(cell, (1, 0, 0))
        np.testing.assert_allclose(n, [1, 0, 0], atol=1e-9)

    def test_zero_hkl_raises(self):
        cell = np.eye(3)
        with self.assertRaises(ValueError):
            miller_normal(cell, (0, 0, 0))

    def test_depth_offset_scales(self):
        cell = np.eye(3) * 4.0
        n = miller_normal(cell, (0, 0, 1))
        # AABB along z is [0,4]; half extent along n = 2
        off = plane_offset(cell, n, depth_frac=1.0)
        np.testing.assert_allclose(off, [0, 0, 2.0], atol=1e-9)
        off0 = plane_offset(cell, n, depth_frac=0.0)
        np.testing.assert_allclose(off0, [0, 0, 0], atol=1e-9)

    def test_plane_size_positive(self):
        cell = np.eye(3) * 4.0
        self.assertGreater(plane_size(cell), 4.0)


if __name__ == "__main__":
    unittest.main()
