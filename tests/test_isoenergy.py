import unittest

import numpy as np

from tensorspec.core.dft.band_service import isoenergy_density


class TestIsoenergyDensity(unittest.TestCase):
    def test_peaks_at_band(self):
        # eigenvalues shape (nk, nb) with constant band at 0.5
        ev = np.full((4, 2), 0.5)
        near = isoenergy_density(ev, energy=0.5, smear=0.05, grid_shape=(2, 2))
        far = isoenergy_density(ev, energy=5.0, smear=0.05, grid_shape=(2, 2))
        self.assertGreater(near.mean(), far.mean())

    def test_resolution_schema_cap(self):
        from tensorspec.web.server.schemas import IsoenergyRequest

        with self.assertRaises(Exception):
            IsoenergyRequest(resolution=100)  # >48


if __name__ == "__main__":
    unittest.main()
