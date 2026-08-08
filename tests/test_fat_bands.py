"""Unit tests for DFT fat-band orbital projection (stdlib unittest)."""
import unittest

import numpy as np

from tensorspec.core.dft import band_service as bs


LABELS = [
    "C_s", "C_pz", "C_px", "C_py",
    "Ta_s", "Ta_pz", "Ta_px", "Ta_py", "Ta_dz2", "Ta_dxz", "Ta_dyz", "Ta_dx2-y2", "Ta_dxy",
]


class TestFatBands(unittest.TestCase):
    def test_resolve_none(self):
        self.assertEqual(bs.resolve_fat_indices(LABELS, "none"), [])
        self.assertEqual(bs.resolve_fat_indices(LABELS, "None (Standard Lines)"), [])

    def test_resolve_shell_p(self):
        idxs = bs.resolve_fat_indices(LABELS, "shell:p")
        labs = [LABELS[i] for i in idxs]
        self.assertEqual(labs, ["C_pz", "C_px", "C_py", "Ta_pz", "Ta_px", "Ta_py"])

    def test_resolve_shell_d(self):
        idxs = bs.resolve_fat_indices(LABELS, "shell:d")
        self.assertEqual(
            [LABELS[i] for i in idxs],
            ["Ta_dz2", "Ta_dxz", "Ta_dyz", "Ta_dx2-y2", "Ta_dxy"],
        )

    def test_resolve_element(self):
        idxs = bs.resolve_fat_indices(LABELS, "element:C")
        self.assertEqual([LABELS[i] for i in idxs], ["C_s", "C_pz", "C_px", "C_py"])

    def test_resolve_orbital_exact_and_prefixed(self):
        self.assertEqual(bs.resolve_fat_indices(LABELS, "orbital:C_pz"), [1])
        self.assertEqual(bs.resolve_fat_indices(LABELS, "C_pz"), [1])

    def test_resolve_soc_suffix(self):
        soc = ["C_pz_up", "C_pz_dn", "C_s_up"]
        self.assertEqual(bs.resolve_fat_indices(soc, "shell:p"), [0, 1])
        self.assertEqual(bs.resolve_fat_indices(soc, "orbital:C_pz_up"), [0])
        self.assertEqual(bs.resolve_fat_indices(soc, "C_pz"), [0, 1])

    def test_resolve_unknown_raises(self):
        with self.assertRaises(ValueError):
            bs.resolve_fat_indices(LABELS, "shell:f")
        with self.assertRaises(ValueError):
            bs.resolve_fat_indices(LABELS, "orbital:Xe_pz")

    def test_fat_band_weights_shape_and_sum(self):
        rng = np.random.default_rng(0)
        nk, norb, nb = 5, 4, 3
        raw = rng.normal(size=(nk, norb, nb)) + 1j * rng.normal(size=(nk, norb, nb))
        for k in range(nk):
            for b in range(nb):
                raw[k, :, b] /= np.linalg.norm(raw[k, :, b])

        w_all = bs.fat_band_weights(raw, list(range(norb)))
        self.assertEqual(w_all.shape, (nk, nb))
        self.assertTrue(np.allclose(w_all, 1.0, atol=1e-9))

        w0 = bs.fat_band_weights(raw, [0])
        w1 = bs.fat_band_weights(raw, [1])
        w01 = bs.fat_band_weights(raw, [0, 1])
        self.assertTrue(np.allclose(w01, w0 + w1))

        empty = bs.fat_band_weights(raw, [])
        self.assertEqual(empty.shape, (nk, nb))
        self.assertTrue(np.all(empty == 0))


if __name__ == "__main__":
    unittest.main()
