import unittest

import numpy as np

from tensorspec.core import peem_sumrule as sr


class TestApplyI0(unittest.TestCase):
    def test_none_unchanged(self):
        spec = np.array([1.0, 2.0, 3.0])
        out, applied = sr.apply_i0(spec, None)
        np.testing.assert_allclose(out, spec)
        self.assertFalse(applied)

    def test_scalar_normalizes(self):
        spec = np.array([2.0, 4.0, 6.0])
        out, applied = sr.apply_i0(spec, 2.0)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])
        self.assertTrue(applied)

    def test_array_matching_length(self):
        spec = np.array([10.0, 20.0, 30.0])
        i0 = np.array([2.0, 4.0, 5.0])
        out, applied = sr.apply_i0(spec, i0)
        np.testing.assert_allclose(out, [5.0, 5.0, 6.0])
        self.assertTrue(applied)

    def test_length_mismatch_skips(self):
        spec = np.array([1.0, 2.0, 3.0])
        out, applied = sr.apply_i0(spec, np.array([1.0, 2.0]))
        np.testing.assert_allclose(out, spec)
        self.assertFalse(applied)


class TestIntegrateWindows(unittest.TestCase):
    def test_square_pulses_analytic(self):
        # energy 0..10 step 1; baseline sμ=4; L3=[2,4] dμ=1; L2=[6,8] dμ=2
        energy = np.linspace(0.0, 10.0, 11)
        mu_plus = np.full_like(energy, 2.0)
        mu_minus = np.full_like(energy, 2.0)
        l3_mask = (energy >= 2.0) & (energy <= 4.0)
        mu_plus[l3_mask] = 3.0  # dμ=1
        l2_mask = (energy >= 6.0) & (energy <= 8.0)
        mu_plus[l2_mask] = 4.0  # dμ=2

        out = sr.integrate_windows(
            energy,
            mu_plus,
            mu_minus,
            l3=(2.0, 4.0),
            l2=(6.0, 8.0),
            r_win=(9.0, 10.0),  # flat sμ=4 outside L edges
        )
        self.assertAlmostEqual(out["p"], 2.0, places=6)
        self.assertAlmostEqual(out["q"], 6.0, places=6)
        self.assertAlmostEqual(out["r"], 4.0, places=6)

    def test_overlapping_l3_l2_no_double_count(self):
        # L3=[2,5] L2=[4,7] overlap [4,5]; dμ=1 on union → q=5 not 6
        energy = np.linspace(0.0, 10.0, 11)
        mu_plus = np.full_like(energy, 3.0)
        mu_minus = np.full_like(energy, 2.0)
        outside = (energy < 2.0) | (energy > 7.0)
        mu_plus[outside] = 2.0
        mu_minus[outside] = 2.0

        out = sr.integrate_windows(
            energy,
            mu_plus,
            mu_minus,
            l3=(2.0, 5.0),
            l2=(4.0, 7.0),
            r_win=(9.0, 10.0),
        )
        self.assertAlmostEqual(out["p"], 3.0, places=6)
        self.assertAlmostEqual(out["q"], 5.0, places=6)
        self.assertNotAlmostEqual(out["q"], 6.0, places=6)


class TestMoments(unittest.TestCase):
    def test_known_values(self):
        p, q, r, nh = 1.0, 3.0, 10.0, 2.0
        out = sr.moments(p, q, r, nh)
        self.assertAlmostEqual(out["m_orb"], -(4.0 / 3.0) * nh * q / r)
        self.assertAlmostEqual(out["m_spin_plus_dipole"], nh * (6.0 * p - 4.0 * q) / r)

    def test_r_near_zero_raises(self):
        with self.assertRaises(ValueError):
            sr.moments(1.0, 2.0, 0.0, 1.0)

    def test_invalid_nh_raises(self):
        with self.assertRaises(ValueError):
            sr.moments(1.0, 2.0, 10.0, 0.0)
        with self.assertRaises(ValueError):
            sr.moments(1.0, 2.0, 10.0, -1.0)


class TestEnsembleSumrule(unittest.TestCase):
    def test_window_jitter_std_positive(self):
        energy = np.linspace(700.0, 740.0, 41)
        mu_plus = 0.01 * energy + 5.0 + 0.5 * np.exp(-((energy - 710.0) ** 2) / 4.0)
        mu_minus = 0.01 * energy + 5.0 - 0.3 * np.exp(-((energy - 710.0) ** 2) / 4.0)
        out = sr.ensemble_sumrule(
            energy,
            mu_plus,
            mu_minus,
            l3=(708.0, 712.0),
            l2=(715.0, 720.0),
            r_win=(700.0, 705.0),
            nh=1.0,
            window_delta=1.0,
            window_n=31,
            seed=0,
        )
        self.assertGreater(out["n_valid"], 1)
        self.assertGreater(out["p_std"], 0.0)
        self.assertGreater(out["m_orb_std"], 0.0)


class TestPickSourceKind(unittest.TestCase):
    def test_prefers_bg_pair(self):
        nodes = ["raw", "processed/CP_bg", "processed/CM_bg", "processed/CP"]
        self.assertEqual(sr.pick_source_kind(nodes, ("CP", "CM")), "bg")

    def test_falls_back_to_separated(self):
        nodes = ["raw", "processed/CP", "processed/CM"]
        self.assertEqual(sr.pick_source_kind(nodes, ("CP", "CM")), "separated")

    def test_falls_back_to_paired(self):
        nodes = ["raw", "processed"]
        self.assertEqual(sr.pick_source_kind(nodes, ("CP", "CM")), "paired")

    def test_no_pair_raises(self):
        with self.assertRaises(ValueError):
            sr.pick_source_kind(["raw"], ("CP", "CM"))


if __name__ == "__main__":
    unittest.main()
