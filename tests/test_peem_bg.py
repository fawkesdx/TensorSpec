import unittest

import numpy as np

from tensorspec.core.peem import bg


class TestExtractSpectrum(unittest.TestCase):
    def test_picture_wide_mean(self):
        stack = np.array(
            [
                [[1.0, 3.0], [5.0, 7.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ]
        )
        np.testing.assert_allclose(bg.extract_spectrum(stack, None), [4.0, 5.0])

    def test_roi_mask_mean(self):
        stack = np.ones((3, 2, 2), dtype=float)
        stack[0, 0, 0] = 10.0
        stack[1, 1, 1] = 6.0
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        np.testing.assert_allclose(bg.extract_spectrum(stack, mask), [10.0, 1.0, 1.0])


class TestFitLinearPreedge(unittest.TestCase):
    def test_known_line(self):
        energy = np.array([700.0, 710.0, 720.0, 730.0, 740.0])
        spectrum = 2.0 * energy + 3.0
        out = bg.fit_linear_preedge(energy, spectrum, 700.0, 720.0)
        self.assertAlmostEqual(out["slope"], 2.0)
        self.assertAlmostEqual(out["intercept"], 3.0)
        np.testing.assert_allclose(out["bg"], spectrum)

    def test_invalid_window_raises(self):
        energy = np.array([700.0, 710.0, 720.0])
        spectrum = energy.copy()
        with self.assertRaises(ValueError):
            bg.fit_linear_preedge(energy, spectrum, 705.0, 705.0)


class TestFitTwoStepPrePost(unittest.TestCase):
    def _expected_two_step_bg(self, energy, pre_slope, pre_intercept, post_slope, post_intercept, pre_e1, post_e0):
        y_pre = pre_slope * energy + pre_intercept
        y_post = post_slope * energy + post_intercept
        y_at_pre_e1 = pre_slope * pre_e1 + pre_intercept
        y_at_post_e0 = post_slope * post_e0 + post_intercept
        bg = np.empty_like(energy, dtype=float)
        for i, e in enumerate(energy):
            if e <= pre_e1:
                bg[i] = y_pre[i]
            elif e >= post_e0:
                bg[i] = y_post[i]
            else:
                t = (e - pre_e1) / (post_e0 - pre_e1)
                bg[i] = (1.0 - t) * y_at_pre_e1 + t * y_at_post_e0
        return bg

    def test_known_geometry(self):
        energy = np.linspace(700.0, 750.0, 11)
        pre_slope, pre_intercept = 2.0, 3.0
        post_slope, post_intercept = 0.5, 50.0
        pre_e1, post_e0 = 710.0, 730.0
        spectrum = self._expected_two_step_bg(
            energy, pre_slope, pre_intercept, post_slope, post_intercept, pre_e1, post_e0
        )
        out = bg.fit_two_step_pre_post(energy, spectrum, 700.0, pre_e1, post_e0, 740.0)
        self.assertAlmostEqual(out["pre_slope"], pre_slope)
        self.assertAlmostEqual(out["pre_intercept"], pre_intercept)
        self.assertAlmostEqual(out["post_slope"], post_slope)
        self.assertAlmostEqual(out["post_intercept"], post_intercept)
        np.testing.assert_allclose(out["bg"], spectrum)

    def test_invalid_window_order_raises(self):
        energy = np.linspace(700.0, 740.0, 9)
        spectrum = energy.copy()
        with self.assertRaises(ValueError):
            bg.fit_two_step_pre_post(energy, spectrum, 700.0, 730.0, 720.0, 740.0)

    def test_too_few_points_raises(self):
        energy = np.array([700.0, 710.0, 720.0, 730.0, 740.0])
        spectrum = energy.copy()
        with self.assertRaises(ValueError):
            bg.fit_two_step_pre_post(energy, spectrum, 700.0, 700.0, 730.0, 740.0)


class TestFitBackground(unittest.TestCase):
    def test_linear_dispatch(self):
        energy = np.array([700.0, 710.0, 720.0, 730.0, 740.0])
        spectrum = 2.0 * energy + 3.0
        out = bg.fit_background("linear", energy, spectrum, e0=700.0, e1=720.0)
        self.assertEqual(out["method"], "linear")
        self.assertAlmostEqual(out["slope"], 2.0)
        self.assertAlmostEqual(out["intercept"], 3.0)

    def test_two_step_dispatch(self):
        energy = np.linspace(700.0, 750.0, 11)
        spectrum = 0.001 * energy**2
        out = bg.fit_background(
            "two_step", energy, spectrum, e0=700.0, e1=710.0, post_e0=730.0, post_e1=740.0
        )
        self.assertEqual(out["method"], "two_step")
        self.assertIn("pre_slope", out)
        self.assertIn("post_slope", out)
        self.assertEqual(out["bg"].shape, energy.shape)


class TestEnsembleBackground(unittest.TestCase):
    def test_two_step_std_positive(self):
        energy = np.linspace(700.0, 750.0, 11)
        spectrum = 0.001 * energy**2 + 0.5 * energy + 10.0
        out = bg.ensemble_background(
            "two_step",
            energy,
            spectrum,
            e0=700.0,
            e1=710.0,
            post_e0=730.0,
            post_e1=740.0,
            delta=2.0,
            n=31,
            seed=0,
        )
        self.assertGreater(out["n_valid"], 1)
        self.assertTrue(np.any(out["bg_std"] > 0))

    def test_linear_matches_preedge(self):
        energy = np.linspace(700.0, 740.0, 9)
        spectrum = 0.001 * energy**2 + 0.5 * energy + 10.0
        preedge = bg.ensemble_preedge(
            energy, spectrum, 702.0, 712.0, delta=2.0, n=31, seed=0
        )
        dispatched = bg.ensemble_background(
            "linear", energy, spectrum, e0=702.0, e1=712.0, delta=2.0, n=31, seed=0
        )
        for key in ("bg_mean", "bg_std", "subtracted_mean", "subtracted_std", "n_valid"):
            np.testing.assert_allclose(dispatched[key], preedge[key])


class TestEnsemblePreedge(unittest.TestCase):
    def test_std_positive_with_jitter(self):
        energy = np.linspace(700.0, 740.0, 9)
        spectrum = 0.001 * energy**2 + 0.5 * energy + 10.0
        out = bg.ensemble_preedge(
            energy, spectrum, 702.0, 712.0, delta=2.0, n=31, seed=0
        )
        self.assertGreater(out["n_valid"], 1)
        self.assertTrue(np.any(out["bg_std"] > 0))


class TestApplyBgToStack(unittest.TestCase):
    def test_subtracts_constant_ramp(self):
        n, ny, nx = 5, 2, 2
        ramp = np.arange(n, dtype=float)
        stack = ramp[:, None, None] * np.ones((n, ny, nx))
        bg_curve = ramp.copy()
        out = bg.apply_bg_to_stack(stack, bg_curve)
        np.testing.assert_allclose(out, 0.0)

    def test_rejects_2d_stack(self):
        stack = np.ones((4, 4), dtype=float)
        bg_curve = np.ones(4, dtype=float)
        with self.assertRaises(ValueError):
            bg.apply_bg_to_stack(stack, bg_curve)


class TestResolveEnergy(unittest.TestCase):
    def test_csv_alias_match(self):
        meta = {
            "beamline_table": {
                "series": {
                    "PhotonEnergy": [700.0, 710.0, 720.0],
                    "I0": [1.0, 1.0, 1.0],
                }
            }
        }
        energy, source = bg.resolve_energy(3, meta)
        np.testing.assert_allclose(energy, [700.0, 710.0, 720.0])
        self.assertEqual(source, "csv")

    def test_index_fallback(self):
        energy, source = bg.resolve_energy(4, {})
        np.testing.assert_allclose(energy, [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(source, "index")


class TestBgChildName(unittest.TestCase):
    def test_raw_and_processed(self):
        self.assertEqual(bg.bg_child_name("raw"), "bg")
        self.assertEqual(bg.bg_child_name("processed"), "bg")

    def test_separated_tag(self):
        self.assertEqual(bg.bg_child_name("processed/CP"), "CP_bg")

    def test_rejects_bg_output_nodes(self):
        for node in ("processed/bg", "processed/CP_bg"):
            with self.subTest(node=node):
                self.assertTrue(bg.is_bg_output_node(node))
                with self.assertRaises(ValueError):
                    bg.bg_child_name(node)


class TestIsBgOutputNode(unittest.TestCase):
    def test_non_bg_nodes(self):
        for node in ("raw", "processed", "processed/CP", "processed/O"):
            self.assertFalse(bg.is_bg_output_node(node))

    def test_bg_output_nodes(self):
        for node in ("processed/bg", "processed/CP_bg"):
            self.assertTrue(bg.is_bg_output_node(node))


if __name__ == "__main__":
    unittest.main()
