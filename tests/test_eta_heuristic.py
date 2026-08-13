"""Lock DFT/ARPES ETA heuristic numbers from the job-timer spec."""

from __future__ import annotations

import unittest

from tensorspec.core.jobs.eta_heuristic import (
    estimate_arpes_seconds,
    estimate_dft_seconds,
    format_elapsed,
    format_estimate,
)


class EtaHeuristicTests(unittest.TestCase):
    def test_vte2_soc_einstein_clamped(self):
        big = estimate_dft_seconds(
            "einstein_ssh", nbnd=324, kx=6, ky=6, kz=6, soc=True, ranks=20
        )
        self.assertGreaterEqual(big, 120)
        self.assertLessEqual(big, 12 * 3600)

        small = estimate_dft_seconds(
            "einstein_ssh", nbnd=324, kx=4, ky=4, kz=1, soc=False, ranks=20
        )
        self.assertGreater(big, small)

    def test_format_elapsed(self):
        self.assertEqual(format_elapsed(75), "01:15")
        self.assertEqual(format_elapsed(3661), "1:01:01")

    def test_format_estimate(self):
        self.assertIn("s", format_estimate(45))
        self.assertIn("min", format_estimate(180))

    def test_arpes_reference_grid(self):
        ref = estimate_arpes_seconds(48, 64, 64)
        self.assertEqual(ref, 180)


if __name__ == "__main__":
    unittest.main()
