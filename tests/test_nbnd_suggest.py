"""Tests for QE/Wannier nbnd suggestion from crystal sites."""

from __future__ import annotations

import unittest

from pymatgen.core import Lattice, Structure

from tensorspec.core.dft.nbnd_suggest import suggest_nbnd_base


class SuggestNbndBaseTests(unittest.TestCase):
    def test_graphene_two_carbon(self):
        # 2 × C → 2 × 4 = 8
        struct = Structure(
            Lattice.hexagonal(2.46, 20.0),
            ["C", "C"],
            [[0, 0, 0], [1 / 3, 2 / 3, 0]],
        )
        self.assertEqual(suggest_nbnd_base(struct), 8)

    def test_transition_metal_site(self):
        struct = Structure(Lattice.cubic(2.8), ["Fe"], [[0, 0, 0]])
        self.assertEqual(suggest_nbnd_base(struct), 9)

    def test_heavy_non_tm_z_gt_30(self):
        # Te Z=52 → +9 by Z>30 rule
        struct = Structure(Lattice.cubic(3.0), ["Te"], [[0, 0, 0]])
        self.assertEqual(suggest_nbnd_base(struct), 9)

    def test_vte2_like_counts(self):
        # V (TM +9) + 2×Te (Z>30 +9 each) = 27
        struct = Structure(
            Lattice.hexagonal(3.6, 20.0),
            ["V", "Te", "Te"],
            [[0, 0, 0], [1 / 3, 2 / 3, 0.1], [2 / 3, 1 / 3, 0.2]],
        )
        self.assertEqual(suggest_nbnd_base(struct), 27)


if __name__ == "__main__":
    unittest.main()
