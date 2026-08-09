"""Unit tests for CrystalEngine.compute_coordination_polyhedra."""
import unittest

import numpy as np

from tensorspec.core.crystallography import CrystalEngine


def _bonds_from_pairs(pairs):
    """Build aligned bonds_i, bonds_j from (i, j) tuples."""
    if not pairs:
        return np.array([], dtype=int), np.array([], dtype=int)
    arr = np.array(pairs, dtype=int)
    return arr[:, 0], arr[:, 1]


class TestCoordinationPolyhedra(unittest.TestCase):
    def test_tetrahedral_center_yields_hull(self):
        """Center with 4 non-coplanar neighbors → one polyhedron with faces."""
        coords = np.array([
            [0.0, 0.0, 0.0],   # 0: center
            [1.0, 1.0, 1.0],   # 1
            [1.0, -1.0, -1.0], # 2
            [-1.0, 1.0, -1.0], # 3
            [-1.0, -1.0, 1.0], # 4
        ])
        bonds_i, bonds_j = _bonds_from_pairs([
            (0, 1), (0, 2), (0, 3), (0, 4),
        ])

        polyhedra = CrystalEngine.compute_coordination_polyhedra(coords, bonds_i, bonds_j)

        self.assertEqual(len(polyhedra), 1)
        p = polyhedra[0]
        self.assertEqual(p["center"], 0)
        self.assertGreaterEqual(len(p["vertices"]), 4)
        self.assertGreater(len(p["simplices"]), 0)
        self.assertEqual(len(p["vertices"]), len(p["vertex_atom_indices"]))
        for idx in p["vertex_atom_indices"]:
            self.assertIn(idx, {1, 2, 3, 4})
        for simplex in p["simplices"]:
            self.assertEqual(len(simplex), 3)
            for vi in simplex:
                self.assertGreaterEqual(vi, 0)
                self.assertLess(vi, len(p["vertices"]))

    def test_two_neighbors_skipped(self):
        """Atom with only 2 bonded neighbors → no polyhedron for that center."""
        coords = np.array([
            [0.0, 0.0, 0.0],  # 0: sparse center
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 0.0],  # 3: tetrahedral center (duplicate origin ok for bonds)
            [1.0, 1.0, 1.0],  # 4
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ])
        # Offset tetrahedral cluster so coords are distinct
        coords[3] = [5.0, 5.0, 5.0]
        coords[4] = [6.0, 6.0, 6.0]
        coords[5] = [6.0, 4.0, 4.0]
        coords[6] = [4.0, 6.0, 4.0]
        coords[7] = [4.0, 4.0, 6.0]

        bonds_i, bonds_j = _bonds_from_pairs([
            (0, 1), (0, 2),           # sparse: 2 neighbors only
            (3, 4), (3, 5), (3, 6), (3, 7),  # full tetrahedron
        ])

        polyhedra = CrystalEngine.compute_coordination_polyhedra(coords, bonds_i, bonds_j)

        centers = {p["center"] for p in polyhedra}
        self.assertNotIn(0, centers)
        self.assertIn(3, centers)

    def test_three_neighbors_skipped(self):
        """Three neighbors insufficient for 3D ConvexHull → skip."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        bonds_i, bonds_j = _bonds_from_pairs([(0, 1), (0, 2), (0, 3)])

        polyhedra = CrystalEngine.compute_coordination_polyhedra(coords, bonds_i, bonds_j)
        self.assertEqual(polyhedra, [])


if __name__ == "__main__":
    unittest.main()
