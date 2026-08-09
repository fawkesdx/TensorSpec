import unittest
from pymatgen.core import Lattice, Structure

from tensorspec.web.server.geometry_filter import (
    filter_geometry_atoms_bonds,
    filter_structure_by_omit,
    normalize_omit_indices,
)


class _Atom:
    def __init__(self, i):
        self.i = i


class _Bond:
    def __init__(self, i, j):
        self.i = i
        self.j = j


class TestGeometryFilter(unittest.TestCase):
    def test_normalize_drops_oor(self):
        self.assertEqual(normalize_omit_indices([-1, 0, 2, 2, 99], 3), {0, 2})

    def test_filter_bonds_remap(self):
        atoms = [_Atom(0), _Atom(1), _Atom(2)]
        bonds = [_Bond(0, 1), _Bond(1, 2)]
        a2, b2 = filter_geometry_atoms_bonds(atoms, bonds, {1})
        self.assertEqual(len(a2), 2)
        self.assertEqual([(b.i, b.j) for b in b2], [])

    def test_filter_keeps_remote_bond(self):
        atoms = [_Atom(0), _Atom(1), _Atom(2)]
        bonds = [_Bond(0, 2)]
        a2, b2 = filter_geometry_atoms_bonds(atoms, bonds, {1})
        self.assertEqual(len(a2), 2)
        self.assertEqual([(b.i, b.j) for b in b2], [(0, 1)])

    def test_filter_structure(self):
        s = Structure(Lattice.cubic(4), ["Si", "Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]])
        out = filter_structure_by_omit(s, {1})
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
