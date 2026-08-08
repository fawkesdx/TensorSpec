"""Crystal geometry labels, basis transform, and load fmt detection."""
import unittest
from io import StringIO

from pymatgen.core import Lattice, Structure

from tensorspec.web.server.routers import crystal as crystal_router


class TestSiteLabel(unittest.TestCase):
    def test_label_falls_back_to_symbol(self):
        s = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
        element, label = crystal_router._site_element_and_label(s[0])
        self.assertEqual(element, "Si")
        self.assertEqual(label, "Si")

    def test_geometry_includes_labels(self):
        s = Structure(Lattice.cubic(4.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        # pymatgen assigns Si0, Si1-style labels on construction
        geo = crystal_router._geometry_from_structure("si", s, show_bonds=False)
        self.assertEqual(len(geo.atoms), 2)
        self.assertTrue(all(a.label for a in geo.atoms))
        self.assertEqual(geo.atoms[0].element, "Si")


class TestBasis(unittest.TestCase):
    def test_primitive_differs_from_conventional_fcc(self):
        # Conventional FCC cubic cell (4 sites) → primitive has 1 site
        lattice = Lattice.cubic(3.6)
        species = ["Cu"] * 4
        frac = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        conventional = Structure(lattice, species, frac)
        prim = crystal_router._apply_basis(conventional, "primitive")
        self.assertLess(len(prim), len(conventional))

    def test_conventional_is_identity_for_already_standard(self):
        s = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
        out = crystal_router._apply_basis(s, "conventional")
        self.assertEqual(len(out), len(s))


class TestLoadFmt(unittest.TestCase):
    def test_detect_fmt(self):
        self.assertEqual(crystal_router._detect_structure_fmt("x.cif"), "cif")
        self.assertEqual(crystal_router._detect_structure_fmt("POSCAR"), "poscar")
        self.assertEqual(crystal_router._detect_structure_fmt("foo.vasp"), "poscar")
        self.assertEqual(crystal_router._detect_structure_fmt("bar.POSCAR"), "poscar")


if __name__ == "__main__":
    unittest.main()
