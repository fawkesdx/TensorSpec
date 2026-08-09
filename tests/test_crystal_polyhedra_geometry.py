"""Schema defaults and geometry polyhedra flag for crystal API."""
import unittest

from pymatgen.core import Lattice, Structure

from tensorspec.web.server.routers import crystal as crystal_router
from tensorspec.web.server.schemas import (
    CrystalGeometry,
    GeometryRequest,
    Polyhedron,
    RelaxRequest,
    StackRequest,
)


def _diamond_si() -> Structure:
    lattice = Lattice.cubic(5.43)
    frac = [
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
    ]
    return Structure(lattice, ["Si"] * 8, frac)


class TestPolyhedraSchemaDefaults(unittest.TestCase):
    def test_geometry_request_show_polyhedra_default_false(self):
        req = GeometryRequest()
        self.assertFalse(req.show_polyhedra)
        self.assertTrue(req.show_bonds)

    def test_stack_request_show_polyhedra_default_false(self):
        req = StackRequest(layers=[{"name": "si"}])
        self.assertFalse(req.show_polyhedra)

    def test_relax_request_show_polyhedra_default_false(self):
        req = RelaxRequest()
        self.assertFalse(req.show_polyhedra)

    def test_crystal_geometry_polyhedra_default_empty(self):
        self.assertEqual(CrystalGeometry.model_fields["polyhedra"].default, [])

    def test_polyhedron_model_fields(self):
        p = Polyhedron(
            center=0,
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            simplices=[[0, 1, 2]],
            vertex_atom_indices=[1, 2],
        )
        self.assertEqual(p.center, 0)
        self.assertEqual(len(p.vertex_atom_indices), 2)


class TestPolyhedraGeometry(unittest.TestCase):
    def test_show_polyhedra_false_yields_empty_list(self):
        s = _diamond_si()
        geo = crystal_router._geometry_from_structure(
            "si", s, show_bonds=True, show_polyhedra=False
        )
        self.assertEqual(geo.polyhedra, [])

    def test_show_polyhedra_true_includes_hulls_without_bonds_in_response(self):
        s = _diamond_si()
        geo = crystal_router._geometry_from_structure(
            "si", s, show_bonds=False, show_polyhedra=True
        )
        self.assertEqual(geo.bonds, [])
        self.assertGreater(len(geo.polyhedra), 0)
        for poly in geo.polyhedra:
            self.assertIsInstance(poly.center, int)
            self.assertGreaterEqual(len(poly.vertices), 4)
            self.assertGreater(len(poly.simplices), 0)
            self.assertEqual(len(poly.vertices), len(poly.vertex_atom_indices))

    def test_show_bonds_and_polyhedra_both_populated(self):
        s = _diamond_si()
        geo = crystal_router._geometry_from_structure(
            "si", s, show_bonds=True, show_polyhedra=True
        )
        self.assertGreater(len(geo.bonds), 0)
        self.assertGreater(len(geo.polyhedra), 0)


if __name__ == "__main__":
    unittest.main()
