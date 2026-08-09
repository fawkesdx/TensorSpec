"""Schema + export/CIF/push omit_atom_indices wiring."""
import unittest

from pymatgen.core import Lattice, Structure

from tensorspec.web.server.geometry_filter import filter_structure_by_omit
from tensorspec.web.server.routers import crystal as crystal_router
from tensorspec.web.server.schemas import (
    CrystalCifRequest,
    PushCrystalRequest,
    SceneExportRequest,
)


class TestOmitSchemas(unittest.TestCase):
    def test_scene_export_omit_default_empty(self):
        req = SceneExportRequest(include_atoms=True, include_cell=False, include_bz=False)
        self.assertEqual(req.omit_atom_indices, [])

    def test_cif_request_accepts_omit(self):
        req = CrystalCifRequest(omit_atom_indices=[0, 2], nx=2, ny=1, nz=1)
        self.assertEqual(req.omit_atom_indices, [0, 2])
        self.assertEqual(req.cell_count, 2)

    def test_push_request_accepts_knobs(self):
        req = PushCrystalRequest(
            store_as="filtered",
            omit_atom_indices=[1],
            nx=2,
            ny=2,
            nz=1,
            basis="primitive",
        )
        self.assertEqual(req.omit_atom_indices, [1])
        self.assertEqual(req.cell_count, 4)


class TestExportGeoFilter(unittest.TestCase):
    def setUp(self):
        self.structure = Structure(
            Lattice.cubic(4.0),
            ["Si", "Si", "Si"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        )
        self.geo = crystal_router._geometry_from_structure(
            "si", self.structure, show_bonds=True
        )

    def test_filter_geometry_drops_omitted_atom(self):
        filtered = crystal_router._filter_geometry_by_omit(self.geo, {1})
        self.assertEqual(filtered.n_atoms, 2)
        self.assertEqual(len(filtered.atoms), 2)
        self.assertTrue(all(b.i < 2 and b.j < 2 for b in filtered.bonds))

    def test_scene_export_parts_respects_filtered_geo(self):
        filtered = crystal_router._filter_geometry_by_omit(self.geo, {0})
        atoms, bonds, _, _ = crystal_router._scene_export_parts(
            filtered,
            include_atoms=True,
            include_cell=False,
            include_bz=False,
            bz_geometry=None,
        )
        self.assertEqual(len(atoms), 2)
        self.assertEqual(len(bonds), len(filtered.bonds))


class TestKnobbedStructure(unittest.TestCase):
    def test_default_knobs_fast_path(self):
        s = Structure(Lattice.cubic(4), ["Si"], [[0, 0, 0]])
        self.assertTrue(
            crystal_router._is_default_knobs(
                omit_atom_indices=[],
                nx=1,
                ny=1,
                nz=1,
                basis="conventional",
            )
        )

    def test_build_knobbed_structure_omit(self):
        s = Structure(
            Lattice.cubic(4),
            ["Si", "Si"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        out = crystal_router._build_knobbed_structure(
            s,
            basis="conventional",
            nx=1,
            ny=1,
            nz=1,
            omit_atom_indices=[1],
        )
        self.assertEqual(len(out), 1)

    def test_filter_structure_keeps_site_props(self):
        s = Structure(Lattice.cubic(4), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        s[0].properties["magmom"] = 1.0
        out = filter_structure_by_omit(s, {1})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].properties.get("magmom"), 1.0)


if __name__ == "__main__":
    unittest.main()
