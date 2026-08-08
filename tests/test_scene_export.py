"""Crystal scene export → SceneExporter script markers."""
import tempfile
import unittest
from pathlib import Path

from pymatgen.core import Lattice, Structure

from tensorspec.core.io.exporters import SceneExporter
from tensorspec.web.server.routers import crystal as crystal_router
from tensorspec.web.server.schemas import SceneExportRequest


class TestSceneExportTuples(unittest.TestCase):
    def setUp(self):
        self.structure = Structure(Lattice.cubic(4.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        self.geo = crystal_router._geometry_from_structure("si", self.structure, show_bonds=True)

    def test_request_requires_at_least_one_include(self):
        with self.assertRaises(Exception):
            SceneExportRequest(include_atoms=False, include_cell=False, include_bz=False)

    def test_atoms_tuples_nonempty(self):
        atoms, bonds, lattice, bz = crystal_router._scene_export_parts(
            self.geo, include_atoms=True, include_cell=False, include_bz=False, bz_geometry=None
        )
        self.assertGreater(len(atoms), 0)
        self.assertEqual(lattice, [])
        self.assertIsNone(bz)

    def test_blender_script_contains_atom(self):
        atoms, bonds, lattice, bz = crystal_router._scene_export_parts(
            self.geo, include_atoms=True, include_cell=True, include_bz=False, bz_geometry=None
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.py"
            SceneExporter.export_blender(str(path), atoms, bonds, lattice, bz)
            text = path.read_text()
            self.assertIn("Atom", text)
            self.assertIn("TensorSpec_Crystal", text)


if __name__ == "__main__":
    unittest.main()
