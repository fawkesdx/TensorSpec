import unittest

from tensorspec.plotting.backends import crystal_figure as cf


class TestCrystalFigure(unittest.TestCase):
    def test_png_nonempty(self):
        atoms = [
            {"element": "Si", "position": [0.0, 0.0, 0.0], "radius": 1.1},
            {"element": "Si", "position": [1.0, 1.0, 1.0], "radius": 1.1},
        ]
        bonds = [(0, 1)]
        cell = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
        raw = cf.export_crystal_figure(
            atoms=atoms, bonds=bonds, cell=cell, fmt="png", title="Si"
        )
        self.assertIsInstance(raw, (bytes, bytearray))
        self.assertGreater(len(raw), 100)
        self.assertTrue(raw[:8] == b"\x89PNG\r\n\x1a\n")

    def test_omit_changes_output(self):
        atoms2 = [
            {"element": "C", "position": [0.0, 0.0, 0.0], "radius": 0.7},
            {"element": "O", "position": [1.5, 0.0, 0.0], "radius": 0.6},
        ]
        a = cf.export_crystal_figure(atoms=atoms2, bonds=[], cell=None, fmt="png")
        b = cf.export_crystal_figure(atoms=atoms2[:1], bonds=[], cell=None, fmt="png")
        self.assertNotEqual(a, b)

    def test_svg_header(self):
        atoms = [{"element": "H", "position": [0, 0, 0], "radius": 0.3}]
        raw = cf.export_crystal_figure(atoms=atoms, bonds=[], cell=None, fmt="svg")
        self.assertIn(b"<svg", raw.lower())

    def test_h_only_png_nonempty(self):
        atoms = [{"element": "H", "position": [0, 0, 0], "radius": 0.3}]
        raw = cf.export_crystal_figure(atoms=atoms, bonds=[], cell=None, fmt="png")
        empty = cf.export_crystal_figure(atoms=[], bonds=[], cell=None, fmt="png")
        self.assertGreater(len(raw), 100)
        self.assertTrue(raw[:8] == b"\x89PNG\r\n\x1a\n")
        self.assertNotEqual(raw, empty)


if __name__ == "__main__":
    unittest.main()
