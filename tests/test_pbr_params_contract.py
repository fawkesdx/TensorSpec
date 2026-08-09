"""Contract: pbr_params.js exports shiny/matte numbers from Track C PBR spec."""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PBR_JS = REPO / "tensorspec/web/static/js/viewers/pbr_params.js"


def _read():
    return PBR_JS.read_text(encoding="utf-8")


class TestPbrParamsContract(unittest.TestCase):
    def test_file_exists(self):
        self.assertTrue(PBR_JS.is_file(), msg=str(PBR_JS))

    def test_exports_function(self):
        text = _read()
        self.assertIn("export function pbrMaterialParams", text)

    def test_shiny_numbers(self):
        text = _read()
        self.assertRegex(text, r"metalness:\s*0\.85")
        self.assertRegex(text, r"roughness:\s*0\.2\b")

    def test_matte_atom_roughness(self):
        text = _read()
        self.assertRegex(text, r"roughness:\s*0\.45")

    def test_matte_bond_roughness(self):
        text = _read()
        self.assertRegex(text, r"roughness:\s*0\.5\b")


if __name__ == "__main__":
    unittest.main()
