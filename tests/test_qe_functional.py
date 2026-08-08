"""QE functional emits input_dft in scf.in."""
import tempfile
import unittest
from pathlib import Path

from pymatgen.core import Lattice, Structure

from tensorspec.core.dft.qe_generator import QEInputGenerator
from tensorspec.web.server.schemas import QERequest


class TestQEFunctional(unittest.TestCase):
    def setUp(self):
        self.structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])

    def _gen_with_stub_pseudo(self, tmp: str) -> QEInputGenerator:
        pseudo_dir = Path(tmp) / "pseudo_src"
        pseudo_dir.mkdir()
        # Pattern in _generate_atomic_species: ^{symbol}[._].*\.upf$
        (pseudo_dir / "Si.pbe.upf").write_text("stub\n")
        return QEInputGenerator(self.structure, pseudo_dir=str(pseudo_dir))

    def test_default_pbe(self):
        self.assertEqual(QERequest().functional, "PBE")

    def test_scf_contains_input_dft_lda(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = self._gen_with_stub_pseudo(tmp)
            out = Path(tmp) / "run"
            path = gen.write_scf_input(str(out), functional="LDA")
            text = Path(path).read_text()
            self.assertIn("input_dft = 'lda'", text)

    def test_scf_default_pbe_keyword(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = self._gen_with_stub_pseudo(tmp)
            out = Path(tmp) / "run"
            path = gen.write_scf_input(str(out), functional="PBE")
            self.assertIn("input_dft = 'pbe'", Path(path).read_text())


if __name__ == "__main__":
    unittest.main()
