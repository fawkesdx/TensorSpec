import tempfile
import unittest
from pathlib import Path

import numpy as np

from tensorspec.core.io.xas_loaders import load_xas_pair, load_xas_spectrum


class TestXasLoaders(unittest.TestCase):
    def test_single_column_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("energy,intensity\n")
            f.write("770,0.1\n770.5,0.5\n771,1.0\n")
            path = f.name
        try:
            td = load_xas_spectrum(path)
            self.assertEqual(td.labels, ["energy"])
            self.assertEqual(td.value.shape, (3,))
            self.assertEqual(td.data_type, "Experimental XAS")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_paired_pol_columns(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("E,CP,CM\n")
            f.write("770,1.0,0.9\n771,1.2,1.0\n")
            path = f.name
        try:
            td = load_xas_spectrum(path)
            self.assertEqual(td.labels, ["channel", "energy"])
            self.assertEqual(td.value.shape, (2, 2))
            self.assertEqual(td.metadata["channel_tags"], ["CP", "CM"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_pair_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            plus = Path(tmp) / "cp.csv"
            minus = Path(tmp) / "cm.csv"
            plus.write_text("energy,mu\n770,1.0\n771,1.1\n")
            minus.write_text("energy,mu\n770,0.95\n771,1.05\n")
            td = load_xas_pair(plus, minus)
            self.assertEqual(td.value.shape, (2, 2))
            np.testing.assert_allclose(td.value[0], [1.0, 1.1])
            np.testing.assert_allclose(td.value[1], [0.95, 1.05])


if __name__ == "__main__":
    unittest.main()
