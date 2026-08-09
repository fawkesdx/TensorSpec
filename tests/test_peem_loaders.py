import tempfile
import unittest
from pathlib import Path

import numpy as np

from tensorspec.core.io import peem_loaders as pl


def _write_multipage_tif(path: Path, stacks: list[np.ndarray]) -> None:
    import tifffile
    tifffile.imwrite(path, np.stack(stacks, axis=0))


def _write_unequal_multipage_tif(path: Path, pages: list[np.ndarray]) -> None:
    import tifffile
    with tifffile.TiffWriter(path) as writer:
        for page in pages:
            writer.write(page)


class TestPeemLoaders(unittest.TestCase):
    def test_infer_pol(self):
        self.assertEqual(pl.infer_pol_from_name("sample_CP_001.tif"), "CP")
        self.assertEqual(pl.infer_pol_from_name("x_cm_y.tif"), "CM")
        self.assertEqual(pl.infer_pol_from_name("plain.tif"), "unknown")

    def test_load_tif_stack_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = np.arange(6, dtype=np.uint16).reshape(2, 3)
            b = (a + 10).astype(np.uint16)
            p = root / "stack.tif"
            _write_multipage_tif(p, [a, b])
            td = pl.load_tif_stack(p)
            self.assertEqual(td.value.shape, (2, 2, 3))
            self.assertEqual(td.labels, ["frame", "y", "x"])
            self.assertEqual(td.data_type, "Experimental PEEM")
            self.assertEqual(td.metadata["loader"], "tif_stack")

    def test_unequal_multipage_tif_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "bad_stack.tif"
            _write_unequal_multipage_tif(
                p,
                [
                    np.ones((2, 2), dtype=np.uint16),
                    np.ones((3, 2), dtype=np.uint16),
                ],
            )
            with self.assertRaises(ValueError):
                pl.load_tif_stack(p)

    def test_load_tif_sequence_csv_auto_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, val in enumerate([1, 2]):
                import tifffile
                tifffile.imwrite(root / f"f{i}_CP.tif", np.full((2, 2), val, dtype=np.uint16))
            (root / "run.csv").write_text("frame,I0\n0,1.5\n1,1.7\n")
            found = pl.find_beamline_csv(root)
            self.assertEqual(len(found), 1)
            td = pl.load_tif_sequence(root)
            self.assertEqual(td.value.shape, (2, 2, 2))
            self.assertEqual(td.metadata["pol"], ["CP", "CP"])
            self.assertTrue(td.metadata.get("csv_attached"))
            self.assertIsNotNone(td.metadata.get("I0"))

    def test_load_tif_stack_csv_auto_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = np.arange(4, dtype=np.uint16).reshape(2, 2)
            p = root / "stack.tif"
            _write_multipage_tif(p, [a])
            (root / "beam.csv").write_text("frame,I0\n0,2.0\n")
            td = pl.load_tif_stack(p)
            self.assertTrue(td.metadata.get("csv_attached"))
            self.assertEqual(td.metadata.get("I0"), 2.0)

    def test_csv_ambiguity_no_auto_attach(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            import tifffile
            tifffile.imwrite(root / "a.tif", np.ones((2, 2), dtype=np.uint16))
            (root / "one.csv").write_text("frame,I0\n0,1.0\n")
            (root / "two.csv").write_text("frame,I0\n0,2.0\n")
            td = pl.load_tif_sequence(root)
            self.assertFalse(td.metadata.get("csv_attached", True))
            self.assertIsNone(td.metadata.get("I0"))

    def test_csv_preferred_stem_unique_attach(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "myrun"
            root.mkdir()
            import tifffile
            tifffile.imwrite(root / "a.tif", np.ones((2, 2), dtype=np.uint16))
            (root / "myrun.csv").write_text("frame,I0\n0,3.3\n")
            (root / "other.csv").write_text("frame,I0\n0,9.9\n")
            td = pl.load_tif_sequence(root)
            self.assertTrue(td.metadata.get("csv_attached"))
            self.assertEqual(td.metadata.get("I0"), 3.3)

    def test_explicit_csv_path_overrides_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            import tifffile
            tifffile.imwrite(root / "a.tif", np.ones((2, 2), dtype=np.uint16))
            one = root / "one.csv"
            two = root / "two.csv"
            one.write_text("frame,I0\n0,1.0\n")
            two.write_text("frame,I0\n0,2.0\n")
            td = pl.load_tif_sequence(root, csv_path=two)
            self.assertTrue(td.metadata.get("csv_attached"))
            self.assertEqual(td.metadata.get("I0"), 2.0)

    def test_load_without_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            import tifffile
            tifffile.imwrite(root / "a.tif", np.ones((2, 2), dtype=np.uint16))
            td = pl.load_tif_sequence(root)
            self.assertFalse(td.metadata.get("csv_attached", True))
            self.assertIsNone(td.metadata.get("I0"))


if __name__ == "__main__":
    unittest.main()
