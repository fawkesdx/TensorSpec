import tempfile
import unittest
from pathlib import Path

import numpy as np

from tensorspec.core.workspace import global_workspace
from tensorspec.gui.services.xas_service import XasService


class TestXasService(unittest.TestCase):
    def setUp(self):
        self.service = XasService()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.csv = Path(self._tmpdir.name) / "pair.csv"
        energy = np.linspace(770, 780, 50)
        cp = 0.2 + 0.8 * np.exp(-0.5 * ((energy - 775) / 1.0) ** 2)
        cm = cp * 0.97 + 0.01
        lines = ["energy,CP,CM\n"]
        for e, a, b in zip(energy, cp, cm):
            lines.append(f"{e:.4f},{a:.6f},{b:.6f}\n")
        self.csv.write_text("".join(lines))

    def tearDown(self):
        global_workspace._data.clear()
        self._tmpdir.cleanup()

    def test_bg_and_sumrule_on_paired_csv(self):
        meta = self.service.load_path(self.csv, name="xas_test")
        self.assertTrue(meta["paired"])
        e0, e1 = 770.0, 772.0
        bg = self.service.bg_preview(
            "xas_test", node="processed", channel=0, method="linear", e0=e0, e1=e1
        )
        self.assertEqual(bg["energy"].shape, bg["spectrum"].shape)
        self.assertEqual(bg["subtracted"].shape, bg["spectrum"].shape)

        sr = self.service.sumrule_preview(
            "xas_test",
            nh=1.0,
            l3_lo=770.0,
            l3_hi=776.0,
            l2_lo=776.0,
            l2_hi=779.0,
            r_lo=770.0,
            r_hi=780.0,
        )
        self.assertIn("m_orb", sr)
        self.assertEqual(sr["dichroism"].shape, sr["energy"].shape)


if __name__ == "__main__":
    unittest.main()
