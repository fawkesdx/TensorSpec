import unittest
import numpy as np
from tensorspec.core.data_models import TensorData
from tensorspec.core import peem_engine as eng


def _raw(pols, frames=None):
    n = len(pols)
    if frames is None:
        frames = np.stack([np.full((2, 2), i + 1, dtype=float) for i in range(n)], axis=0)
    return TensorData(
        value=frames,
        axes=[np.arange(n), np.arange(2), np.arange(2)],
        labels=["frame", "y", "x"],
        units=["", "px", "px"],
        data_type="Experimental PEEM",
        metadata={
            "pol": list(pols),
            "frame_names": [f"f{i}_{p}" for i, p in enumerate(pols)],
            "csv_attached": True,
            "I0": [1.0] * n,
            "source": "test",
            "loader": "tif_sequence",
        },
    )


class TestPairStack(unittest.TestCase):
    def test_cp_cm_happy(self):
        out = eng.pair_stack(_raw(["CP", "CM", "CP", "CM"]), "CP_CM")
        self.assertEqual(out.value.shape, (2, 2, 2, 2))
        self.assertEqual(out.labels, ["pair", "channel", "y", "x"])
        self.assertEqual(out.metadata["channel_tags"], ["CP", "CM"])
        self.assertEqual(out.metadata["unpaired"], [])
        np.testing.assert_array_equal(out.value[0, 0], np.full((2, 2), 1.0))
        np.testing.assert_array_equal(out.value[0, 1], np.full((2, 2), 2.0))

    def test_auto_cp_cm(self):
        out = eng.pair_stack(_raw(["CM", "CP"]), "auto")
        self.assertEqual(out.metadata["pair_mode"], "CP_CM")
        # file order: first CM with first CP
        self.assertEqual(out.value.shape[0], 1)

    def test_unequal_leftovers(self):
        out = eng.pair_stack(_raw(["CP", "CP", "CM"]), "CP_CM")
        self.assertEqual(out.value.shape[0], 1)
        self.assertEqual(len(out.metadata["unpaired"]), 1)
        self.assertEqual(out.metadata["unpaired"][0]["pol"], "CP")

    def test_mixed_auto_fails(self):
        with self.assertRaises(ValueError):
            eng.pair_stack(_raw(["CP", "CM", "LH"]), "auto")

    def test_zero_pairs_fails(self):
        with self.assertRaises(ValueError):
            eng.pair_stack(_raw(["CP", "CP"]), "CP_CM")

    def test_invalid_mode_fails(self):
        with self.assertRaises(ValueError):
            eng.pair_stack(_raw(["CP", "CM"]), "bogus")


class TestSeparatePairs(unittest.TestCase):
    def test_cp_cm_split(self):
        paired = eng.pair_stack(_raw(["CP", "CM", "CP", "CM"]), "CP_CM")
        out = eng.separate_pairs(paired)
        self.assertEqual(set(out), {"CP", "CM"})
        self.assertEqual(out["CP"].value.shape, (2, 2, 2))
        self.assertEqual(out["CP"].labels, ["frame", "y", "x"])
        self.assertEqual(out["CP"].data_type, "Experimental PEEM (CP)")
        self.assertEqual(out["CP"].metadata["channel_tag"], "CP")
        self.assertEqual(out["CP"].metadata["separated_from"], "paired")
        self.assertEqual(out["CP"].metadata["pair_mode"], "CP_CM")
        self.assertTrue(out["CP"].metadata["csv_attached"])
        np.testing.assert_array_equal(out["CP"].value[0], paired.value[0, 0])
        np.testing.assert_array_equal(out["CM"].value[0], paired.value[0, 1])

    def test_lh_lv_tags(self):
        paired = eng.pair_stack(_raw(["LH", "LV"]), "LH_LV")
        out = eng.separate_pairs(paired)
        self.assertEqual(set(out), {"LH", "LV"})

    def test_rejects_raw_3d(self):
        with self.assertRaises(ValueError):
            eng.separate_pairs(_raw(["CP", "CM"]))

    def test_rejects_bad_channel_tags(self):
        paired = eng.pair_stack(_raw(["CP", "CM"]), "CP_CM")
        paired.metadata["channel_tags"] = ["CP"]
        with self.assertRaises(ValueError):
            eng.separate_pairs(paired)
