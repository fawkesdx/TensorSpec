import unittest
import numpy as np
from tensorspec.core.data_models import TensorData
from tensorspec.core.data_tree import DataTreeBuilder
from tensorspec.core.workspace import WorkspaceManager
from tensorspec.core import peem_engine as eng


def _raw(pols):
    n = len(pols)
    frames = np.stack([np.full((2, 2), i + 1, dtype=float) for i in range(n)], axis=0)
    return TensorData(
        value=frames,
        axes=[np.arange(n), np.arange(2), np.arange(2)],
        labels=["frame", "y", "x"],
        units=["", "px", "px"],
        data_type="Experimental PEEM",
        metadata={"pol": list(pols), "frame_names": [f"f{i}" for i in range(n)]},
    )


class TestProcessedChildren(unittest.TestCase):
    def test_write_child_keeps_paired_parent(self):
        raw = _raw(["CP", "CM"])
        tree = DataTreeBuilder.build_from_tensor("t", raw)
        paired = eng.pair_stack(raw, "CP_CM")
        tree = DataTreeBuilder.write_processed(tree, paired)
        channels = eng.separate_pairs(paired)
        tree = DataTreeBuilder.write_processed_child(tree, "CP", channels["CP"])
        tree = DataTreeBuilder.write_processed_child(tree, "CM", channels["CM"])

        parent = tree["processed"].to_dataset()
        self.assertIn("data", parent)
        self.assertEqual(parent["data"].ndim, 4)
        self.assertEqual(set(DataTreeBuilder.list_processed_children(tree)), {"CP", "CM"})
        self.assertEqual(tree["processed/CP"].ds["data"].shape, (1, 2, 2))

    def test_workspace_pull_nested(self):
        ws = WorkspaceManager()
        raw = _raw(["CP", "CM", "CP", "CM"])
        ws.push_tensor_data("peem", raw)
        paired = eng.pair_stack(raw, "CP_CM")
        ws.write_processed_data("peem", paired)
        for tag, td in eng.separate_pairs(paired).items():
            self.assertTrue(ws.write_processed_child_data("peem", tag, td))
        child = ws.pull_tensor_data("peem", "processed/CP")
        self.assertIsNotNone(child)
        self.assertEqual(child.value.shape, (2, 2, 2))
        parent = ws.pull_tensor_data("peem", "processed")
        self.assertEqual(parent.value.ndim, 4)
        self.assertEqual(ws.list_processed_children("peem"), ["CM", "CP"])  # sorted

    def test_write_processed_wipes_children(self):
        ws = WorkspaceManager()
        raw = _raw(["CP", "CM"])
        ws.push_tensor_data("peem", raw)
        paired = eng.pair_stack(raw, "CP_CM")
        ws.write_processed_data("peem", paired)
        ws.write_processed_child_data("peem", "CP", eng.separate_pairs(paired)["CP"])
        ws.write_processed_data("peem", paired)
        self.assertEqual(ws.list_processed_children("peem"), [])
