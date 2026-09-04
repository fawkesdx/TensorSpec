"""Tests for Local vs Hybrid compute mode helpers."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from tensorspec.gui.services import compute_mode as cm


class _FakeCombo:
    def __init__(self, data):
        self._data = data

    def currentData(self):
        return self._data


class ComputeModeTests(unittest.TestCase):
    def test_local_mode(self):
        combo = _FakeCombo("local")
        self.assertTrue(cm.is_local_mode(combo))
        self.assertFalse(cm.is_hybrid_mode(combo))
        self.assertIsNone(cm.combo_cluster(combo))

    def test_hybrid_entry(self):
        cluster = {"name": "remote-cluster", "host": "gpu.example.edu", "user": "YOUR_USER"}
        combo = _FakeCombo(cm.hybrid_entry(cluster))
        self.assertTrue(cm.is_hybrid_mode(combo))
        self.assertEqual(cm.combo_cluster(combo)["host"], "gpu.example.edu")

    def test_legacy_cluster_dict(self):
        cluster = {"host": "hpc.example.edu", "user": "alice"}
        combo = _FakeCombo(cluster)
        self.assertTrue(cm.is_hybrid_mode(combo))
        self.assertEqual(cm.combo_cluster(combo), cluster)

    def test_hybrid_fast_diag_w90(self):
        cluster = {"host": "gpu.example.edu", "user": "YOUR_USER"}
        combo = _FakeCombo(cm.hybrid_entry(cluster))
        engine, device = cm.effective_band_diag(
            combo, "chinook", "cpu", auto_gpu=True, w90_loaded=True
        )
        self.assertEqual((engine, device), ("grizzly", "cuda"))

    def test_local_ignores_fast_path(self):
        combo = _FakeCombo("local")
        engine, device = cm.effective_band_diag(
            combo, "chinook", "cpu", auto_gpu=True, w90_loaded=True
        )
        self.assertEqual((engine, device), ("chinook", "cpu"))

    def test_prefs_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prefs.json")
            orig = cm.PREFS_FILE
            try:
                cm.PREFS_FILE = path
                cm.save_default_mode(cm.MODE_HYBRID)
                self.assertEqual(cm.load_default_mode(), cm.MODE_HYBRID)
            finally:
                cm.PREFS_FILE = orig


if __name__ == "__main__":
    unittest.main()
