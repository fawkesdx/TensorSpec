"""Shared cluster dropdown helpers for TensorSpec GUI."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

CLUSTERS_FILE = os.path.expanduser("~/.tensorspec_clusters.json")
LOCAL_TARGET = "local"


def load_clusters() -> List[Dict[str, Any]]:
    if not os.path.isfile(CLUSTERS_FILE):
        return []
    try:
        with open(CLUSTERS_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def cluster_display_name(cluster: Dict[str, Any]) -> str:
    return str(cluster.get("name") or cluster.get("host") or "remote cluster")


def populate_compute_target_combo(combo, *, include_local: bool = True) -> None:
    """Fill a QComboBox with Local + clusters from ~/.tensorspec_clusters.json."""
    combo.blockSignals(True)
    combo.clear()
    if include_local:
        combo.addItem("💻 Local Computation", LOCAL_TARGET)
    for cluster in load_clusters():
        combo.addItem(f"🚀 Remote: {cluster_display_name(cluster)}", cluster)
    combo.blockSignals(False)


def is_remote_target(combo) -> bool:
    data = combo.currentData()
    return isinstance(data, dict)


def selected_cluster(combo) -> Optional[Dict[str, Any]]:
    data = combo.currentData()
    if isinstance(data, dict):
        return data
    clusters = load_clusters()
    return clusters[0] if clusters else None


def find_cluster_by_name(name: str) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    key = name.strip().lower()
    for cluster in load_clusters():
        for field in ("name", "host", "user"):
            val = str(cluster.get(field, "")).lower()
            if val == key or key in val:
                return cluster
    return None
