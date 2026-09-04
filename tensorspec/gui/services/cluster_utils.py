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
    """Fill combo with Local only + Hybrid server entries (see compute_mode.py)."""
    from tensorspec.gui.services.compute_mode import populate_compute_mode_combo

    populate_compute_mode_combo(combo, include_local=include_local)


def is_remote_target(combo) -> bool:
    from tensorspec.gui.services.compute_mode import is_hybrid_mode

    return is_hybrid_mode(combo)


def selected_cluster(combo) -> Optional[Dict[str, Any]]:
    from tensorspec.gui.services.compute_mode import combo_cluster

    return combo_cluster(combo)


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
