"""Shared Local vs Hybrid compute mode for TensorSpec GUI suites."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from tensorspec.gui.services.cluster_utils import (
    CLUSTERS_FILE,
    LOCAL_TARGET,
    cluster_display_name,
    load_clusters,
)

MODE_LOCAL = "local"
MODE_HYBRID = "hybrid"
PREFS_FILE = os.path.expanduser("~/.tensorspec_compute_prefs.json")


def hybrid_entry(cluster: Dict[str, Any]) -> Dict[str, Any]:
    return {"mode": MODE_HYBRID, "cluster": cluster}


def load_default_mode() -> str:
    if not os.path.isfile(PREFS_FILE):
        return MODE_LOCAL
    try:
        with open(PREFS_FILE, "r") as f:
            data = json.load(f)
        mode = str(data.get("default_mode", MODE_LOCAL))
        return mode if mode in (MODE_LOCAL, MODE_HYBRID) else MODE_LOCAL
    except Exception:
        return MODE_LOCAL


def save_default_mode(mode: str) -> None:
    mode = MODE_HYBRID if mode == MODE_HYBRID else MODE_LOCAL
    payload = {"default_mode": mode}
    try:
        with open(PREFS_FILE, "w") as f:
            json.dump(payload, f, indent=2)
    except OSError:
        pass


def populate_compute_mode_combo(combo, *, include_local: bool = True) -> None:
    """Local only + Hybrid (remote server compute, local plot) per cluster."""
    combo.blockSignals(True)
    combo.clear()
    if include_local:
        combo.addItem("💻 Local only (this Mac)", LOCAL_TARGET)
    for cluster in load_clusters():
        name = cluster_display_name(cluster)
        combo.addItem(f"⚡ Hybrid: {name} (server compute)", hybrid_entry(cluster))
    combo.blockSignals(False)


def combo_mode(combo) -> str:
    data = combo.currentData()
    if data in (None, LOCAL_TARGET, MODE_LOCAL):
        return MODE_LOCAL
    if isinstance(data, dict):
        if data.get("mode") == MODE_HYBRID or "cluster" in data:
            return MODE_HYBRID
        if "host" in data or "user" in data:
            return MODE_HYBRID
    return MODE_LOCAL


def combo_cluster(combo) -> Optional[Dict[str, Any]]:
    if is_local_mode(combo):
        return None
    data = combo.currentData()
    if not isinstance(data, dict):
        return None
    if "cluster" in data:
        return data["cluster"]
    if "host" in data or "user" in data:
        return data
    return None


def is_local_mode(combo) -> bool:
    return combo_mode(combo) == MODE_LOCAL


def is_hybrid_mode(combo) -> bool:
    return combo_mode(combo) == MODE_HYBRID


# Back-compat alias used across suites
is_remote_target = is_hybrid_mode
selected_cluster = combo_cluster


def hybrid_exec_summary(combo) -> str:
    if is_local_mode(combo):
        return (
            "Local only: parse Wannier90, diagonalize, and plot on this Mac. "
            "GUI may pause on large TB models."
        )
    cluster = combo_cluster(combo) or {}
    name = cluster_display_name(cluster)
    return (
        f"Hybrid ({name}): heavy work on server — cached W90 upload, "
        "band diag on cluster (GPU if enabled), download NPZ, plot here. "
        "Repeat runs skip re-upload when cache unchanged."
    )


def effective_band_diag(
    combo,
    ui_engine: str,
    ui_device: str,
    *,
    auto_gpu: bool,
    w90_loaded: bool,
) -> Tuple[str, str]:
    """Hybrid fast path: Grizzly CUDA on server when user opts in.

    Explicit Grizzly + CPU is never overridden — remote CPU band diag is intentional.
    """
    if not is_hybrid_mode(combo):
        return ui_engine, ui_device
    if ui_engine == "grizzly" and str(ui_device).lower() == "cpu":
        return ui_engine, ui_device
    if auto_gpu and w90_loaded:
        return "grizzly", "cuda"
    if auto_gpu and ui_engine == "chinook":
        return "grizzly", "cuda"
    return ui_engine, ui_device
