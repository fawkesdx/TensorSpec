"""Guards + smoke for Qt-free ML core jobs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


CORE_ML = Path(__file__).resolve().parents[1] / "tensorspec" / "core" / "ml"


def test_core_ml_has_no_pyside6():
    hits = []
    for path in CORE_ML.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("PySide6", "PyQt", "QThread", "from Qt"):
            if needle in text:
                hits.append(f"{path.name}: {needle}")
    assert hits == [], f"Qt leak in core/ml: {hits}"


def test_run_clustering_kmeans_with_progress_callback():
    from tensorspec.core.ml.clustering import run_clustering

    rng = np.random.default_rng(0)
    embeds = rng.normal(size=(40, 8)).astype(np.float64)
    events = []

    labels, umap_res = run_clustering(
        embeds,
        algo="K-Means",
        k=3,
        eps=0.5,
        metric="euclidean",
        on_progress=lambda v, m: events.append((v, m)),
    )

    assert labels.shape == (40,)
    assert umap_res.shape == (40, 2)
    assert len(np.unique(labels)) <= 3
    assert events, "on_progress should fire"
    assert events[-1][0] == 100


def test_gui_workers_still_importable():
    from tensorspec.gui.ml.maestroai_clustering import ClusterWorker
    from tensorspec.gui.ml.maestroai_training_ssl import TrainWorker
    from tensorspec.gui.ml.maestroai_training_sup import SupTrainWorker, SupTestWorker
    from tensorspec.gui.ml.maestroai_active_learning import (
        ActiveLearningWorker,
        SimulateALWorker,
    )
    from tensorspec.gui.ml.maestroai_alignment import (
        AzimuthalTwistWorker,
        CoupledAzimuthTiltWorker,
        NormalTiltWorker,
    )

    assert ClusterWorker is not None
    assert TrainWorker is not None
    assert SupTrainWorker is not None
    assert SupTestWorker is not None
    assert ActiveLearningWorker is not None
    assert SimulateALWorker is not None
    assert AzimuthalTwistWorker is not None
    assert CoupledAzimuthTiltWorker is not None
    assert NormalTiltWorker is not None
