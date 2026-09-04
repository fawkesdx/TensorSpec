"""Pure clustering / UMAP — no Qt."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import umap
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import hdbscan

ProgressFn = Optional[Callable[[int, str], None]]


def _emit(on_progress: ProgressFn, value: int, message: str) -> None:
    if on_progress is not None:
        on_progress(value, message)


def run_clustering(
    embeds: np.ndarray,
    algo: str,
    k: int,
    eps: float,
    metric: str = "euclidean",
    use_umap_first: bool = False,
    normalize_edcs: bool = False,
    on_progress: ProgressFn = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster embeddings; return (labels, umap_2d)."""
    data = embeds.copy()
    _emit(on_progress, 5, "Preprocessing data...")

    if normalize_edcs:
        _emit(on_progress, 10, "Normalizing EDC intensities...")
        max_vals = np.max(data, axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1.0
        norm_embeds = data / max_vals
    else:
        norm_embeds = data

    if metric == "cosine" and not use_umap_first:
        norm_embeds = normalize(norm_embeds, norm="l2", axis=1)
        safe_metric = "euclidean"
    else:
        safe_metric = metric

    if use_umap_first and algo != "DBSCAN":
        _emit(on_progress, 30, "Running pre-clustering UMAP...")
        cluster_data = umap.UMAP(
            n_components=3, metric=safe_metric, random_state=42
        ).fit_transform(norm_embeds)
        cluster_metric = "euclidean"
    else:
        cluster_data = norm_embeds
        cluster_metric = safe_metric

    _emit(on_progress, 50, f"Running {algo}...")
    if algo == "K-Means":
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(
            cluster_data
        )
    elif algo == "Gaussian Mixture":
        labels = GaussianMixture(
            n_components=k, random_state=42, reg_covar=1e-3
        ).fit_predict(cluster_data)
    elif algo == "Hierarchical":
        linkage_type = "ward" if cluster_metric == "euclidean" else "average"
        labels = AgglomerativeClustering(
            n_clusters=k, metric=cluster_metric, linkage=linkage_type
        ).fit_predict(cluster_data)
    elif algo == "DBSCAN":
        _emit(on_progress, 40, "Running UMAP for DBSCAN...")
        umap_res = umap.UMAP(
            n_components=2, metric=metric, random_state=42
        ).fit_transform(norm_embeds)
        _emit(on_progress, 80, "Running DBSCAN...")
        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(umap_res)
    elif algo == "HDBSCAN":
        labels = hdbscan.HDBSCAN(
            min_cluster_size=5, metric=cluster_metric
        ).fit_predict(cluster_data)
    else:
        raise ValueError(f"Unknown clustering algo: {algo}")

    if algo != "DBSCAN":
        _emit(on_progress, 80, "Running UMAP Reduction for Visuals...")
        umap_res = umap.UMAP(
            n_components=2, metric=metric, random_state=42
        ).fit_transform(norm_embeds)

    _emit(on_progress, 100, "Done!")
    return labels, umap_res
