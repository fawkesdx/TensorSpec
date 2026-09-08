from __future__ import annotations

import numpy as np


def labels_to_grid(assignments, provenances, *, ny, nx):
    if len(assignments) != len(provenances):
        raise ValueError("assignments and provenances length mismatch")
    grid = np.full((ny, nx), -1, dtype=np.int32)
    for lab, prov in zip(assignments, provenances, strict=True):
        y = int(prov["index"]["y"])
        x = int(prov["index"]["x"])
        grid[y, x] = int(lab)
    if (grid < 0).any():
        raise ValueError("incomplete spatial coverage for source grid")
    return grid


def reference_to_binary(map_, *, seed=0):
    from sklearn.cluster import KMeans

    flat = np.asarray(map_, dtype=np.float64).reshape(-1, 1)
    km = KMeans(n_clusters=2, random_state=seed, n_init=10)
    return km.fit_predict(flat).reshape(map_.shape).astype(np.int32)


def _mean_iou(pred: np.ndarray, ref: np.ndarray) -> float:
    classes = np.unique(np.concatenate([pred.ravel(), ref.ravel()]))
    ious: list[float] = []
    for c in classes:
        p_mask = pred == c
        r_mask = ref == c
        inter = float((p_mask & r_mask).sum())
        union = float((p_mask | r_mask).sum())
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def _best_permutation_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    pred = np.asarray(pred)
    ref = np.asarray(ref)
    if pred.shape != ref.shape:
        raise ValueError("pred and ref shape mismatch")

    best: dict[str, float] | None = None
    best_pred = pred
    for flip in (False, True):
        p = (1 - pred) if flip else pred
        iou = _mean_iou(p, ref)
        cand = {
            "ari": float(adjusted_rand_score(ref.ravel(), p.ravel())),
            "nmi": float(normalized_mutual_info_score(ref.ravel(), p.ravel())),
            "iou": float(iou),
        }
        if best is None or cand["iou"] > best["iou"]:
            best = cand
            best_pred = p
    assert best is not None
    best["contiguity"] = float(spatial_contiguity(best_pred))
    return best


def agreement_metrics(pred, ref) -> dict[str, float]:
    return _best_permutation_metrics(np.asarray(pred), np.asarray(ref))


def spatial_contiguity(labels) -> float:
    """Fraction of pixels whose label equals the 8-neighbour majority (ties count)."""
    lab = np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError("labels must be 2D")
    if lab.size == 0:
        return 0.0

    ny, nx = lab.shape
    matches = 0
    for y in range(ny):
        for x in range(nx):
            center = int(lab[y, x])
            neigh: list[int] = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < ny and 0 <= xx < nx:
                        neigh.append(int(lab[yy, xx]))
                    else:
                        neigh.append(center)
            counts: dict[int, int] = {}
            for v in neigh:
                counts[v] = counts.get(v, 0) + 1
            max_count = max(counts.values())
            modes = [v for v, c in counts.items() if c == max_count]
            if center in modes:
                matches += 1
    return float(matches) / float(ny * nx)
