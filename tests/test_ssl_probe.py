import numpy as np
from tensorspec.core.ml.ssl.probe import (
    agreement_metrics,
    labels_to_grid,
    reference_to_binary,
    spatial_contiguity,
)


def test_labels_to_grid_fills_yx():
    assigns = np.array([0, 1, 0, 1])
    prov = [
        {"index": {"y": 0, "x": 0}},
        {"index": {"y": 0, "x": 1}},
        {"index": {"y": 1, "x": 0}},
        {"index": {"y": 1, "x": 1}},
    ]
    grid = labels_to_grid(assigns, prov, ny=2, nx=2)
    np.testing.assert_array_equal(grid, [[0, 1], [0, 1]])


def test_labels_to_grid_rejects_hole():
    import pytest
    with pytest.raises(ValueError, match="incomplete"):
        labels_to_grid(
            np.array([0]),
            [{"index": {"y": 0, "x": 0}}],
            ny=2,
            nx=2,
        )


def test_agreement_permutation_invariant():
    ref = np.array([[0, 0], [1, 1]])
    pred = np.array([[1, 1], [0, 0]])  # swapped labels
    m = agreement_metrics(pred, ref)
    assert m["ari"] == 1.0
    assert m["nmi"] == 1.0
    assert m["iou"] == 1.0


def test_contiguity_perfect_blocks():
    lab = np.array([[0, 0], [1, 1]])
    assert spatial_contiguity(lab) == 1.0
