import numpy as np

from tensorspec.gui.components.data_viewer_panel import is_ml_label_layer, ml_label_spatial_map


def test_is_ml_label_layer():
    assert is_ml_label_layer("domains_SimCLR")
    assert is_ml_label_layer("Labels_k3")
    assert is_ml_label_layer("Supervised Probabilities")
    assert not is_ml_label_layer("Intensity")


def test_ml_label_spatial_map_flat_domains():
    value_shape = (10, 5, 4, 6)  # E, A, Y, X
    labels = np.arange(24)
    out = ml_label_spatial_map(labels, value_shape)
    assert out is not None
    assert out.shape == (4, 6)
    assert out[0, 0] == 0
    assert out[-1, -1] == 23


def test_ml_label_spatial_map_prob_cube():
    value_shape = (8, 4, 3, 3)
    probs = np.zeros((3, 3, 2))
    probs[:, :, 1] = 1.0
    out = ml_label_spatial_map(probs, value_shape)
    assert out is not None
    assert out.shape == (3, 3)
    assert np.all(out == 1)
