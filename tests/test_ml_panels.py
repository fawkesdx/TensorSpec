"""Behavioural tests for the ML suite panels."""
import numpy as np
from PySide6.QtWidgets import QListWidgetItem


def xy_scan(**extra):
    """A minimal 4D "XY Scan (Cleaned)" entry shaped like the loader produces.

    _convert_to_tensor_data always builds four axes, so value must be 4D and
    each axis length must match its dimension.
    """
    data = {
        "kind": "XY Scan (Cleaned)",
        "value": np.zeros((3, 4, 5, 6)),
        "E": np.linspace(-1, 0, 3),
        "angle": np.linspace(-5, 5, 4),
        "y": np.arange(5, dtype=float),
        "x": np.arange(6, dtype=float),
    }
    data.update(extra)
    return data


def test_activating_a_dataset_pushes_it_to_the_viewer(qapp):
    """Regression: activate_data called viewer.set_data, which does not exist.

    DataViewerPanel exposes load_data(TensorData); set_data belongs to
    Maestro4DViewer. Clicking a workspace entry raised AttributeError.
    """
    from tensorspec.gui.maestroai.maestroai_gui import MaestroAIApp

    win = MaestroAIApp()
    win.workspace["probe"] = xy_scan(embeddings_ae=[1], domains_k5=[2])

    win.activate_data(QListWidgetItem("probe"))

    assert win.current_view_data is win.workspace["probe"]
    assert win.viewer.tensor_data is not None
    assert win.viewer.tensor_data.ndim == 4
    win.close()
