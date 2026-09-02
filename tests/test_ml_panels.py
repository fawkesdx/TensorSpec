"""Behavioural tests for the ML suite panels."""
import numpy as np
import pytest
from PySide6.QtWidgets import QListWidgetItem

from tensorspec.gui.ml_session import MLSession


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


@pytest.fixture
def session():
    return MLSession()


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


def test_active_learning_panel_builds(qapp, session):
    from tensorspec.gui.components.ml_tabs.active_learning_panel import ActiveLearningPanel

    panel = ActiveLearningPanel(session)
    assert panel.combo_al_algo.count() == 5


def test_active_learning_domain_combo_follows_the_session(qapp, session):
    from tensorspec.gui.components.ml_tabs.active_learning_panel import ActiveLearningPanel

    panel = ActiveLearningPanel(session)
    assert panel.combo_gp_domain.count() == 0

    session.activate({"domains_k5": [1], "domains_k8": [2], "other": 3})
    session.notify_domains()

    assert [panel.combo_gp_domain.itemText(i)
            for i in range(panel.combo_gp_domain.count())] == ["domains_k5", "domains_k8"]


def test_active_learning_domain_combo_clears_on_new_data(qapp, session):
    from tensorspec.gui.components.ml_tabs.active_learning_panel import ActiveLearningPanel

    panel = ActiveLearningPanel(session)
    session.activate({"domains_k5": [1]})
    session.notify_domains()
    session.activate({"no_domains_here": 1})
    session.notify_domains()
    assert panel.combo_gp_domain.count() == 0
