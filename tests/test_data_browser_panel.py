"""Tests for DataBrowserPanel (disk browser + RAM workspace)."""
import numpy as np
import pytest
from PySide6.QtWidgets import QListWidgetItem

from tensorspec.core.data_models import TensorData
from tensorspec.gui.ml_session import MLSession


def xy_scan(**extra):
    return {
        "kind": "XY Scan (Cleaned)",
        "value": np.zeros((3, 4, 5, 6)),
        "E": np.linspace(-1, 0, 3),
        "angle": np.linspace(-5, 5, 4),
        "y": np.arange(5, dtype=float),
        "x": np.arange(6, dtype=float),
        **extra,
    }


@pytest.fixture
def session():
    return MLSession()


def test_data_browser_panel_builds(qapp, session):
    from tensorspec.gui.components.ml_tabs.data_browser_panel import DataBrowserPanel

    panel = DataBrowserPanel(session)
    assert panel.disk_list is not None
    assert panel.workspace_list is not None
    assert panel.path_input is not None
    assert panel.btn_load is not None


def test_workspace_list_syncs_on_add_dataset(qapp, session):
    from tensorspec.gui.components.ml_tabs.data_browser_panel import DataBrowserPanel

    panel = DataBrowserPanel(session)
    assert panel.workspace_list.count() == 0

    session.add_dataset("scan_a", {"kind": "XY Scan (Cleaned)"})

    assert panel.workspace_list.count() == 1
    assert panel.workspace_list.item(0).text() == "scan_a"
    assert "scan_a" in session.workspace


def test_activate_xy_scan_updates_session_and_viewer(qapp, session):
    from tensorspec.gui.components.ml_tabs.data_browser_panel import DataBrowserPanel
    from tensorspec.gui.components.data_viewer_panel import DataViewerPanel

    session.viewer = DataViewerPanel()
    panel = DataBrowserPanel(session)
    data = xy_scan(embeddings_ae=[1], domains_k5=[2])
    session.add_dataset("probe", data)

    domains, embeds = [], []
    session.domains_changed.connect(lambda k: domains.append(list(k)))
    session.embeddings_changed.connect(lambda k: embeds.append(list(k)))

    panel.activate_data(panel.workspace_list.item(0))

    assert session.current_view_data is data
    assert session.viewer.tensor_data is not None
    assert session.viewer.tensor_data.ndim == 4
    assert domains == [["domains_k5"]]
    assert embeds == [["embeddings_ae"]]


def test_activate_tensor_data_keeps_ml_dict_and_viewer_tensor(qapp, session):
    from tensorspec.gui.components.ml_tabs.data_browser_panel import DataBrowserPanel
    from tensorspec.gui.components.data_viewer_panel import DataViewerPanel

    tensor = TensorData(
        value=np.zeros((2, 3, 4, 5)),
        axes=[np.arange(2), np.arange(3), np.arange(4), np.arange(5)],
        labels=["Y", "X", "Energy", "Angle"],
        units=["um", "um", "eV", "deg"],
        data_type="XY Scan Fine",
        metadata={"layers": {"embeddings_ae": [1], "domains_k5": [2]}},
    )
    session.viewer = DataViewerPanel()
    panel = DataBrowserPanel(session)
    panel.workspace_tensors["probe"] = tensor
    session.add_dataset("probe", tensor)

    panel.activate_data(QListWidgetItem("probe"))

    assert session.viewer.tensor_data is tensor
    ml = session.current_view_data
    assert ml["value"] is tensor.value
    assert ml["embeddings_ae"] == [1]
    assert session.workspace["probe"] is ml


def test_legacy_convert_helpers_roundtrip():
    from tensorspec.gui.components.ml_tabs.legacy_data import (
        convert_to_tensor_data,
        tensor_to_ml_dict,
    )

    data = xy_scan(embeddings_ae=[1], domains_k5=[2])
    td = convert_to_tensor_data(data)
    assert td.ndim == 4
    back = tensor_to_ml_dict(td)
    assert back["kind"] == "XY Scan (Cleaned)"
    assert back["embeddings_ae"] == [1]
    assert back["domains_k5"] == [2]
