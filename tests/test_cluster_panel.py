"""Behavioural tests for ClusterPanel."""
import pytest

from tensorspec.gui.ml.session import MLSession


@pytest.fixture
def session():
    return MLSession()


def test_cluster_panel_builds(qapp, session):
    from tensorspec.gui.components.ml_tabs.cluster_panel import ClusterPanel

    panel = ClusterPanel(session)
    assert panel.combo_algo.count() == 5


def test_cluster_embedding_combo_follows_the_session(qapp, session):
    from tensorspec.gui.components.ml_tabs.cluster_panel import ClusterPanel

    panel = ClusterPanel(session)
    assert panel.combo_embed.count() == 2
    assert panel.combo_embed.itemText(0) == "Integrated EDC (from Viewer)"
    assert panel.combo_embed.itemText(1) == "Integrated MDC (from Viewer)"

    session.activate({"embeddings_ae": [1], "embeddings_simclr": [2], "other": 3})
    session.notify_embeddings()

    assert [panel.combo_embed.itemText(i) for i in range(panel.combo_embed.count())] == [
        "embeddings_ae",
        "embeddings_simclr",
        "Integrated EDC (from Viewer)",
        "Integrated MDC (from Viewer)",
    ]
