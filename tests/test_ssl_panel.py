"""Behavioural tests for SSLTrainingPanel."""
import numpy as np
import pytest

from tensorspec.gui.ml.session import MLSession


@pytest.fixture
def session():
    return MLSession()


def test_ssl_panel_builds(qapp, session):
    from tensorspec.gui.components.ml_tabs.ssl_panel import SSLTrainingPanel

    panel = SSLTrainingPanel(session)
    assert panel.chk_ae is not None
    assert panel.chk_ae.isChecked()
    assert panel.combo_loss_view is not None


def test_ssl_panel_on_train_finish_notifies_embeddings(qapp, session):
    from tensorspec.gui.components.ml_tabs.ssl_panel import SSLTrainingPanel

    data = {"value": np.zeros((2, 3, 4, 5))}
    session.activate(data)
    panel = SSLTrainingPanel(session)
    panel.active_train_target = data

    notified = []
    session.embeddings_changed.connect(lambda keys: notified.append(list(keys)))

    emb = np.zeros((20, 32))
    panel.on_train_finish({"embeddings_autoencoder": emb})

    assert data["embeddings_autoencoder"] is emb
    assert notified == [["embeddings_autoencoder"]]
    assert panel.btn_train.isEnabled()
