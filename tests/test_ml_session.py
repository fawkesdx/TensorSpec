"""Tests for MLSession, the state the ML panels share.

The ML tabs used to reach directly into each other's widgets: activate_data
repopulated combo boxes owned by the clustering, active-learning and
simulate-AL tabs, and on_cluster_finish did the same. MLSession replaces those
writes with signals, so these tests pin the signal contract the panels rely on.
"""
import pytest

from tensorspec.gui.ml_session import MLSession


@pytest.fixture
def session():
    return MLSession()


@pytest.fixture
def recorder():
    """Collects signal payloads so tests can assert on emission order."""
    return []


def test_starts_empty(session):
    assert session.workspace == {}
    assert session.current_folder == ""
    assert session.current_view_data is None
    assert session.viewer is None


def test_key_helpers_are_safe_with_no_active_data(session):
    assert session.embedding_keys() == []
    assert session.domain_keys() == []


def test_add_dataset_stores_and_announces(session, recorder):
    session.workspace_changed.connect(lambda: recorder.append("changed"))

    session.add_dataset("scan_a", {"kind": "XY Scan (Cleaned)"})

    assert session.workspace["scan_a"]["kind"] == "XY Scan (Cleaned)"
    assert recorder == ["changed"]


def test_remove_dataset_announces(session, recorder):
    session.add_dataset("scan_a", {})
    session.workspace_changed.connect(lambda: recorder.append("changed"))

    session.remove_dataset("scan_a")

    assert "scan_a" not in session.workspace
    assert recorder == ["changed"]


def test_removing_an_absent_dataset_is_silent(session, recorder):
    session.workspace_changed.connect(lambda: recorder.append("changed"))

    session.remove_dataset("never_added")

    assert recorder == []


def test_activate_publishes_the_dataset(session, recorder):
    session.data_activated.connect(recorder.append)
    data = {"kind": "XY Scan (Cleaned)", "embeddings_ae": [1]}

    session.activate(data)

    assert session.current_view_data is data
    assert recorder == [data]


def test_key_helpers_filter_by_prefix(session):
    session.activate({
        "embeddings_ae": 1,
        "embeddings_vae": 2,
        "domains_k5": 3,
        "value": 4,
    })

    assert session.embedding_keys() == ["embeddings_ae", "embeddings_vae"]
    assert session.domain_keys() == ["domains_k5"]


def test_notify_helpers_emit_the_current_keys(session):
    session.activate({"embeddings_ae": 1, "domains_k5": 2})
    embeds, domains = [], []
    session.embeddings_changed.connect(embeds.extend)
    session.domains_changed.connect(domains.extend)

    session.notify_embeddings()
    session.notify_domains()

    assert embeds == ["embeddings_ae"]
    assert domains == ["domains_k5"]


def test_notify_helpers_emit_empty_after_switching_data(session):
    """Panels must clear their combos when the new dataset has no such keys."""
    session.activate({"domains_k5": 1})
    session.notify_domains()

    domains = []
    session.domains_changed.connect(lambda keys: domains.append(list(keys)))
    session.activate({"value": 0})
    session.notify_domains()

    assert domains == [[]]


def test_set_status_relays_value_and_message(session, recorder):
    session.status_changed.connect(lambda v, m: recorder.append((v, m)))

    session.set_status(42, "training")

    assert recorder == [(42, "training")]
