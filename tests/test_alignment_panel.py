import pytest
from tensorspec.gui.ml_session import MLSession


@pytest.fixture
def session():
    return MLSession()


def test_alignment_panel_builds(qapp, session):
    from tensorspec.gui.components.ml_tabs.alignment_panel import AlignmentPanel
    panel = AlignmentPanel(session)
    assert panel.combo_align_mode.count() == 3


def test_alignment_ref_combo_follows_workspace(qapp, session):
    from tensorspec.gui.components.ml_tabs.alignment_panel import AlignmentPanel
    panel = AlignmentPanel(session)
    assert panel.combo_align_ref.count() == 0
    session.add_dataset("fermi_a", {"kind": "Fermi Map (Cleaned)"})
    session.add_dataset("fermi_b", {"kind": "Fermi Map (Cleaned)"})
    assert [panel.combo_align_ref.itemText(i) for i in range(panel.combo_align_ref.count())] == ["fermi_a", "fermi_b"]


def test_alignment_ref_combo_clears_when_empty(qapp, session):
    from tensorspec.gui.components.ml_tabs.alignment_panel import AlignmentPanel
    panel = AlignmentPanel(session)
    session.add_dataset("fermi_a", {})
    session.remove_dataset("fermi_a")
    assert panel.combo_align_ref.count() == 0
