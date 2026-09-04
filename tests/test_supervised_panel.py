import pytest

from tensorspec.gui.ml.session import MLSession


@pytest.fixture
def session():
    return MLSession()


def test_supervised_panel_builds(qapp, session):
    from tensorspec.gui.components.ml_tabs.supervised_panel import SupervisedPanel

    panel = SupervisedPanel(session)
    assert panel.spin_sup_classes.minimum() == 2
    assert panel.spin_sup_classes.maximum() == 10
    assert panel.spin_sup_classes.value() == 3
    assert panel.sup_buttons == []
    assert panel.sup_data == {}
    assert panel.sup_coords == {}


def test_create_sup_buttons_makes_n_buttons(qapp, session):
    from tensorspec.gui.components.ml_tabs.supervised_panel import SupervisedPanel

    panel = SupervisedPanel(session)
    panel.spin_sup_classes.setValue(5)
    panel.create_sup_buttons()
    assert len(panel.sup_buttons) == 5
    assert len(panel.sup_data) == 5
    assert len(panel.sup_coords) == 5
    for i, btn in enumerate(panel.sup_buttons):
        assert btn.text() == f"Assign Target Coordinate to Label {i + 1} (Count: 0)"
