import pytest
from tensorspec.gui.ml_session import MLSession


@pytest.fixture
def session():
    return MLSession()


def test_simulate_al_panel_builds(qapp, session):
    from tensorspec.gui.components.ml_tabs.simulate_al_panel import SimulateALPanel
    panel = SimulateALPanel(session)
    assert panel.combo_sim_algo.count() == 5


def test_simulate_domain_combo_follows_the_session(qapp, session):
    from tensorspec.gui.components.ml_tabs.simulate_al_panel import SimulateALPanel
    panel = SimulateALPanel(session)
    assert panel.combo_sim_domain.count() == 0
    session.activate({"domains_k5": [1], "domains_k8": [2], "other": 3})
    session.notify_domains()
    assert [panel.combo_sim_domain.itemText(i) for i in range(panel.combo_sim_domain.count())] == ["domains_k5", "domains_k8"]


def test_simulate_domain_combo_clears_on_new_data(qapp, session):
    from tensorspec.gui.components.ml_tabs.simulate_al_panel import SimulateALPanel
    panel = SimulateALPanel(session)
    session.activate({"domains_k5": [1]})
    session.notify_domains()
    session.activate({"no_domains_here": 1})
    session.notify_domains()
    assert panel.combo_sim_domain.count() == 0
