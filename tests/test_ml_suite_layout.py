"""Layout regression tests for the ML suite.

These guard two defects that made the suite hard to use:

1. The plot canvases collapsed because MplCanvas used an Ignored/Ignored size
   policy while the control widgets stacked above them had real minimums. At
   the launcher's window size the SSL loss canvas rendered 36px tall and the
   memory plot 0px, i.e. invisible.
2. The whole QTabWidget sat inside a QScrollArea, so scrolling a tall tab
   scrolled the tab bar itself out of reach.
"""
import pytest
from PySide6.QtWidgets import QScrollArea, QSplitter, QTabWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

MIN_CANVAS_HEIGHT = 150
MIN_VIEWER_WIDTH = 320
WINDOW_SIZES = [(1500, 900), (1100, 700), (1000, 620)]
EXPECTED_GROUPS = ["Train", "Cluster", "Align", "Steer", "Models"]


@pytest.fixture(scope="module")
def suite(qapp):
    from tensorspec.gui.maestroai.maestroai_gui import MaestroAIApp

    win = MaestroAIApp()
    win.show()
    qapp.processEvents()
    yield win
    win.close()


def ancestors(widget):
    node = widget.parentWidget() if widget else None
    while node is not None:
        yield node
        node = node.parentWidget()


def leaf_pages(tabs, qapp, trail=()):
    """Yield (path, page) for every leaf tab, activating each one on the way.

    A page is only laid out once its tab has been shown, so every tab has to be
    made current before its canvas geometry means anything.
    """
    for i in range(tabs.count()):
        tabs.setCurrentIndex(i)
        qapp.processEvents()
        page = tabs.widget(i)
        path = trail + (tabs.tabText(i),)
        if isinstance(page, QTabWidget):
            yield from leaf_pages(page, qapp, path)
        else:
            yield " > ".join(path), page


def root_tabs(suite):
    return suite.centralWidget().findChild(QTabWidget)


def test_top_level_groups_are_workflow_ordered(suite):
    tabs = root_tabs(suite)
    assert [tabs.tabText(i) for i in range(tabs.count())] == EXPECTED_GROUPS


def test_suite_does_not_open_on_a_placeholder_tab(suite):
    """The Models group holds unimplemented stubs, so it must not be first."""
    tabs = root_tabs(suite)
    assert tabs.tabText(tabs.currentIndex()) == "Train"


def test_single_page_groups_have_no_nested_bar(suite):
    """Cluster and Align hold one tab each; they should render it directly."""
    tabs = root_tabs(suite)
    for label in ("Cluster", "Align"):
        index = [tabs.tabText(i) for i in range(tabs.count())].index(label)
        assert not isinstance(tabs.widget(index), QTabWidget)


def test_tab_bar_is_not_inside_a_scroll_area(suite):
    tabs = root_tabs(suite)
    assert not any(isinstance(a, QScrollArea) for a in ancestors(tabs))


@pytest.mark.parametrize("width,height", WINDOW_SIZES)
def test_every_canvas_has_usable_height(suite, qapp, width, height):
    suite.resize(width, height)
    qapp.processEvents()

    thin = []
    for label, page in leaf_pages(root_tabs(suite), qapp):
        for canvas in page.findChildren(FigureCanvas):
            if canvas.height() < MIN_CANVAS_HEIGHT:
                thin.append(f"{label} ({canvas.height()}px)")
    assert not thin, f"canvases too short at {width}x{height}: {thin}"


@pytest.mark.parametrize("width,height", WINDOW_SIZES)
def test_viewer_pane_is_never_crushed(suite, qapp, width, height):
    suite.resize(width, height)
    qapp.processEvents()
    splitter = suite.centralWidget().findChild(QSplitter)
    assert splitter.sizes()[1] >= MIN_VIEWER_WIDTH


def test_canvases_expand_rather_than_being_ignored(suite, qapp):
    """Pin the size policy directly, since that was the root cause."""
    from PySide6.QtWidgets import QSizePolicy

    for _, page in leaf_pages(root_tabs(suite), qapp):
        for canvas in page.findChildren(FigureCanvas):
            policy = canvas.sizePolicy()
            assert policy.verticalPolicy() != QSizePolicy.Policy.Ignored
            assert canvas.minimumHeight() > 0
