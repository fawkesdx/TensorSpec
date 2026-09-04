"""Unit tests for DataViewerPanel lego grid insert + halfwidth bookkeeping."""

from types import SimpleNamespace

from tensorspec.gui.components.data_viewer_panel import DataViewerPanel


def _cell(row, col):
    return SimpleNamespace(grid_row=row, grid_col=col)


def test_insert_left_chains_without_collision():
    a = _cell(0, 0)
    views = [a]
    r, c = DataViewerPanel.compute_insert_slot(views, a.grid_row, a.grid_col, "Left")
    n1 = _cell(r, c)
    views.append(n1)
    assert sorted((w.grid_row, w.grid_col) for w in views) == [(0, 0), (0, 1)]
    assert (n1.grid_row, n1.grid_col) == (0, 0)
    assert (a.grid_row, a.grid_col) == (0, 1)

    r, c = DataViewerPanel.compute_insert_slot(views, n1.grid_row, n1.grid_col, "Left")
    n2 = _cell(r, c)
    views.append(n2)
    assert sorted((w.grid_row, w.grid_col) for w in views) == [(0, 0), (0, 1), (0, 2)]
    assert (n2.grid_row, n2.grid_col) == (0, 0)


def test_insert_right_chains():
    a = _cell(0, 0)
    views = [a]
    r, c = DataViewerPanel.compute_insert_slot(views, a.grid_row, a.grid_col, "Right")
    n1 = _cell(r, c)
    views.append(n1)
    r, c = DataViewerPanel.compute_insert_slot(views, n1.grid_row, n1.grid_col, "Right")
    n2 = _cell(r, c)
    views.append(n2)
    assert {(w.grid_row, w.grid_col) for w in views} == {(0, 0), (0, 1), (0, 2)}


def test_insert_top_and_bottom_chain():
    a = _cell(0, 0)
    views = [a]
    r, c = DataViewerPanel.compute_insert_slot(views, a.grid_row, a.grid_col, "Top")
    n1 = _cell(r, c)
    views.append(n1)
    assert (n1.grid_row, n1.grid_col) == (0, 0)
    assert (a.grid_row, a.grid_col) == (1, 0)

    r, c = DataViewerPanel.compute_insert_slot(views, a.grid_row, a.grid_col, "Bottom")
    n2 = _cell(r, c)
    views.append(n2)
    assert sorted((w.grid_row, w.grid_col) for w in views) == [(0, 0), (1, 0), (2, 0)]


def test_insert_left_into_occupied_column_shifts():
    """Snap Left from a non-leftmost panel must not collide with neighbor."""
    a = _cell(0, 0)
    b = _cell(0, 1)
    views = [a, b]
    r, c = DataViewerPanel.compute_insert_slot(views, b.grid_row, b.grid_col, "Left")
    n = _cell(r, c)
    views.append(n)
    positions = sorted((w.grid_row, w.grid_col) for w in views)
    assert positions == [(0, 0), (0, 1), (0, 2)]
    assert len(positions) == len(set(positions))
