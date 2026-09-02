# ML Suite Modularization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Break the 1596-line `tensorspec/gui/maestroai/maestroai_gui.py` monolith into a thin `tensorspec/gui/suites/ml_suite.py` shell plus one focused panel module per tab, matching how the DFT, ARPES and Crystal suites are already structured.

**Architecture:** A strangler refactor. An `MLSession` object owns the state the tabs share (workspace dict, active dataset, status reporting) and publishes four Qt signals that replace the direct cross-tab widget writes. Panels are extracted one at a time; after each extraction the existing `MaestroAIApp` immediately instantiates the new panel, so the application stays runnable and the layout regression test stays green at every commit. Only once every panel is extracted does the slim shell move to `suites/ml_suite.py` and become a `QWidget` like its sibling suites.

**Tech Stack:** Python 3.11, PySide6 (Qt6), matplotlib QtAgg backend, pytest, psutil.

## Global Constraints

- Branch: `TensorSpec_GUI`. Never merge to `main`.
- All GUI tests must run headless: set `QT_QPA_PLATFORM=offscreen` and `MPLCONFIGDIR` to a writable temp path.
- Python interpreter for all commands: `TensorSpec_env/bin/python`. Tests need `PYTHONPATH` set to the repo root.
- The suite must remain importable and launchable from `tensorspec/gui/main_browser.py` after every single task.
- Canvas minimum height is 220px and `MplCanvas` must keep an `Expanding/Expanding` size policy. Never reintroduce `QSizePolicy.Ignored` (that was the bug fixed in commit `92d700c`).
- No tab bar may live inside a `QScrollArea`. Scroll the control column only.
- Panel modules must not import `maestroai_gui`. Dependencies point one way: `suites/ml_suite.py` -> `components/ml_tabs/*` -> `ml_session` -> workers in `maestroai/`.
- Existing worker/model/guide modules under `tensorspec/gui/maestroai/` (`maestroai_training_ssl.py`, `maestroai_models.py`, `maestroai_guides.py`, `maestro_loader.py`, etc.) are already modular. Do not move or rewrite them in this plan.

---

## Why this is tractable

AST analysis of all 57 `MaestroAIApp` methods (`scratch/analyze_coupling.py`) found only **five** methods that touch another tab's widgets:

| Method | Reaches into |
|---|---|
| `activate_data` | cluster (`combo_embed`), al (`combo_gp_domain`), sim (`combo_sim_domain`) |
| `load_session` | cluster (`combo_embed`, `combo_parent_filter`), al, sim |
| `on_load_finish` | align (`combo_align_ref`) |
| `on_train_finish` | cluster (`combo_embed`) |
| `on_cluster_finish` | al (`combo_gp_domain`), sim (`combo_sim_domain`) |

Every method belonging to the supervised, active-learning, simulate-AL and alignment tabs is already self-contained apart from shared infrastructure. The shared surface is small: `status` (25 methods), `current_view_data` (23), `prog_bar` (12), `viewer` (10), `workspace` (7), `current_folder` (4).

So four signals plus one context object replace the entire cross-tab coupling.

## File Structure

**Create:**

| Path | Responsibility | Est. lines |
|---|---|---|
| `tensorspec/gui/ml_session.py` | `MLSession`: workspace dict, active dataset, status relay, 4 cross-tab signals | ~130 |
| `tensorspec/gui/components/ml_tabs/__init__.py` | Re-export the panel classes | ~20 |
| `tensorspec/gui/components/ml_tabs/layout.py` | `scrollable()`, `split_panel()`, `tab_group()` | ~70 |
| `tensorspec/gui/components/ml_tabs/active_learning_panel.py` | `ActiveLearningPanel` | ~90 |
| `tensorspec/gui/components/ml_tabs/simulate_al_panel.py` | `SimulateALPanel` | ~160 |
| `tensorspec/gui/components/ml_tabs/alignment_panel.py` | `AlignmentPanel` | ~150 |
| `tensorspec/gui/components/ml_tabs/supervised_panel.py` | `SupervisedPanel` | ~220 |
| `tensorspec/gui/components/ml_tabs/ssl_panel.py` | `SSLTrainingPanel` | ~160 |
| `tensorspec/gui/components/ml_tabs/cluster_panel.py` | `ClusterPanel` | ~300 |
| `tensorspec/gui/components/ml_tabs/data_browser_panel.py` | `DataBrowserPanel`: disk browse, load worker, session save/load | ~310 |
| `tensorspec/gui/suites/ml_suite.py` | `MLSuite(QWidget)`: 3-pane splitter, grouped tabs, status row | ~190 |
| `tests/test_ml_session.py` | Session state + signal emission | ~110 |
| `tests/test_ml_panels.py` | Per-panel construction + signal-driven combo refresh | ~200 |
| `tests/test_ml_suite_layout.py` | Layout regression: canvas heights, tab groups, no scrolled tab bar | ~120 |

**Modify:**

- `tensorspec/gui/maestroai/maestroai_gui.py` — progressively emptied, then deleted in Task 12.
- `tensorspec/gui/main_browser.py:526-543` (`launch_ml_suite`) — switch to `MLSuite` + `FloatingViewerWindow`.

**Naming note:** a `ml_tabs/` subpackage is used rather than flat `components/ml_*.py` files because there are seven panels; this follows the existing `components/crystal_tabs/` precedent.

---

### Task 0: Fix the broken `viewer.set_data` call (pre-existing bug, blocks everything)

`MaestroAIApp` sets `self.viewer = DataViewerPanel()`, but two methods call `self.viewer.set_data(...)`. `set_data` exists only on `Maestro4DViewer` (`maestro_4d_viewer.py:170`); `DataViewerPanel` exposes `load_data(tensor_data: TensorData)`. Both call sites raise `AttributeError` at runtime:

```
>>> w.activate_data(item)
AttributeError: 'DataViewerPanel' object has no attribute 'set_data'
```

This breaks the suite's main interaction path — clicking a workspace entry — and also `load_session`. It is present in baseline commit `050c84b` and is unrelated to the layout work. Fix it before refactoring, so the extraction is not chasing a moving target.

**Files:**
- Modify: `tensorspec/gui/maestroai/maestroai_gui.py:859` and `:943`
- Test: `tests/test_ml_panels.py`

- [ ] **Step 1: Write the failing test**

```python
def test_activating_a_dataset_pushes_it_to_the_viewer(qapp):
    from PySide6.QtWidgets import QListWidgetItem
    from tensorspec.gui.maestroai.maestroai_gui import MaestroAIApp

    win = MaestroAIApp()
    win.workspace["probe"] = {
        "kind": "XY Scan (Cleaned)",
        "value": __import__("numpy").zeros((2, 2, 2, 2)),
        "embeddings_ae": [1],
        "domains_k5": [2],
    }
    win.activate_data(QListWidgetItem("probe"))
    assert win.current_view_data is win.workspace["probe"]
    win.close()
```

- [ ] **Step 2: Run to verify it fails**

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/mplcache PYTHONPATH="$PWD" \
  TensorSpec_env/bin/python -m pytest tests/test_ml_panels.py -k viewer -v
```

Expected: FAIL with `AttributeError: 'DataViewerPanel' object has no attribute 'set_data'`.

- [ ] **Step 3: Use the same conversion the working call site uses**

`load_workspace_to_viewer` already does this correctly at line 770. Apply that pattern at both broken sites:

```python
            td = self._convert_to_tensor_data(data)
            self.viewer.load_data(td)
```

At line 859 the variable is `self.current_view_data` rather than `data`; convert that instead.

- [ ] **Step 4: Run to verify it passes**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tensorspec/gui/maestroai/maestroai_gui.py tests/test_ml_panels.py
git commit -m "fix(gui): push activated ML datasets to the viewer via load_data"
```

---

### Task 1: Layout regression test (safety net, before any code moves)

**Files:**
- Create: `tests/test_ml_suite_layout.py`
- Reference: `scratch/ml_layout_smoke.py` (existing throwaway harness to port)

**Interfaces:**
- Consumes: nothing.
- Produces: `ml_window(qapp)` pytest fixture pattern and the constant `MIN_CANVAS_HEIGHT = 150`, reused by `tests/test_ml_panels.py` in Task 4 onward.

- [ ] **Step 1: Write the test**

```python
"""Layout regression tests for the ML suite.

Guards commit 92d700c: the plot canvases previously collapsed to as little as
0px tall because MplCanvas used an Ignored/Ignored size policy while the
control widgets above it had real minimums.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "QtAgg")

from PySide6.QtWidgets import QApplication, QScrollArea, QSplitter, QTabWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

MIN_CANVAS_HEIGHT = 150
WINDOW_SIZES = [(1500, 900), (1100, 700), (1000, 620)]
EXPECTED_GROUPS = ["Train", "Cluster", "Align", "Steer", "Models", "System"]


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
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
    for i in range(tabs.count()):
        tabs.setCurrentIndex(i)
        qapp.processEvents()
        page = tabs.widget(i)
        path = trail + (tabs.tabText(i),)
        if isinstance(page, QTabWidget):
            yield from leaf_pages(page, qapp, path)
        else:
            yield " > ".join(path), page


def test_top_level_groups(suite):
    tabs = suite.centralWidget().findChild(QTabWidget)
    groups = [tabs.tabText(i) for i in range(tabs.count())]
    assert groups == EXPECTED_GROUPS


def test_tab_bar_is_not_inside_a_scroll_area(suite):
    tabs = suite.centralWidget().findChild(QTabWidget)
    assert not any(isinstance(a, QScrollArea) for a in ancestors(tabs))


@pytest.mark.parametrize("width,height", WINDOW_SIZES)
def test_every_canvas_has_usable_height(suite, qapp, width, height):
    suite.resize(width, height)
    qapp.processEvents()

    tabs = suite.centralWidget().findChild(QTabWidget)
    thin = []
    for label, page in leaf_pages(tabs, qapp):
        for canvas in page.findChildren(FigureCanvas):
            if canvas.height() < MIN_CANVAS_HEIGHT:
                thin.append(f"{label}: {canvas.height()}px")
    assert not thin, f"canvases too short at {width}x{height}: {thin}"


@pytest.mark.parametrize("width,height", WINDOW_SIZES)
def test_viewer_pane_is_never_collapsed(suite, qapp, width, height):
    suite.resize(width, height)
    qapp.processEvents()
    splitter = suite.centralWidget().findChild(QSplitter)
    assert splitter.sizes()[1] >= 320
```

- [ ] **Step 2: Run it and confirm it passes against current code**

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/mplcache PYTHONPATH="$PWD" \
  TensorSpec_env/bin/python -m pytest tests/test_ml_suite_layout.py -v
```

Expected: all tests PASS. If any fail, stop — the baseline is not what this plan assumes.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ml_suite_layout.py
git commit -m "test(gui): pin the ML suite layout against canvas collapse"
```

- [ ] **Step 4: Delete the throwaway harness**

```bash
git rm --cached scratch/ml_layout_smoke.py 2>/dev/null; rm -f scratch/ml_layout_smoke.py scratch/ml_layout_shot.py scratch/analyze_coupling.py
git commit -am "chore: drop scratch layout harnesses now covered by tests" || true
```

---

### Task 2: `MLSession` shared context

**Files:**
- Create: `tensorspec/gui/ml_session.py`
- Test: `tests/test_ml_session.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the `MLSession` API that every later task depends on:
  - Signals: `workspace_changed()`, `data_activated(dict)`, `embeddings_changed(list)`, `domains_changed(list)`, `status_changed(int, str)`
  - Attributes: `workspace: dict`, `current_folder: str`, `current_view_data: dict | None`, `viewer` (set by the suite)
  - Methods: `set_status(value: int, message: str) -> None`, `add_dataset(name: str, data: dict) -> None`, `activate(data: dict) -> None`, `embedding_keys() -> list[str]`, `domain_keys() -> list[str]`, `notify_embeddings() -> None`, `notify_domains() -> None`

- [ ] **Step 1: Write the failing test**

```python
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tensorspec.gui.ml_session import MLSession


@pytest.fixture
def session():
    return MLSession()


def test_starts_empty(session):
    assert session.workspace == {}
    assert session.current_view_data is None
    assert session.embedding_keys() == []
    assert session.domain_keys() == []


def test_add_dataset_announces_workspace_change(session, qtbot=None):
    seen = []
    session.workspace_changed.connect(lambda: seen.append(True))
    session.add_dataset("scan_a", {"kind": "XY Scan (Cleaned)"})
    assert session.workspace["scan_a"]["kind"] == "XY Scan (Cleaned)"
    assert seen == [True]


def test_activate_publishes_the_dataset(session):
    payload = {}
    session.data_activated.connect(payload.update)
    data = {"kind": "XY Scan (Cleaned)", "embeddings_ae": [1], "domains_k5": [2]}
    session.activate(data)
    assert session.current_view_data is data
    assert payload["embeddings_ae"] == [1]


def test_key_helpers_filter_by_prefix(session):
    session.activate({"embeddings_ae": 1, "embeddings_vae": 2,
                      "domains_k5": 3, "other": 4})
    assert session.embedding_keys() == ["embeddings_ae", "embeddings_vae"]
    assert session.domain_keys() == ["domains_k5"]


def test_key_helpers_are_safe_with_no_active_data(session):
    assert session.embedding_keys() == []
    assert session.domain_keys() == []


def test_notify_helpers_emit_current_keys(session):
    session.activate({"embeddings_ae": 1, "domains_k5": 2})
    embeds, domains = [], []
    session.embeddings_changed.connect(embeds.extend)
    session.domains_changed.connect(domains.extend)
    session.notify_embeddings()
    session.notify_domains()
    assert embeds == ["embeddings_ae"]
    assert domains == ["domains_k5"]


def test_set_status_relays_value_and_message(session):
    got = []
    session.status_changed.connect(lambda v, m: got.append((v, m)))
    session.set_status(42, "training")
    assert got == [(42, "training")]
```

- [ ] **Step 2: Run to verify it fails**

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH="$PWD" TensorSpec_env/bin/python -m pytest tests/test_ml_session.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tensorspec.gui.ml_session'`.

- [ ] **Step 3: Write the implementation**

```python
"""Shared state for the ML suite.

The ML tabs used to reach directly into each other's widgets: activate_data
repopulated combo boxes belonging to the clustering, active-learning and
simulate-AL tabs, and on_cluster_finish did the same. MLSession replaces those
writes with signals so each panel only ever touches its own widgets.
"""
from PySide6.QtCore import QObject, Signal


class MLSession(QObject):
    """Workspace and active-dataset state shared by every ML panel."""

    # A dataset was added to or removed from the in-memory workspace.
    workspace_changed = Signal()
    # The user selected a dataset to work on; payload is that dataset.
    data_activated = Signal(dict)
    # The "embeddings_*" keys available on the active dataset changed.
    embeddings_changed = Signal(list)
    # The "domains_*" keys available on the active dataset changed.
    domains_changed = Signal(list)
    # Progress value (0-100) and message for the suite status bar.
    status_changed = Signal(int, str)

    EMBEDDING_PREFIX = "embeddings_"
    DOMAIN_PREFIX = "domains_"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.workspace = {}
        self.current_folder = ""
        self.current_view_data = None
        # Assigned by MLSuite once the shared DataViewerPanel exists.
        self.viewer = None

    def set_status(self, value, message):
        self.status_changed.emit(value, message)

    def add_dataset(self, name, data):
        self.workspace[name] = data
        self.workspace_changed.emit()

    def remove_dataset(self, name):
        if name in self.workspace:
            del self.workspace[name]
            self.workspace_changed.emit()

    def activate(self, data):
        self.current_view_data = data
        self.data_activated.emit(data)

    def _keys_with_prefix(self, prefix):
        if not self.current_view_data:
            return []
        return [k for k in self.current_view_data if k.startswith(prefix)]

    def embedding_keys(self):
        return self._keys_with_prefix(self.EMBEDDING_PREFIX)

    def domain_keys(self):
        return self._keys_with_prefix(self.DOMAIN_PREFIX)

    def notify_embeddings(self):
        self.embeddings_changed.emit(self.embedding_keys())

    def notify_domains(self):
        self.domains_changed.emit(self.domain_keys())
```

- [ ] **Step 4: Run to verify it passes**

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH="$PWD" TensorSpec_env/bin/python -m pytest tests/test_ml_session.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tensorspec/gui/ml_session.py tests/test_ml_session.py
git commit -m "feat(gui): add MLSession to carry ML suite state between panels"
```

---

### Task 3: Shared layout helpers

**Files:**
- Create: `tensorspec/gui/components/ml_tabs/__init__.py`, `tensorspec/gui/components/ml_tabs/layout.py`
- Modify: `tensorspec/gui/maestroai/maestroai_gui.py:188-241` (delete `_scrollable`, `_split_tab`, `_tab_group`) and their call sites at lines 318, 370, 445, 498, 533, 590, 646, 686

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `scrollable(widget: QWidget) -> QScrollArea`
  - `split_panel(controls: QWidget, canvas: QWidget, sizes=(300, 460)) -> QWidget`
  - `tab_group(pages: list[tuple[str, QWidget]]) -> QWidget`

- [ ] **Step 1: Create the package init**

```python
"""Panel widgets for the Machine Learning suite, one module per tab."""
```

- [ ] **Step 2: Write `layout.py` by moving the three helpers verbatim**

Move the bodies of `MaestroAIApp._scrollable`, `_split_tab` and `_tab_group` (currently lines 188-241) into module-level functions. Drop the `self`/`cls` parameters and the `cls._scrollable` call becomes a direct `scrollable` call.

```python
"""Layout helpers shared by the ML suite panels.

Every panel puts its controls in a scroll area above its canvas, split by a
drag handle, so the canvas keeps its height when the controls are tall.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget


def scrollable(widget):
    """Wrap a control column so it can shrink below its natural height."""
    scroll = QScrollArea()
    scroll.setWidget(widget)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    return scroll


def split_panel(controls, canvas, sizes=(300, 460)):
    """Scrollable controls above a canvas, divided by a drag handle.

    The canvas takes the stretch so it absorbs extra height, and the controls
    stay reachable via the scroll area when the pane is short.
    """
    # Inset the canvas to line up with the control column, which the scroll
    # area indents by its own layout margin.
    canvas_holder = QWidget()
    holder_layout = QVBoxLayout(canvas_holder)
    holder_layout.setContentsMargins(9, 0, 9, 9)
    holder_layout.addWidget(canvas)

    splitter = QSplitter(Qt.Orientation.Vertical)
    splitter.addWidget(scrollable(controls))
    splitter.addWidget(canvas_holder)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setCollapsible(1, False)
    splitter.setSizes(list(sizes))

    page = QWidget()
    page_layout = QVBoxLayout(page)
    page_layout.setContentsMargins(0, 0, 0, 0)
    page_layout.addWidget(splitter)
    return page


def tab_group(pages):
    """Build one top-level group's page.

    A group holding a single tab renders that tab directly, so we don't draw a
    nested bar with only one entry in it.
    """
    if len(pages) == 1:
        return pages[0][1]

    inner = QTabWidget()
    inner.setDocumentMode(True)
    inner.setUsesScrollButtons(True)
    inner.setElideMode(Qt.TextElideMode.ElideNone)
    for label, page in pages:
        inner.addTab(page, label)
    return inner
```

- [ ] **Step 3: Point `maestroai_gui.py` at the helpers**

Delete the three methods and add `from tensorspec.gui.components.ml_tabs.layout import scrollable, split_panel, tab_group` at the top. Replace `self._scrollable(` -> `scrollable(`, `self._split_tab(` -> `split_panel(`, `self._tab_group(` -> `tab_group(`.

- [ ] **Step 4: Run the layout test**

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/mplcache PYTHONPATH="$PWD" \
  TensorSpec_env/bin/python -m pytest tests/test_ml_suite_layout.py -v
```

Expected: all PASS, unchanged.

- [ ] **Step 5: Commit**

```bash
git add tensorspec/gui/components/ml_tabs tensorspec/gui/maestroai/maestroai_gui.py
git commit -m "refactor(gui): move ML tab layout helpers into a shared module"
```

---

### Tasks 4-10: Extract the panels

All seven extraction tasks follow the same shape. **Do them in this order** — smallest and least coupled first, so the pattern is proven before the hard ones:

| Task | Panel class | Module | UI source lines | Logic source lines | Session signals it uses |
|---|---|---|---|---|---|
| 4 | `ActiveLearningPanel` | `active_learning_panel.py` | 514-548 | 1395-1428, 721-722 | listens `domains_changed` |
| 5 | `SimulateALPanel` | `simulate_al_panel.py` | 549-607 | 1429-1514 | listens `domains_changed` |
| 6 | `AlignmentPanel` | `alignment_panel.py` | 608-659 | 1515-1596 | listens `workspace_changed` |
| 7 | `SupervisedPanel` | `supervised_panel.py` | 462-513 | 1246-1394, 719-720 | listens `data_activated` |
| 8 | `SSLTrainingPanel` | `ssl_panel.py` | 320-387 | 964-1034, 715-716 | emits via `notify_embeddings` |
| 9 | `ClusterPanel` | `cluster_panel.py` | 388-461 | 1035-1245, 717-718 | listens `embeddings_changed`, emits via `notify_domains` |
| 10 | `DataBrowserPanel` | `data_browser_panel.py` | 246-294 | 724-963, 713-714 | emits all four |

**Interfaces (identical for all seven):**
- Consumes: `MLSession` from Task 2; `scrollable`, `split_panel` from Task 3.
- Produces: `PanelClass(session: MLSession, parent=None) -> QWidget`. The panel **is** the tab page — it builds its own internal `split_panel` layout. No `page()` accessor.

**Per-task steps (repeat for each of Tasks 4-10):**

- [ ] **Step 1: Write the failing panel test**

Add to `tests/test_ml_panels.py`. Example for Task 4 (`ActiveLearningPanel`) — write the equivalent for each panel, asserting on that panel's own combo:

```python
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "QtAgg")

from PySide6.QtWidgets import QApplication

from tensorspec.gui.ml_session import MLSession


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session():
    return MLSession()


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
```

- [ ] **Step 2: Run to verify it fails**

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/mplcache PYTHONPATH="$PWD" \
  TensorSpec_env/bin/python -m pytest tests/test_ml_panels.py -v
```

Expected: FAIL with `ModuleNotFoundError` for the new panel module.

- [ ] **Step 3: Create the panel module**

Move the UI construction lines and the logic method bodies listed in the table above into the new class. Mechanical rules:

- Class skeleton:

```python
class ActiveLearningPanel(QWidget):
    """Active-learning controls plus its prediction/uncertainty canvas."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._build()
        self.session.domains_changed.connect(self.set_domain_keys)

    def _build(self):
        controls = QWidget()
        al_layout = QVBoxLayout(controls)
        # ... moved from maestroai_gui.py lines 514-548 ...
        al_layout.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split_panel(controls, self.al_canvas, sizes=(260, 540)))

    def set_domain_keys(self, keys):
        """Repopulate the domain combo; replaces the write from activate_data."""
        self.combo_gp_domain.blockSignals(True)
        self.combo_gp_domain.clear()
        self.combo_gp_domain.addItems(keys)
        self.combo_gp_domain.blockSignals(False)
```

- Rewrite shared-state access exactly as follows:

| Old | New |
|---|---|
| `self.status.showMessage(msg)` | `self.session.set_status(0, msg)` |
| `self.prog_bar.setValue(v)` / `setVisible` | `self.session.set_status(v, msg)` |
| `self.current_view_data` | `self.session.current_view_data` |
| `self.workspace` | `self.session.workspace` |
| `self.current_folder` | `self.session.current_folder` |
| `self.viewer` | `self.session.viewer` |
| `self` as a `QDialog`/`QFileDialog` parent | `self` (unchanged — the panel is a widget) |

- Cross-tab writes are **deleted**, not moved. Replace them with the session call:
  - in `on_cluster_finish` (Task 9): delete the `combo_gp_domain` / `combo_sim_domain` writes, call `self.session.notify_domains()` instead.
  - in `on_train_finish` (Task 8): delete the `combo_embed` writes, call `self.session.notify_embeddings()`.
  - in `activate_data` / `load_session` / `on_load_finish` (Task 10): delete all four foreign combo writes; call `self.session.activate(data)`, then `notify_embeddings()` and `notify_domains()`.

- [ ] **Step 4: Have `MaestroAIApp` use the new panel**

Delete the moved UI lines and logic methods from `maestroai_gui.py`. In `init_ui`, replace the removed block with:

```python
self.al_panel = ActiveLearningPanel(self.session)
al_page = self.al_panel
```

For Tasks 4-9, `MaestroAIApp` needs an `MLSession` — add `self.session = MLSession()` in `__init__` during Task 4 and connect it to the still-inline code:

```python
self.session.status_changed.connect(
    lambda v, m: (self.prog_bar.setValue(v), self.status.showMessage(m)))
```

Remove that shim in Task 11 when the suite grows a real status row.

- [ ] **Step 5: Run both test files**

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/mplcache PYTHONPATH="$PWD" \
  TensorSpec_env/bin/python -m pytest tests/test_ml_panels.py tests/test_ml_suite_layout.py -v
```

Expected: all PASS. The layout test proves the extraction did not change geometry.

- [ ] **Step 6: Launch-smoke the app**

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/mplcache PYTHONPATH="$PWD" \
  TensorSpec_env/bin/python -c "
from PySide6.QtWidgets import QApplication
app = QApplication([])
from tensorspec.gui.main_browser import TensorSpecMainBrowser
b = TensorSpecMainBrowser(); b.launch_ml_suite()
print('ML suite launches OK')"
```

Expected: prints `ML suite launches OK` with no traceback.

- [ ] **Step 7: Commit**

```bash
git add tensorspec/gui/components/ml_tabs tensorspec/gui/maestroai/maestroai_gui.py tests/test_ml_panels.py
git commit -m "refactor(gui): extract the <panel name> panel from the ML monolith"
```

---

### Task 11: The `MLSuite` shell

**Files:**
- Create: `tensorspec/gui/suites/ml_suite.py`
- Modify: `tests/test_ml_suite_layout.py` (point the `suite` fixture at `MLSuite`)

**Interfaces:**
- Consumes: `MLSession`, all seven panel classes, `tab_group`.
- Produces: `MLSuite(parent=None) -> QWidget` with attribute `session: MLSession`.

- [ ] **Step 1: Update the layout test fixture to the new class**

```python
@pytest.fixture
def suite(qapp):
    from tensorspec.gui.suites.ml_suite import MLSuite
    widget = MLSuite()
    widget.resize(1500, 900)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()
```

Also replace `suite.centralWidget().findChild(...)` with `suite.findChild(...)` in every test, since `MLSuite` is a `QWidget` and has no central widget.

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL with `ModuleNotFoundError: No module named 'tensorspec.gui.suites.ml_suite'`.

- [ ] **Step 3: Write the shell**

`MLSuite` is a `QWidget` (matching `DFTSuite`, `ARPESSuite`, `CrystalViewerSuite`). It owns the 3-pane splitter, the grouped tabs, and a status row replacing the `QStatusBar`:

```python
class MLSuite(QWidget):
    """Machine Learning suite: data browser, shared N-D viewer, ML panels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session = MLSession()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.viewer = DataViewerPanel()
        self.session.viewer = self.viewer

        self.browser_panel = DataBrowserPanel(self.session)
        self.ssl_panel = SSLTrainingPanel(self.session)
        self.cluster_panel = ClusterPanel(self.session)
        self.supervised_panel = SupervisedPanel(self.session)
        self.al_panel = ActiveLearningPanel(self.session)
        self.sim_panel = SimulateALPanel(self.session)
        self.alignment_panel = AlignmentPanel(self.session)
        self.diagnostics_tab = DiagnosticsTab(self)

        mid_panel = QWidget()
        mid_layout = QVBoxLayout(mid_panel)
        mid_layout.addWidget(self.viewer)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setMovable(True)
        tabs.setUsesScrollButtons(True)
        tabs.setElideMode(Qt.TextElideMode.ElideNone)
        for group_label, pages in (
            ("Train", [("SSL Training", self.ssl_panel),
                       ("Supervised Learning", self.supervised_panel)]),
            ("Cluster", [("Clustering", self.cluster_panel)]),
            ("Align", [("3D Alignment", self.alignment_panel)]),
            ("Steer", [("Active Learning", self.al_panel),
                       ("Simulate AL", self.sim_panel)]),
            ("Models", self._model_pages()),
            ("System", [("Diagnostics", self.diagnostics_tab)]),
        ):
            tabs.addTab(tab_group(pages), group_label)
        tabs.setCurrentIndex(0)

        self.browser_panel.setMinimumWidth(220)
        mid_panel.setMinimumWidth(320)
        tabs.setMinimumWidth(380)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.browser_panel)
        splitter.addWidget(mid_panel)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setCollapsible(1, False)
        splitter.setSizes([260, 680, 560])
        layout.addWidget(splitter, 1)

        # Status row replaces the QMainWindow status bar the monolith used.
        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.prog_bar = QProgressBar()
        self.prog_bar.setVisible(False)
        self.prog_bar.setMaximumWidth(220)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.prog_bar)
        layout.addLayout(status_row)

        self.session.status_changed.connect(self._on_status)

    def _on_status(self, value, message):
        if message:
            self.status_label.setText(message)
        self.prog_bar.setVisible(0 < value < 100)
        self.prog_bar.setValue(value)
```

- [ ] **Step 4: Run the full ML test suite**

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/mplcache PYTHONPATH="$PWD" \
  TensorSpec_env/bin/python -m pytest tests/test_ml_session.py tests/test_ml_panels.py tests/test_ml_suite_layout.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tensorspec/gui/suites/ml_suite.py tests/test_ml_suite_layout.py
git commit -m "feat(gui): add MLSuite shell matching the other suite panels"
```

---

### Task 12: Switch the launcher and retire the monolith

**Files:**
- Modify: `tensorspec/gui/main_browser.py:526-543`
- Delete: `tensorspec/gui/maestroai/maestroai_gui.py`
- Move: `DiagnosticsTab` into `tensorspec/gui/components/ml_tabs/diagnostics_panel.py` (it is the last class left in the monolith)

- [ ] **Step 1: Move `DiagnosticsTab`**

Move lines 22-147 into `components/ml_tabs/diagnostics_panel.py` unchanged, except import `MplCanvas` from `tensorspec.gui.maestroai.maestroai_viewers`. Update the `MLSuite` import.

- [ ] **Step 2: Rewrite `launch_ml_suite` to match `launch_dft_suite`**

```python
    def launch_ml_suite(self):
        win_id = "ML Suite"
        if win_id in self.active_windows:
            items = self.window_tracker_list.findItems(win_id, Qt.MatchExactly)
            if items: self.bring_window_to_front(items[0])
            return

        try:
            from tensorspec.gui.suites.ml_suite import MLSuite
            ml_widget = MLSuite()
            wrapper = FloatingViewerWindow(
                win_id=win_id, title="Machine Learning Suite",
                inner_widget=ml_widget, parent=self)
            wrapper.window_closed.connect(self.unregister_window)
            wrapper.resize(1500, 900)
            self.active_windows[win_id] = wrapper
            self.window_tracker_list.addItem(win_id)
            wrapper.show()
        except Exception as e:
            print(f"Failed to launch ML Suite: {e}")
```

- [ ] **Step 3: Delete the monolith and check nothing imports it**

```bash
git rm tensorspec/gui/maestroai/maestroai_gui.py
rg -n "maestroai_gui|MaestroAIApp" --glob '!docs/**' . && echo "STILL REFERENCED - fix before continuing" || echo "no references remain"
```

Expected: `no references remain`.

- [ ] **Step 4: Run the whole test suite, not just the ML files**

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/mplcache PYTHONPATH="$PWD" \
  TensorSpec_env/bin/python -m pytest tests/ -v
```

Expected: all PASS, including the three pre-existing test files.

- [ ] **Step 5: Launch-smoke twice to confirm window reuse still works**

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/mplcache PYTHONPATH="$PWD" \
  TensorSpec_env/bin/python -c "
from PySide6.QtWidgets import QApplication
app = QApplication([])
from tensorspec.gui.main_browser import TensorSpecMainBrowser
b = TensorSpecMainBrowser()
b.launch_ml_suite(); first = b.active_windows['ML Suite']
b.launch_ml_suite(); assert b.active_windows['ML Suite'] is first
assert b.window_tracker_list.count() == 1
print('launcher reuse OK')"
```

- [ ] **Step 6: Commit**

```bash
git add -A tensorspec/gui
git commit -m "refactor(gui): retire the maestroai_gui monolith for MLSuite"
```

---

### Task 13: Decide the Diagnostics tab

The main browser's `TelemetryWindow` (`main_browser.py:30-147`) tracks per-core CPU and system RAM with a record/save-graph workflow. `DiagnosticsTab` tracks system RAM **and this process's own RSS** on a dual axis, with delta-triggered sampling tuned for multi-hour runs.

The overlap is system RAM only. The process-RSS trace is the ML-relevant part and `TelemetryWindow` cannot show it.

- [ ] **Step 1: Add process RSS to `TelemetryWindow`**

In `TelemetryWindow.__init__`, add `self.process = psutil.Process(os.getpid())` and a third history list `self.history_rss`. In `update_metrics`, append `self.process.memory_info().rss / (1024 ** 3)` and plot it on a twin axis of `self.ax_ram`.

- [ ] **Step 2: Bound the telemetry history**

`history_time`, `history_cpu` and `history_ram` are plain lists appended once per second with no cap, so a long recording grows without bound. Convert all four to `collections.deque(maxlen=5000)`, matching `DiagnosticsTab.max_history`.

- [ ] **Step 3: Delete the Diagnostics panel and the System group**

Remove `components/ml_tabs/diagnostics_panel.py`, drop the `("System", ...)` entry from the `MLSuite` group tuple, and update `EXPECTED_GROUPS` in `tests/test_ml_suite_layout.py` to the five remaining groups.

- [ ] **Step 4: Run the tests**

```bash
QT_QPA_PLATFORM=offscreen MPLCONFIGDIR=/tmp/mplcache PYTHONPATH="$PWD" \
  TensorSpec_env/bin/python -m pytest tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git add -A tensorspec/gui tests
git commit -m "refactor(gui): fold ML process memory tracking into the telemetry window"
```

---

## Rollback

Every task ends at a green test run and a commit, so `git revert <sha>` undoes any single step. The pre-refactor baseline is commit `2a2e7aa`.

## Self-Review Notes

- **Coverage:** Tasks 4-10 cover all seven UI regions (lines 246-659) and all logic methods (lines 713-1596) of `MaestroAIApp`. Task 12 covers `DiagnosticsTab` (lines 22-147). Nothing in the 1596-line file is unaccounted for.
- **Cross-tab coupling:** all five cross-tab methods identified by the AST analysis are explicitly reassigned — `activate_data`, `load_session`, `on_load_finish` in Task 10; `on_train_finish` in Task 8; `on_cluster_finish` in Task 9.
- **Type consistency:** the `MLSession` method names used in Tasks 4-12 (`set_status`, `activate`, `add_dataset`, `notify_embeddings`, `notify_domains`, `embedding_keys`, `domain_keys`) all match the Task 2 implementation. Signal signatures match their `connect` sites.
- **Viewer API verified:** the six `DataViewerPanel` methods the panels call all exist — `get_dispersion_contrast` (:888), `get_slider_values` (:885), `add_overlay_mode` (:891), `get_current_coords` (:882), `load_data` (:688). The one exception, `set_data`, does not exist and is fixed in Task 0.
- **Dead code to delete in Task 10:** `load_workspace_to_viewer` (lines 765-772) has no callers anywhere in the repo and reads `self.current_workspace`, an attribute that is never assigned. Delete it rather than porting it into `DataBrowserPanel`.
- **Deferred:** the ML workers under `tensorspec/gui/maestroai/` stay put. Moving them to `tensorspec/core/ml/` would match the engine layout used by the DFT and ARPES suites, but it is orthogonal to this plan and adds import churn.
