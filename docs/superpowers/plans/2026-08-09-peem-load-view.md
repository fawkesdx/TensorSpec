# PEEM Load + View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship PEEM Suite load+view: multipage TIF / folder sequences via upload or server path, auto-search/prompt/later-attach beamline CSV (I0), simple canvas+slider viewer; no drift/BG/sum-rule yet.

**Architecture:** `peem_loaders.py` builds `TensorData(frame,y,x)` + metadata → `push_spectroscopy_data`. FastAPI `peem` router for load / meta / frame / attach-csv. Thin `peem_suite.js` draws frames on canvas. Pairing/drift/BG stay disabled.

**Tech Stack:** Python, NumPy, `tifffile`, pandas (CSV), FastAPI, vanilla JS canvas, existing session workspace / DataTree.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-09-peem-load-view-design.md`
- Branch: `HTML_einstein_app` only — never merge to `main`
- No drift, CP/CM stack/separate, BG, sum-rule math, or I0 **application** (only store I0)
- XAS suite unchanged
- DataTree via `push_spectroscopy_data` (do not invent a parallel store)
- Path safety: reject escapes outside allowed roots
- After ship: push + Einstein `git pull` (uvicorn reload if needed)
- Update `roadmap.md` / README when features land

## File map

| File | Role |
|------|------|
| `tensorspec/core/io/peem_loaders.py` | TIF stack/sequence, CSV find/parse, pol tags → `TensorData` |
| `tensorspec/core/data_tree.py` | `merge_raw_attrs` for attach-csv without wiping intensity |
| `tensorspec/core/workspace.py` | Thin wrapper `merge_spectroscopy_raw_attrs` |
| `tensorspec/web/server/schemas.py` | `PeemLoadSummary`, frame response models |
| `tensorspec/web/server/routers/peem.py` | load / meta / frame / attach-csv |
| `tensorspec/web/server/app.py` | `include_router(peem)` |
| `requirements.txt` | add `tifffile` |
| `tensorspec/web/static/js/api.js` | `loadPeem`, `peemMeta`, `peemFrame`, `peemAttachCsv` |
| `tensorspec/web/static/js/peem_suite.js` | Load UX + canvas viewer |
| `tensorspec/web/templates/suites/peem_suite.html` | Enable load/view controls |
| `tests/test_peem_loaders.py` | Loader unit tests |
| `tests/test_peem_api.py` | Router / session tests |
| `roadmap.md` / `README.md` | Mark loaders + load+view |

---

### Task 1: `peem_loaders.py` + unit tests

**Files:**
- Create: `tensorspec/core/io/peem_loaders.py`
- Create: `tests/test_peem_loaders.py`
- Modify: `requirements.txt` (add `tifffile` near imaging deps, e.g. after `pillow`)

**Interfaces:**
- Produces:

```python
# tensorspec/core/io/peem_loaders.py
from pathlib import Path
from typing import Any
import numpy as np
from tensorspec.core.data_models import TensorData

I0_COLUMN_ALIASES = ("I0", "I_0", "i0", "beam_current", "BeamCurrent", "current", "I0_nA")

def infer_pol_from_name(name: str) -> str:
    """Return 'CP'|'CM'|'LH'|'LV'|'unknown' via case-insensitive substring."""
    ...

def find_beamline_csv(directory: Path, preferred_stem: str | None = None) -> list[Path]:
    """
    Return candidate *.csv paths in directory (non-recursive).
    If preferred_stem set, sort stem-matching paths first.
    Empty list if none.
    """
    ...

def load_beamline_csv(path: Path | str) -> dict[str, Any]:
    """
    Parse CSV with pandas. Return JSON-serializable dict:
      columns: list[str]
      rows: list[dict]  # optional if huge — prefer columnar:
      series: dict[str, list]  # column -> values
      I0: float | list[float] | None  # first matching alias; scalar if len==1 else list
      beam_current: same as I0 when alias matches current*
    """
    ...

def load_tif_stack(path: Path | str, *, csv_path: Path | str | None = None) -> TensorData:
    """Multipage TIF → TensorData shape (n_frames, ny, nx), float64."""
    ...

def load_tif_sequence(directory: Path | str, *, csv_path: Path | str | None = None) -> TensorData:
    """Sorted *.tif/*.tiff in directory → TensorData (n_frames, ny, nx)."""
    ...

def package_stack(
    frames: np.ndarray,
    frame_names: list[str],
    *,
    source: str,
    loader: str,
    csv_meta: dict[str, Any] | None = None,
) -> TensorData:
    """Shared builder: axes, labels, pol tags, merge csv_meta into metadata."""
    ...
```

- Consumes: `tifffile`, `pandas`, `numpy`, `TensorData`

- [ ] **Step 1: Add dependency**

Append to `requirements.txt`:

```
tifffile>=2024.8.30
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_peem_loaders.py
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tensorspec.core.io import peem_loaders as pl


def _write_multipage_tif(path: Path, stacks: list[np.ndarray]) -> None:
    import tifffile
    tifffile.imwrite(path, np.stack(stacks, axis=0))


class TestPeemLoaders(unittest.TestCase):
    def test_infer_pol(self):
        self.assertEqual(pl.infer_pol_from_name("sample_CP_001.tif"), "CP")
        self.assertEqual(pl.infer_pol_from_name("x_cm_y.tif"), "CM")
        self.assertEqual(pl.infer_pol_from_name("plain.tif"), "unknown")

    def test_load_tif_stack_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = np.arange(6, dtype=np.uint16).reshape(2, 3)
            b = (a + 10).astype(np.uint16)
            p = root / "stack.tif"
            _write_multipage_tif(p, [a, b])
            td = pl.load_tif_stack(p)
            self.assertEqual(td.value.shape, (2, 2, 3))
            self.assertEqual(td.labels, ["frame", "y", "x"])
            self.assertEqual(td.data_type, "Experimental PEEM")
            self.assertEqual(td.metadata["loader"], "tif_stack")

    def test_load_tif_sequence_and_csv_auto(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, val in enumerate([1, 2]):
                import tifffile
                tifffile.imwrite(root / f"f{i}_CP.tif", np.full((2, 2), val, dtype=np.uint16))
            (root / "run.csv").write_text("frame,I0\n0,1.5\n1,1.7\n")
            found = pl.find_beamline_csv(root)
            self.assertEqual(len(found), 1)
            td = pl.load_tif_sequence(root, csv_path=found[0])
            self.assertEqual(td.value.shape, (2, 2, 2))
            self.assertEqual(td.metadata["pol"], ["CP", "CP"])
            self.assertTrue(td.metadata.get("csv_attached"))
            self.assertIsNotNone(td.metadata.get("I0"))

    def test_load_without_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            import tifffile
            tifffile.imwrite(root / "a.tif", np.ones((2, 2), dtype=np.uint16))
            td = pl.load_tif_sequence(root)
            self.assertFalse(td.metadata.get("csv_attached", True))
            self.assertIsNone(td.metadata.get("I0"))
```

- [ ] **Step 3: Run tests — expect FAIL**

```bash
pytest tests/test_peem_loaders.py -v
```

Expected: import error or missing module.

- [ ] **Step 4: Implement `peem_loaders.py`**

Minimal behavior notes:
- Use `tifffile.imread`; if result is 2D, wrap to `(1,y,x)`; if `(n,y,x)` keep; if TIFF pages differ, raise `ValueError`.
- Cast to `float64`.
- Axes: `frame=np.arange(n)`, `y=np.arange(ny)`, `x=np.arange(nx)`.
- `frame_names` from page names if available else `f"frame_{i}"` / file stems for sequence.
- Sort sequence with `sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))` (casefold unique).
- CSV: `pandas.read_csv`; pick I0 via aliases (case-insensitive column match).
- All metadata values must be JSON-friendly (lists, not `np.ndarray`).

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/test_peem_loaders.py -v
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tensorspec/core/io/peem_loaders.py tests/test_peem_loaders.py
git commit -m "feat(peem): TIF loaders and beamline CSV parse"
```

---

### Task 2: DataTree attr merge + PEEM API

**Files:**
- Modify: `tensorspec/core/data_tree.py`
- Modify: `tensorspec/core/workspace.py`
- Modify: `tensorspec/web/server/schemas.py`
- Create: `tensorspec/web/server/routers/peem.py`
- Modify: `tensorspec/web/server/app.py`
- Create: `tests/test_peem_api.py`

**Interfaces:**
- Produces:

```python
# DataTreeBuilder
@staticmethod
def merge_raw_attrs(tree: DataTree, attrs: dict) -> DataTree:
    """Update /raw dataset attrs in place (copy-on-write OK); return tree."""
    ...

# WorkspaceManager
def merge_spectroscopy_raw_attrs(self, name: str, attrs: dict) -> bool:
    ...

# peem router prefix /api/peem
# POST /load  → PeemLoadSummary
# POST /{name}/attach-csv → PeemLoadSummary (or small PeemCsvSummary)
# GET  /{name}/meta → PeemMeta
# GET  /{name}/frame/{i} → PeemFrame
```

```python
# schemas.py
class PeemLoadSummary(BaseModel):
    name: str
    shape: list[int]
    n_frames: int
    data_type: str
    pol_summary: dict[str, int]
    source: str
    loader: str
    csv_attached: bool
    I0_present: bool
    csv_prompt: bool = False
    csv_candidates: list[str] = []

class PeemMeta(BaseModel):
    name: str
    shape: list[int]
    labels: list[str]
    n_frames: int
    frame_names: list[str]
    pol: list[str]
    csv_attached: bool
    I0_present: bool
    I0: float | list[float] | None = None

class PeemFrame(BaseModel):
    index: int
    shape: list[int]  # [ny, nx]
    intensity: list[list[float]]  # 2D nested lists
    vmin: float
    vmax: float
    pol: str | None = None
    frame_name: str | None = None
```

**Path safety helper (in `peem.py`):**

```python
def _resolve_allowed(path: Path, session: Session) -> Path:
    resolved = path.expanduser().resolve()
    roots = [Path(session.workspace.project_dir).resolve()]
    # Optional: Path.home() / "TensorSpec" style — keep at least project_dir;
    # also allow absolute paths under Path.home() for Einstein beamline folders:
    roots.append(Path.home().resolve())
    if not any(resolved == r or r in resolved.parents for r in roots):
        raise HTTPException(403, detail="Path outside allowed roots.")
    if not resolved.exists():
        raise HTTPException(404, detail=f"Not found: {resolved}")
    return resolved
```

**Load behavior:**
- Form fields: `file` (optional UploadFile), `server_path` (optional str), `csv` (optional), `csv_path` (optional), `name` (optional).
- Exactly one of `file` or `server_path`.
- Upload TIF → `uploads/peem/`; zip → extract to `uploads/peem/<stem>/` then `load_tif_sequence`.
- If `csv`/`csv_path` given → use it.
- Else auto `find_beamline_csv` on load directory; if exactly one (or clear stem match) attach; if multiple → load images, set `csv_prompt=True`, `csv_candidates=[...]`, `csv_attached=False`; if zero → same with empty candidates.
- Limits: `MAX_PEEM_BYTES = 512 * 1024 * 1024` (file/zip); CSV `8 * 1024 * 1024`.

**Frame clim:** `vmin`/`vmax` = 1st / 99th percentile of frame (or min/max if flat).

**attach-csv:** parse CSV → `merge_spectroscopy_raw_attrs` with keys `csv_attached=True`, `beamline_csv`, `I0`, `beamline_table` / `series`, etc.

- Consumes: Task 1 loaders; `push_spectroscopy_data`; `pull_tensor_data`

- [ ] **Step 1: Write failing API tests**

```python
# tests/test_peem_api.py
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from fastapi import HTTPException

from tensorspec.core.workspace import WorkspaceManager
from tensorspec.web.server.routers import peem as peem_router
from tensorspec.web.server.session import Session


class TestPeemApi(unittest.TestCase):
    def _session(self, tmp: str) -> Session:
        return Session(session_id="t", workspace=WorkspaceManager(project_dir=Path(tmp)))

    def test_load_server_path_with_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            tifffile.imwrite(root / "a_CP.tif", np.ones((4, 4), dtype=np.uint16))
            tifffile.imwrite(root / "b_CM.tif", np.full((4, 4), 2, dtype=np.uint16))
            (root / "run.csv").write_text("I0\n1.1\n1.2\n")
            summary = peem_router.load_peem(
                file=None,
                server_path=str(root),
                csv=None,
                csv_path=None,
                name="peem1",
                session=session,
            )
            # Implement load_peem as the core function TestClient can also hit
            self.assertEqual(summary.n_frames, 2)
            self.assertTrue(summary.csv_attached)
            self.assertTrue(summary.I0_present)
            meta = peem_router.get_meta("peem1", session=session)
            self.assertEqual(meta.pol, ["CP", "CM"])
            frame = peem_router.get_frame("peem1", 0, session=session)
            self.assertEqual(frame.shape, [4, 4])
            self.assertEqual(len(frame.intensity), 4)

    def test_attach_csv_after(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            root = Path(tmp) / "run"
            root.mkdir()
            tifffile.imwrite(root / "a.tif", np.ones((2, 2), dtype=np.uint16))
            summary = peem_router.load_peem(
                file=None, server_path=str(root), csv=None, csv_path=None,
                name="peem2", session=session,
            )
            self.assertFalse(summary.csv_attached)
            csv_path = root / "late.csv"
            csv_path.write_text("I0\n3.3\n")
            peem_router.attach_csv("peem2", csv=None, csv_path=str(csv_path), session=session)
            meta = peem_router.get_meta("peem2", session=session)
            self.assertTrue(meta.csv_attached)
            self.assertTrue(meta.I0_present)

    def test_path_escape_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(tmp)
            with self.assertRaises(HTTPException) as ctx:
                peem_router.load_peem(
                    file=None,
                    server_path="/etc/passwd",
                    csv=None, csv_path=None, name="x", session=session,
                )
            self.assertIn(ctx.exception.status_code, (403, 404))
```

Wire FastAPI routes to call the same functions (or make route handlers thin and test handlers with `session=` like crystal figure tests).

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/test_peem_api.py -v
```

- [ ] **Step 3: Implement merge + router + register**

In `app.py`:

```python
from tensorspec.web.server.routers import peem as peem_router
...
app.include_router(peem_router.router)
```

Router:

```python
router = APIRouter(prefix="/api/peem", tags=["peem"])
```

Implement `merge_raw_attrs` so attach does not re-encode the full intensity cube unnecessarily: update `tree["raw"].ds.attrs` (or equivalent DataTree assign) and store back on workspace.

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/test_peem_loaders.py tests/test_peem_api.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/core/data_tree.py tensorspec/core/workspace.py \
  tensorspec/web/server/schemas.py tensorspec/web/server/routers/peem.py \
  tensorspec/web/server/app.py tests/test_peem_api.py
git commit -m "feat(peem): load/meta/frame/attach-csv API"
```

---

### Task 3: PEEM Suite UI (load + canvas)

**Files:**
- Modify: `tensorspec/web/static/js/api.js`
- Create: `tensorspec/web/static/js/peem_suite.js`
- Modify: `tensorspec/web/templates/suites/peem_suite.html`

**Interfaces:**
- Produces API helpers:

```javascript
loadPeem: ({ file, serverPath, csvFile, csvPath, name } = {}) => { /* FormData → upload("/api/peem/load", form) */ },
peemMeta: (name) => request(`/api/peem/${encodeURIComponent(name)}/meta`),
peemFrame: (name, i) => request(`/api/peem/${encodeURIComponent(name)}/frame/${i}`),
peemAttachCsv: (name, { csvFile, csvPath } = {}) => { /* FormData → upload attach-csv */ },
```

**UI behavior:**
- Badge → `Load + view`
- Enable: Load TIF stack (file input `.tif/.tiff`), Load folder path (text + Load button), optional zip file input for folder upload.
- Hidden/secondary: CSV file input used by prompt + Attach CSV button.
- On load response: if `csv_attached`, status “CSV + I0 OK”; if `csv_prompt` or `!csv_attached`, show panel: “No beamline CSV found. Choose CSV, or Continue without.” Candidates listed if present.
- Attach / update CSV always available when a dataset name is set.
- Viewer: `<canvas id="peem-canvas">`, range `#peem-frame`, number inputs clim; draw grayscale from `intensity` mapped through clim.
- Keep pairing/drift/BG fieldsets disabled; update planned list note that loaders are live.
- Footer: “Load + view” not “Engine not implemented”.
- Script tag: `<script type="module" src="../../static/js/peem_suite.js"></script>` (match other suites’ pattern — if they use classic scripts, match that instead).

Check `arpes_suite.html` / `crystal_suite.html` for script include style and copy that.

- [ ] **Step 1: Wire `api.js` helpers** (near `loadArpes`)

- [ ] **Step 2: Rewrite load + viewer section of `peem_suite.html`**

Replace disabled load buttons with working controls; replace glyph canvas with real `<canvas>`; add frame slider + clim + CSV prompt panel + Attach CSV.

- [ ] **Step 3: Implement `peem_suite.js`**

State: `{ name, nFrames, frameIndex, vmin, vmax }`.  
`async function showFrame(i)` → `peemFrame` → putImageData / fillRect loop (for large frames prefer ImageData).  
Clamp clim; re-draw on slider/clim change.

- [ ] **Step 4: Manual smoke (local)**

```bash
# from repo root, with TensorSpec_env or venv that has deps
pip install tifffile   # if needed
uvicorn tensorspec.web.server.app:app --reload --port 8000
# open /suites/peem_suite.html — load a test folder with TIFs + CSV
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/web/static/js/api.js tensorspec/web/static/js/peem_suite.js \
  tensorspec/web/templates/suites/peem_suite.html
git commit -m "feat(peem): suite load UI and frame canvas"
```

---

### Task 4: Docs, deploy Einstein

**Files:**
- Modify: `roadmap.md` (check PEEM loader / CSV auto-search / load+view bullets that this slice completes; leave drift/BG/sum-rule open)
- Modify: `README.md` if it lists suites / key features
- Optional: one-line note in spec Status → implemented (or leave Status and rely on roadmap)

- [ ] **Step 1: Update roadmap checkboxes** for:
  - loader of tif stacks
  - loader of folder sequences
  - accompanying CSV auto-search / prompt / load-without / later attach (load+view parts)
  - Do **not** check sum-rule I0 apply, stack pairs, drift, BG

- [ ] **Step 2: README Key Features** — short PEEM load+view bullet if README mentions suite list

- [ ] **Step 3: Commit**

```bash
git add roadmap.md README.md
git commit -m "docs: mark PEEM load+view shipped on roadmap"
```

- [ ] **Step 4: Push + Einstein pull**

```bash
git push -u origin HEAD
ssh einstein 'cd ~/TensorSpec && git fetch && git checkout HTML_einstein_app && git pull && (pip install -q tifffile || TensorSpec_env/bin/pip install -q tifffile); pgrep -af uvicorn || true'
# restart uvicorn if not --reload:
# cd ~/TensorSpec && nohup TensorSpec_env/bin/uvicorn tensorspec.web.server.app:app --host 0.0.0.0 --port 8000 --reload > ~/tensorspec-uvicorn.log 2>&1 &
```

- [ ] **Step 5: Verify on Einstein** — `/api/health` and PEEM suite page loads; optional server_path smoke if a TIF folder exists on disk

---

## Spec coverage (self-review)

| Spec item | Task |
|-----------|------|
| Multipage TIF + folder sequence | 1, 2, 3 |
| Upload + server path | 2, 3 |
| Auto-search CSV → prompt → load without → attach later | 1 (`find`/`load`), 2 (`attach-csv`), 3 (UI) |
| Store I0; do not apply sum-rule norm | 1–2 (store only); non-goal honored |
| TensorData → DataTree | 2 |
| meta + frame JSON | 2, 3 |
| Simple canvas + slider + clim | 3 |
| Pairing/drift/BG disabled | 3 |
| Tests loaders + API | 1, 2 |
| Roadmap / Einstein | 4 |
| Profiles / XAS / engine | out of scope |

## Placeholder scan

No TBD steps; CSV column aliases listed; path roots defined (`project_dir` + `Path.home()`); upload limits numeric.

## Type consistency

- `PeemLoadSummary.csv_attached` / `I0_present` used in UI and tests.
- Loader metadata keys: `csv_attached`, `I0`, `pol`, `frame_names`, `loader`, `source`.
- Router functions tested: `load_peem`, `attach_csv`, `get_meta`, `get_frame`.
