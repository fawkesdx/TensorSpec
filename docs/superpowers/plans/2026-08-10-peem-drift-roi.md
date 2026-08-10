# PEEM ROI NCC Drift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Fiji-like translational drift correction: user ROI (rect/ellipse/polygon) + NCC integer shifts + edge-clamp, for raw frames and paired cubes, writing `/processed` with shift attrs and PEEM suite ROI UI.

**Architecture:** `peem_roi.roi_to_mask` builds masks. `peem_engine.drift_correct` estimates `(dx,dy)` via NCC in the ROI vs a user reference plane and applies the same shift to both channels of a pair. PEEM router `POST /drift` pulls `raw` or `processed`, writes `/processed`. Suite adds ROI drawing tools and Apply Drift. Processed node must accept **3D** (drifted raw) and **4D** (drifted paired) cubes — today `_processed_pair_tensor` is pair-only and must be generalized carefully.

**Tech Stack:** NumPy only (no OpenCV/skimage required), FastAPI, existing canvas viewer JS.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-10-peem-drift-roi-design.md`
- Branch: `HTML_einstein_app` only — never merge to `main`
- Translation only; NCC + integer pixels; edge-clamp fill
- No Separate, phase/manual algorithms, BG, sum-rule, or I0 application
- Pair always reads `/raw`; drift may overwrite `/processed`
- After ship: push + Einstein pull; update `roadmap.md` / README
- Tests: `PYTHONPATH=. TensorSpec_env/bin/pytest …`

## File map

| File | Role |
|------|------|
| `tensorspec/core/peem_roi.py` | `roi_to_mask` for rect/ellipse/polygon |
| `tensorspec/core/peem_engine.py` | `drift_correct` (+ shift apply helpers) |
| `tests/test_peem_roi.py` | Mask unit tests |
| `tests/test_peem_drift.py` | Drift unit tests (raw + paired) |
| `tensorspec/web/server/schemas.py` | `PeemRoi`, `PeemDriftRequest`, `PeemDriftSummary`; extend `PeemMeta` |
| `tensorspec/web/server/routers/peem.py` | `/drift`; generalize processed pull for 3D/4D; meta `has_drift` |
| `tests/test_peem_api.py` | API drift tests |
| `tensorspec/web/static/js/api.js` | `peemDrift` |
| `tensorspec/web/static/js/peem_suite.js` | ROI tools + Apply Drift |
| `tensorspec/web/templates/suites/peem_suite.html` | Enable drift fieldset + ROI chrome |
| `roadmap.md` / `README.md` | Mark drift shipped |

## Locked implementation details (open items from spec)

- **Rect:** integer pixels; normalize so `x0≤x1`, `y0≤y1`; include both endpoints (`x0..x1` inclusive).
- **Ellipse:** filled ` ((x-cx)/rx)^2 + ((y-cy)/ry)^2 ≤ 1 `; require `rx>0`, `ry>0`.
- **Polygon:** ≥3 points; close implicitly; fill with matplotlib-path-free even-odd scan or winding using pure NumPy (implement `_point_in_poly` / row raster). Max 256 vertices at API.
- **NCC:** use bounding box of mask; correlate only masked pixels (mean-subtracted). Peak in `dx,dy ∈ [-R..R]`. If template variance ~0 → `ValueError`.
- **Shift sign:** `out[y, x] = in[clamp(y - dy, 0, ny-1), clamp(x - dx, 0, nx-1)]` so positive `dx` moves content right.
- **Caps:** `1 ≤ search_radius ≤ 200`; stack length ≤ `MAX_PEEM_FRAMES` (reuse existing); reject if ROI bbox area < 9 pixels.
- **Processed after drift:**
  - From raw: labels stay `["frame","y","x"]`; `data_type` stay `"Experimental PEEM"` (or append note only in metadata `drift_method`).
  - From paired: keep `"Experimental PEEM (paired)"` + `["pair","channel","y","x"]`.
- **Frame/meta for processed:** support both 3D frame stack and 4D paired (see Task 3).

---

### Task 1: `peem_roi.roi_to_mask` + tests

**Files:**
- Create: `tensorspec/core/peem_roi.py`
- Create: `tests/test_peem_roi.py`

**Interfaces:**
- Produces:

```python
# tensorspec/core/peem_roi.py
def roi_to_mask(ny: int, nx: int, roi: dict) -> np.ndarray:
    """
    Return bool mask (ny, nx).
    roi["kind"] in {"rect","ellipse","polygon"}.
    Raises ValueError on invalid/empty.
    """
```

- Consumes: NumPy only

- [ ] **Step 1: Write failing tests**

```python
# tests/test_peem_roi.py
import unittest
import numpy as np
from tensorspec.core.peem_roi import roi_to_mask


class TestPeemRoi(unittest.TestCase):
    def test_rect_inclusive(self):
        m = roi_to_mask(5, 5, {"kind": "rect", "x0": 1, "y0": 1, "x1": 2, "y1": 2})
        self.assertEqual(int(m.sum()), 4)
        self.assertTrue(m[1, 1] and m[2, 2])
        self.assertFalse(m[0, 0])

    def test_rect_normalizes_order(self):
        m = roi_to_mask(5, 5, {"kind": "rect", "x0": 3, "y0": 3, "x1": 1, "y1": 1})
        self.assertEqual(int(m.sum()), 9)

    def test_ellipse_center(self):
        m = roi_to_mask(7, 7, {"kind": "ellipse", "cx": 3, "cy": 3, "rx": 2, "ry": 1})
        self.assertTrue(m[3, 3])
        self.assertFalse(m[0, 0])
        self.assertGreater(int(m.sum()), 0)

    def test_polygon_triangle(self):
        m = roi_to_mask(
            6,
            6,
            {"kind": "polygon", "points": [[1, 1], [4, 1], [1, 4]]},
        )
        self.assertTrue(m[2, 1])
        self.assertGreater(int(m.sum()), 3)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            roi_to_mask(4, 4, {"kind": "rect", "x0": 10, "y0": 10, "x1": 12, "y1": 12})
```

- [ ] **Step 2: Run — expect FAIL**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_roi.py -v
```

- [ ] **Step 3: Implement `peem_roi.py`**

- [ ] **Step 4: Run — expect PASS**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_roi.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/core/peem_roi.py tests/test_peem_roi.py
git commit -m "feat(peem): ROI mask helper for rect/ellipse/polygon"
```

---

### Task 2: `drift_correct` engine + tests

**Files:**
- Modify: `tensorspec/core/peem_engine.py`
- Create: `tests/test_peem_drift.py`

**Interfaces:**
- Consumes: `roi_to_mask` from Task 1
- Produces:

```python
def drift_correct(
    tensor: TensorData,
    *,
    ref_index: int,
    roi: dict,
    search_radius: int,
    track_channel: int = 0,
) -> TensorData:
    """
    Raw (frame,y,x) or paired (pair,channel,y,x).
    NCC in ROI vs ref; integer shifts; edge-clamp apply.
    Paired: estimate on track_channel; apply same (dx,dy) to both channels.
    Metadata: drift_method, drift_ref_index, drift_roi, drift_search_radius,
    drift_track_channel, drift_shifts [{index,dx,dy}, ...]; passthrough pair/CSV keys.
    """
```

Helper (private OK):

```python
def _shift_plane(plane: np.ndarray, dx: int, dy: int) -> np.ndarray: ...
def _ncc_best_shift(ref_plane, mov_plane, mask, search_radius) -> tuple[int, int]: ...
```

- [ ] **Step 1: Write failing tests**

```python
# tests/test_peem_drift.py
import unittest
import numpy as np
from tensorspec.core.data_models import TensorData
from tensorspec.core.peem_engine import drift_correct


def _raw_blob(n=3, ny=32, nx=32, shifts=None):
    """Bright 5x5 square whose true translation per frame is shifts[i]=(dx,dy)."""
    if shifts is None:
        shifts = [(0, 0), (3, -2), (-1, 4)]
    frames = []
    yy, xx = np.mgrid[0:ny, 0:nx]
    for dx, dy in shifts:
        img = np.zeros((ny, nx), dtype=float)
        # place square so content at frame0 is centered; later frames shifted
        cy, cx = 16 + dy, 16 + dx
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= 4] = 1.0
        frames.append(img)
    value = np.stack(frames, axis=0)
    return TensorData(
        value=value,
        axes=[np.arange(n), np.arange(ny), np.arange(nx)],
        labels=["frame", "y", "x"],
        units=["", "px", "px"],
        data_type="Experimental PEEM",
        metadata={"pol": ["unknown"] * n, "source": "test", "loader": "tif_sequence"},
    ), shifts


class TestDriftCorrect(unittest.TestCase):
    def test_recovers_integer_shifts(self):
        tensor, true = _raw_blob()
        roi = {"kind": "rect", "x0": 8, "y0": 8, "x1": 24, "y1": 24}
        out = drift_correct(tensor, ref_index=0, roi=roi, search_radius=8)
        got = [(s["dx"], s["dy"]) for s in out.metadata["drift_shifts"]]
        # shifts relative to ref: frame i should undo true[i]-true[0]
        expected = [
            (true[0][0] - true[i][0], true[0][1] - true[i][1]) for i in range(len(true))
        ]
        self.assertEqual(got, expected)
        self.assertEqual(out.metadata["drift_method"], "ncc_roi")

    def test_paired_same_shift_both_channels(self):
        n, ny, nx = 2, 24, 24
        # channel0 has feature; channel1 is offset copy of same geometry
        base = np.zeros((ny, nx), float)
        base[10:15, 10:15] = 1.0
        moved = np.zeros_like(base)
        moved[10:15, 13:18] = 1.0  # +3 in x
        cube = np.zeros((n, 2, ny, nx), float)
        cube[0, 0] = base
        cube[0, 1] = base * 0.5
        cube[1, 0] = moved
        cube[1, 1] = moved * 0.5
        td = TensorData(
            value=cube,
            axes=[np.arange(n), np.arange(2), np.arange(ny), np.arange(nx)],
            labels=["pair", "channel", "y", "x"],
            units=["", "", "px", "px"],
            data_type="Experimental PEEM (paired)",
            metadata={"channel_tags": ["CP", "CM"], "pair_mode": "CP_CM"},
        )
        roi = {"kind": "rect", "x0": 5, "y0": 5, "x1": 20, "y1": 20}
        out = drift_correct(td, ref_index=0, roi=roi, search_radius=6, track_channel=0)
        self.assertEqual(out.metadata["drift_shifts"][1]["dx"], -3)
        # both channels of pair 1 should match after correction (feature back)
        np.testing.assert_allclose(out.value[1, 0][10:15, 10:15], 1.0, atol=1e-6)
        np.testing.assert_allclose(out.value[1, 1][10:15, 10:15], 0.5, atol=1e-6)

    def test_bad_ref_raises(self):
        tensor, _ = _raw_blob()
        with self.assertRaises(ValueError):
            drift_correct(
                tensor,
                ref_index=99,
                roi={"kind": "rect", "x0": 1, "y0": 1, "x1": 10, "y1": 10},
                search_radius=3,
            )
```

Adjust the synthetic blob construction if needed so NCC recovers the documented expected relative shifts — tests must assert recoverable known integers.

- [ ] **Step 2: Run — expect FAIL**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_drift.py -v
```

- [ ] **Step 3: Implement `drift_correct` in `peem_engine.py`**

Import `roi_to_mask`. Do not change `pair_stack` behavior.

- [ ] **Step 4: Run — expect PASS**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_roi.py tests/test_peem_drift.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/core/peem_engine.py tests/test_peem_drift.py
git commit -m "feat(peem): ROI NCC drift_correct for raw and paired stacks"
```

---

### Task 3: Schemas + `/drift` API + processed 3D/4D

**Files:**
- Modify: `tensorspec/web/server/schemas.py`
- Modify: `tensorspec/web/server/routers/peem.py`
- Modify: `tests/test_peem_api.py`

**Interfaces:**
- Produces:

```python
class PeemRoi(BaseModel):
    kind: Literal["rect", "ellipse", "polygon"]
    # rect
    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None
    # ellipse
    cx: float | None = None
    cy: float | None = None
    rx: float | None = None
    ry: float | None = None
    # polygon
    points: list[list[float]] | None = None

class PeemDriftRequest(BaseModel):
    source: Literal["raw", "processed"] = "raw"
    ref_index: int = Field(ge=0, le=100_000)
    search_radius: int = Field(default=20, ge=1, le=200)
    track_channel: int = Field(default=0, ge=0, le=1)
    roi: PeemRoi

class PeemDriftSummary(BaseModel):
    name: str
    source: str
    n_planes: int
    ref_index: int
    search_radius: int
    has_processed: bool = True
    has_drift: bool = True
    max_abs_dx: int
    max_abs_dy: int
    shape: list[int]

# PeemMeta additions:
#   has_drift: bool = False
#   drift_method: str | None = None
```

**Router changes (critical):**

1. Replace pair-only processed access with helpers:

```python
def _processed_tensor(session, name) -> TensorData | None:
    """Return processed cube if present; validate 3D frame stack OR 4D paired."""

def _processed_pair_tensor(...):  # keep name for callers OR thin wrapper
    """Require 4D paired; used by pair-specific paths / drift source=processed when 4D."""
```

Rules:
- `get_meta` / `get_frame(node=processed)`:
  - If 4D paired → existing pair/channel behavior.
  - If 3D `["frame","y","x"]` with PEEM data_type → treat like raw frames (`i` = frame index; ignore channel or require channel omitted).
  - Else 422.
- `POST /{name}/drift`:
  - `source=raw` → `_require_tensor` → `drift_correct` → `write_processed_data`.
  - `source=processed` → pull processed; must be 4D paired **or** 3D frame PEEM; then drift → overwrite processed.
- `has_drift` True when processed metadata contains `drift_method`.

2. Endpoint:

```python
@router.post("/{name}/drift", response_model=PeemDriftSummary)
def drift_peem(name: str, request: PeemDriftRequest, session: Session = Depends(...)): ...
```

Map `PeemRoi` → plain `dict` for engine (`model_dump(exclude_none=True)`).

- Consumes: Task 2 `drift_correct`

- [ ] **Step 1: Write failing API tests** (append to `tests/test_peem_api.py`)

```python
from tensorspec.web.server.schemas import PeemDriftRequest, PeemRoi

def test_drift_raw_writes_processed_3d(self):
    # load 2-frame folder, POST drift with rect ROI covering feature, assert has_drift,
    # get_frame node=processed index 0 works as 3D
    ...

def test_drift_paired_keeps_4d_and_same_channel_shift(self):
    # load CP/CM, pair, drift source=processed, meta.has_drift, frame pair/channel OK
    ...
```

Include at least one TestClient path for POST `/api/peem/{name}/drift`.

- [ ] **Step 2: Run — expect FAIL**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_api.py -k drift -v
```

- [ ] **Step 3: Implement schemas + router**

Ensure existing pair tests still pass (4D path unchanged for pairing).

- [ ] **Step 4: Run full PEEM tests — expect PASS**

```bash
PYTHONPATH=. TensorSpec_env/bin/pytest tests/test_peem_roi.py tests/test_peem_drift.py tests/test_peem_engine.py tests/test_peem_api.py -q
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/web/server/schemas.py tensorspec/web/server/routers/peem.py tests/test_peem_api.py
git commit -m "feat(peem): drift API and processed 3D/4D frame access"
```

---

### Task 4: Suite UI — ROI tools + Apply Drift

**Files:**
- Modify: `tensorspec/web/static/js/api.js`
- Modify: `tensorspec/web/static/js/peem_suite.js`
- Modify: `tensorspec/web/templates/suites/peem_suite.html`

**Interfaces:**
- Produces:

```javascript
peemDrift: (name, payload) =>
  request(`/api/peem/${encodeURIComponent(name)}/drift`, {
    method: "POST",
    body: JSON.stringify(payload),
  }),
```

**HTML:**
- Enable Drift fieldset (remove `disabled` on fieldset).
- `#peem-algo`: single enabled option `ncc_roi` label “NCC (ROI)”; keep Phase/Manual as `disabled` options.
- Enable `#peem-ref`, `#peem-search`, Apply button `id="peem-apply-drift"`.
- Add `#peem-track-channel` select (0/1); hide unless processed is 4D paired.
- ROI toolbar near canvas: Rect / Ellipse / Polygon / Clear (`#peem-roi-rect` etc.); `#peem-roi-close` for polygon; status `#peem-roi-status`.

**JS behavior:**
- State: `roi` object or null; `roiMode` drawing tool; polygon `points` while drawing.
- Mouse on canvas: map client → image pixel coords (account for canvas draw scale).
- Overlay ROI on canvas after image draw (stroke).
- Default drift `source`: `processed` if `hasProcessed`, else `raw`.
- Apply: require ROI; POST drift; stale-name guards like pair; on success `has_drift`, switch to processed, refresh meta/viewer, status with `max_abs_dx/dy`.
- Separate/BG stay disabled.

- [ ] **Step 1: Wire `api.js` `peemDrift`**

- [ ] **Step 2: Update `peem_suite.html`**

- [ ] **Step 3: Implement ROI + drift in `peem_suite.js`**

- [ ] **Step 4: Syntax / DOM sanity**

```bash
# if node unavailable, use jsc or skip with note — still verify getElementById ids exist in HTML
grep -E 'peem-apply-drift|peem-roi-' tensorspec/web/templates/suites/peem_suite.html
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/web/static/js/api.js tensorspec/web/static/js/peem_suite.js \
  tensorspec/web/templates/suites/peem_suite.html
git commit -m "feat(peem): ROI drift UI and Apply Drift"
```

---

### Task 5: Docs + Einstein deploy

**Files:**
- Modify: `roadmap.md` — check “once stacked, build drift-correction options” (and/or add sub-bullet for ROI NCC if clearer)
- Modify: `README.md` — PEEM bullet mentions ROI drift → `/processed`

- [ ] **Step 1: Update docs** — do **not** check Separate / BG / sum-rule / I0 apply

- [ ] **Step 2: Commit**

```bash
git add roadmap.md README.md
git commit -m "docs: mark PEEM ROI drift shipped"
```

- [ ] **Step 3: Push + Einstein**

```bash
git push origin HEAD
ssh einstein 'cd ~/TensorSpec && git fetch && git checkout HTML_einstein_app && git pull && curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/api/health'
```

- [ ] **Step 4: Verify** health 200; OpenAPI/routes include `/api/peem/{name}/drift`

---

## Spec coverage (self-review)

| Spec item | Task |
|-----------|------|
| `roi_to_mask` rect/ellipse/polygon | 1 |
| NCC integer drift + edge-clamp | 2 |
| Raw + paired; same shift both channels | 2 |
| `/processed` write + drift attrs | 2–3 |
| POST /drift, meta has_drift, 3D/4D processed view | 3 |
| ROI UI + Apply Drift | 4 |
| Separate/phase/manual/BG deferred | 4 (left disabled) |
| Roadmap + Einstein | 5 |
| Tests ROI + drift + API | 1–3 |

## Placeholder scan

No TBD steps; rect inclusivity, caps, shift sign, and processed 3D/4D rules locked above.

## Type consistency

- Modes/kinds: `rect|ellipse|polygon`; source `raw|processed`.
- Metadata keys: `drift_method`, `drift_ref_index`, `drift_roi`, `drift_search_radius`, `drift_track_channel`, `drift_shifts`.
- Summary: `has_drift`, `max_abs_dx`, `max_abs_dy`.
