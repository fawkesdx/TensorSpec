# Crystal Matplotlib Figure Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add headless Matplotlib Crystal figure export (PNG/SVG/PDF) matching Draw styles, with optional three.js camera; keep three.js as the interactive viewer.

**Architecture:** New `crystal_figure.py` draws from geometry (atoms/bonds/cell/polyhedra). Crystal router `POST /{name}/export/figure` builds geometry via existing structure path (basis/supercell/omit), then returns download bytes. UI adds Export figure controls next to 3ds/Blender.

**Tech Stack:** matplotlib Agg, FastAPI, existing Crystal geometry helpers, three.js camera snapshot from `CrystalViewer`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-09-crystal-matplotlib-figure-export-design.md`
- Branch: `HTML_einstein_app` only — never merge to `main`
- No Qt / PyVista; `matplotlib.use("Agg")` only
- three.js stays interactive; `#cr-backend` not a live Matplotlib switch
- Ignore PBR in figure (flat colors)
- Polyhedra: best-effort wireframe from `CrystalGeometry.polyhedra`; OK to draw faces lightly if cheap
- Atom cap: same `MAX_RENDER_ATOMS` as geometry/export (422 if over)
- Leave `#cr-hires` alone (unrelated stub; out of scope)

## File map

| File | Role |
|------|------|
| `tensorspec/plotting/backends/crystal_figure.py` | Agg renderer → bytes |
| `tensorspec/web/server/schemas.py` | `CrystalFigureExportRequest` |
| `tensorspec/web/server/routers/crystal.py` | `POST /{name}/export/figure` |
| `tensorspec/web/static/js/api.js` | `crystalExportFigure` |
| `tensorspec/web/static/js/viewers/viewer_3d.js` | `getCameraSnapshot()` |
| `tensorspec/web/static/js/crystal_suite.js` | Export handler |
| `tensorspec/web/templates/suites/crystal_suite.html` | Controls + backend hint |
| `tests/test_crystal_figure.py` | Unit tests for renderer + omit |
| `tests/test_crystal_figure_export_api.py` | API smoke with TestClient/session (optional thin) |

---

### Task 1: `crystal_figure.py` + unit tests

**Files:**
- Create: `tensorspec/plotting/backends/crystal_figure.py`
- Test: `tests/test_crystal_figure.py`

**Interfaces:**
- Produces:

```python
def export_crystal_figure(
    *,
    atoms: list[dict],          # {element, position: [x,y,z], radius: float}
    bonds: list[tuple[int, int]],
    cell: list[list[float]] | None,  # 3 lattice vectors; None → no cell
    polyhedra: list[dict] | None = None,  # {vertices, simplices} optional
    show_bonds: bool = True,
    show_cell: bool = True,
    atom_scale: float = 0.5,
    colors: dict[str, str] | None = None,  # element → "#rrggbb"
    title: str = "",
    fmt: Literal["png", "svg", "pdf"] = "png",
    camera: dict | None = None,  # {position, target, up} each len-3 float lists
    dpi: int = 200,
) -> bytes: ...
```

- Consumes: matplotlib Agg only

- [ ] **Step 1: Write failing tests**

```python
# tests/test_crystal_figure.py
import unittest
from tensorspec.plotting.backends import crystal_figure as cf


class TestCrystalFigure(unittest.TestCase):
    def test_png_nonempty(self):
        atoms = [
            {"element": "Si", "position": [0.0, 0.0, 0.0], "radius": 1.1},
            {"element": "Si", "position": [1.0, 1.0, 1.0], "radius": 1.1},
        ]
        bonds = [(0, 1)]
        cell = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
        raw = cf.export_crystal_figure(
            atoms=atoms, bonds=bonds, cell=cell, fmt="png", title="Si"
        )
        self.assertIsInstance(raw, (bytes, bytearray))
        self.assertGreater(len(raw), 100)
        self.assertTrue(raw[:8] == b"\x89PNG\r\n\x1a\n")

    def test_omit_changes_output(self):
        atoms2 = [
            {"element": "C", "position": [0.0, 0.0, 0.0], "radius": 0.7},
            {"element": "O", "position": [1.5, 0.0, 0.0], "radius": 0.6},
        ]
        a = cf.export_crystal_figure(atoms=atoms2, bonds=[], cell=None, fmt="png")
        b = cf.export_crystal_figure(atoms=atoms2[:1], bonds=[], cell=None, fmt="png")
        self.assertNotEqual(a, b)

    def test_svg_header(self):
        atoms = [{"element": "H", "position": [0, 0, 0], "radius": 0.3}]
        raw = cf.export_crystal_figure(atoms=atoms, bonds=[], cell=None, fmt="svg")
        self.assertIn(b"<svg", raw.lower())
```

- [ ] **Step 2: Run — expect FAIL (module missing)**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_crystal_figure -v
```

- [ ] **Step 3: Implement renderer**

```python
# tensorspec/plotting/backends/crystal_figure.py
"""Headless Crystal structure figure export (atoms / bonds / cell).

No GUI toolkit imports — matplotlib Agg only. Flat materials (no PBR).
"""
from __future__ import annotations

import io
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

_DEFAULT_CPK = {
    "H": "#ffffff", "C": "#909090", "N": "#3050f8", "O": "#ff0d0d",
    "Si": "#f0c8a0", "S": "#ffff30", "Fe": "#e06633", "Cu": "#c88033",
}
_FALLBACK = "#808080"


def export_crystal_figure(...):  # signature above
    fig = plt.figure(figsize=(6.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    # atoms: ax.scatter(xs, ys, zs, s=scaled, c=colors)
    # bonds: Line3DCollection of segments between atom positions
    # cell: 12 edges from origin + lattice vectors
    # polyhedra: Poly3DCollection faces alpha~0.25 if provided
    # camera: if camera dict, set ax.view_init / ax.dist best-effort from
    #   position-target vector (elev/azim); else elev=20, azim=45
    # equal aspect: set_box_aspect on ranges
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi if fmt == "png" else None,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()
```

Reuse CPK from router later via passed `colors` dict; defaults in module for unit tests.

For `atom_scale`: marker size ≈ `(radius * atom_scale * k)**2` with k tuned so Si looks reasonable (document constant).

- [ ] **Step 4: Tests PASS**

- [ ] **Step 5: Commit**

```bash
git add tensorspec/plotting/backends/crystal_figure.py tests/test_crystal_figure.py
git commit -m "$(cat <<'EOF'
feat: headless Matplotlib crystal figure renderer

PNG/SVG/PDF atoms, bonds, cell; flat CPK colors.
EOF
)"
```

---

### Task 2: Schema + API endpoint

**Files:**
- Modify: `tensorspec/web/server/schemas.py`
- Modify: `tensorspec/web/server/routers/crystal.py`
- Test: `tests/test_crystal_figure_export_api.py` (or extend existing crystal router tests if present)

**Interfaces:**
- Consumes: `export_crystal_figure`, `_structure_for_export` / geometry helpers, `_cpk_color` / `_CPK_COLORS` already in `crystal.py`
- Produces: `POST /{name}/export/figure` → file Response

- [ ] **Step 1: Schema**

```python
class CrystalFigureCamera(BaseModel):
    position: list[float] = Field(min_length=3, max_length=3)
    target: list[float] = Field(min_length=3, max_length=3)
    up: list[float] = Field(default_factory=lambda: [0.0, 1.0, 0.0], min_length=3, max_length=3)


class CrystalFigureExportRequest(BaseModel):
    omit_atom_indices: list[int] = []
    nx: int = Field(default=1, ge=1, le=20)
    ny: int = Field(default=1, ge=1, le=20)
    nz: int = Field(default=1, ge=1, le=20)
    basis: Literal["conventional", "primitive"] = "conventional"
    bond_threshold: float = Field(default=1.15, ge=0.5, le=3.0)
    show_bonds: bool = True
    show_polyhedra: bool = False
    show_cell: bool = True
    atom_scale: float = Field(default=0.5, ge=0.1, le=3.0)
    fmt: Literal["png", "svg", "pdf"] = "png"
    title: str = Field(default="", max_length=128)
    use_current_view: bool = False
    camera: CrystalFigureCamera | None = None

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny * self.nz
```

- [ ] **Step 2: Failing API test**

```python
# tests/test_crystal_figure_export_api.py
"""Crystal figure export endpoint (no browser)."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from pymatgen.core import Lattice, Structure

# Follow existing crystal API test patterns in repo if any; else:
from tensorspec.core.workspace import WorkspaceManager
from tensorspec.web.server.session import Session
# Mount app the same way other web tests do — search tests/ for TestClient + crystal
```

If no TestClient crystal pattern exists, test the handler logic via a focused function:

```python
def test_build_figure_bytes_from_structure(self):
    # call internal helper that router uses, or
    from tensorspec.web.server.routers import crystal as cr
    ...
```

Prefer: invent `_figure_bytes_for_request(session, name, request) -> tuple[bytes, str, str]` (payload, media_type, filename) used by the route — unit-test that without full app if app bootstrap is heavy.

Simplest reliable approach matching DFT tests: temporary Session + call route function directly:

```python
def test_export_figure_png(self):
    with TemporaryDirectory() as tmp:
        session = Session(session_id="t", workspace=WorkspaceManager(project_dir=Path(tmp)))
        structure = Structure(Lattice.cubic(5.43), ["Si", "Si"], [[0,0,0],[0.25,0.25,0.25]])
        session.workspace.push_crystal_structure("Si", structure.lattice.matrix, structure=structure)
        req = CrystalFigureExportRequest(fmt="png", show_bonds=True)
        resp = crystal_router.export_crystal_figure("Si", req, session=session)
        self.assertEqual(resp.media_type, "image/png")
        self.assertGreater(len(resp.body), 100)
```

(Adjust to how FastAPI `Response` exposes body in this codebase.)

- [ ] **Step 3: Implement route**

```python
@router.post("/{name}/export/figure")
def export_crystal_figure_route(
    name: str,
    request: CrystalFigureExportRequest,
    session: Session = Depends(current_session),
):
    structure = _require_structure(session, name)
    # Same basis/supercell/omit path as export_scene / CIF
    # Build geo via _geometry_from_structure(... show_polyhedra=request.show_polyhedra)
    # Filter omit
    # colors = {el: _cpk_color(el) for el in geo.elements}
    # atoms = [{element, position, radius} for each geo.atom]
    # camera = request.camera.model_dump() if request.use_current_view and request.camera else None
    payload = export_crystal_figure(...)
    media = {"png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf"}[request.fmt]
    return Response(content=payload, media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{name}_figure.{request.fmt}"'})
```

Import `export_crystal_figure` from plotting backend (alias route name to avoid clash — e.g. function `export_figure` calling `export_crystal_figure`).

- [ ] **Step 4: Tests PASS**

- [ ] **Step 5: Commit**

```bash
git add tensorspec/web/server/schemas.py tensorspec/web/server/routers/crystal.py tests/test_crystal_figure_export_api.py
git commit -m "$(cat <<'EOF'
feat: Crystal Suite Matplotlib figure export API

POST /export/figure builds styled PNG/SVG/PDF from session structure.
EOF
)"
```

---

### Task 3: UI + camera snapshot + hint

**Files:**
- Modify: `tensorspec/web/templates/suites/crystal_suite.html`
- Modify: `tensorspec/web/static/js/api.js`
- Modify: `tensorspec/web/static/js/viewers/viewer_3d.js`
- Modify: `tensorspec/web/static/js/crystal_suite.js`

**Interfaces:**
- `CrystalViewer.getCameraSnapshot()` → `{ position: [x,y,z], target: [x,y,z], up: [x,y,z] }`
- `TensorSpecAPI.crystalExportFigure(name, payload)` → blob (mirror `crystalExportScene`)
- `#cr-export-figure`, `#cr-export-figure-fmt`, `#cr-export-figure-view`

- [ ] **Step 1: HTML** (inside Export Elements fieldset, after 3ds/Blender row)

```html
<div class="form-row form-row--stacked">
    <label for="cr-export-figure-fmt">Figure format:</label>
    <select class="field" id="cr-export-figure-fmt">
        <option value="png" selected>PNG</option>
        <option value="svg">SVG</option>
        <option value="pdf">PDF</option>
    </select>
</div>
<label class="check"><input type="checkbox" id="cr-export-figure-view"> Use current view</label>
<button type="button" class="btn btn--block" id="cr-export-figure">Export figure (Matplotlib)</button>
```

Update `#cr-backend` hint:

```html
<p class="hint">Interactive viewer: three.js. Publication figures: Export figure (Matplotlib). PyVista: not in this build.</p>
```

(Keep select disabled with single `three.js` option.)

- [ ] **Step 2: Viewer helper**

In `viewer_3d.js` on `CrystalViewer`:

```javascript
getCameraSnapshot() {
  const p = this.camera.position;
  const t = this.controls?.target ?? new THREE.Vector3();
  const u = this.camera.up;
  return {
    position: [p.x, p.y, p.z],
    target: [t.x, t.y, t.z],
    up: [u.x, u.y, u.z],
  };
}
```

- [ ] **Step 3: api.js**

```javascript
crystalExportFigure: async (name, payload) => {
  const response = await fetch(
    `/api/crystal/${encodeURIComponent(name)}/export/figure`,
    { method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload) }
  );
  // same error/blob handling as crystalExportScene
  return response.blob();
},
```

- [ ] **Step 4: crystal_suite.js**

```javascript
function figureExportPayload() {
  const geo = geometryRequest();
  const payload = {
    ...geo,
    omit_atom_indices: omittedAtomIndices(),
    show_cell: Boolean(dom.showCell?.checked),
    atom_scale: Number(dom.radius?.value) || 0.5,
    fmt: dom.exportFigureFmt?.value || "png",
    title: activeCrystal || "",
    use_current_view: Boolean(dom.exportFigureView?.checked),
    camera: null,
  };
  if (payload.use_current_view) {
    const view = ensureViewer();
    payload.camera = view.getCameraSnapshot();
  }
  return payload;
}

async function exportFigure() {
  if (!activeCrystal) { setStatus("Load or stack a structure first.", true); return; }
  const payload = figureExportPayload();
  setStatus(`Exporting figure (${payload.fmt}) for ${activeCrystal}…`);
  try {
    const blob = await TensorSpecAPI.crystalExportFigure(activeCrystal, payload);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${activeCrystal}_figure.${payload.fmt}`;
    link.click();
    URL.revokeObjectURL(url);
    setStatus(`Exported ${activeCrystal}_figure.${payload.fmt}`);
  } catch (err) {
    setStatus(err.message, true);
  }
}
// wire click on #cr-export-figure
```

Register `dom.exportFigure`, `exportFigureFmt`, `exportFigureView` in the DOM map.

- [ ] **Step 5: Manual sanity** — unique ids; no JS syntax errors.

- [ ] **Step 6: Commit**

```bash
git add tensorspec/web/templates/suites/crystal_suite.html \
  tensorspec/web/static/js/api.js \
  tensorspec/web/static/js/viewers/viewer_3d.js \
  tensorspec/web/static/js/crystal_suite.js
git commit -m "$(cat <<'EOF'
feat: Crystal UI Matplotlib figure export controls

Format select, use-current-view, download via export/figure API.
EOF
)"
```

---

### Task 4: Push + Einstein pull (controller)

- [ ] Push `HTML_einstein_app`
- [ ] Einstein `git pull` (Mac hosts UI; pull keeps deploy in sync)
- [ ] Optional: load Si → Export PNG → open file

Do not merge to `main`.

---

## Spec coverage

| Spec item | Task |
|-----------|------|
| `crystal_figure.py` | 1 |
| PNG/SVG/PDF | 1–2 |
| Styles + omit | 2 |
| Camera optional | 2–3 |
| UI button + hint | 3 |
| No live Matplotlib backend | 3 |
| Tests | 1–2 |

## Self-review

- `#cr-hires` left alone (explicit).
- Route name must not shadow imported `export_crystal_figure` — use distinct def name.
- Polyhedra passed when `show_polyhedra` and geo has entries.
