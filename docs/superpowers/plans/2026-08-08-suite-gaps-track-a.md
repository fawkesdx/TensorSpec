# Suite Gaps Track A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Track A wire-now gaps: Align [hkl], drop unused `overlay_crystal`, W90 dual-solve band overlay, crystal 3ds/Blender export + checkboxes, Render + hi-res PNG; push and Einstein deploy.

**Architecture:** Hybrid — server dual-solves bands and builds DCC scripts via `SceneExporter`; browser owns Miller camera, BZ overlay toggle, and PNG capture. Thin vertical slices with a commit per task.

**Tech Stack:** FastAPI, Pydantic, pymatgen, chinook band_service, SceneExporter, three.js CrystalViewer, BandPlot canvas, unittest via `TensorSpec_env/bin/python`.

## Global Constraints

- Branch: `HTML_einstein_app` only — never merge to `main`
- Tests: `./TensorSpec_env/bin/python -m unittest …`
- After final push: `ssh einstein` → pull `HTML_einstein_app` → restart uvicorn on `:8000`
- Spec: `docs/superpowers/specs/2026-08-08-suite-gaps-track-a-design.md`
- Out of scope: QE XC, ARPES slit physics, 2D isoenergy, polyhedra/eraser/PBR/Matplotlib, shell suites, ARPES history

## File map

| File | Responsibility |
|------|----------------|
| `tensorspec/web/static/js/viewers/viewer_3d.js` | `lookAlongMiller`; PNG capture; `preserveDrawingBuffer` |
| `tensorspec/web/static/js/crystal_suite.js` | Align, export, render/PNG handlers; stop sending `overlay_crystal` |
| `tensorspec/web/templates/suites/crystal_suite.html` | Enable Align; export IDs; render/hires IDs; hint |
| `tensorspec/web/server/schemas.py` | Drop `overlay_crystal`; add `overlay_wannier` / `overlay_bands`; `SceneExportRequest` |
| `tensorspec/web/server/routers/crystal.py` | Export endpoint + geometry→SceneExporter tuples |
| `tensorspec/web/server/routers/dft.py` | Dual-solve when `overlay_wannier` |
| `tensorspec/web/static/js/viewers/band_plot.js` | Draw dashed red `overlay_bands` |
| `tensorspec/web/static/js/dft_suite.js` / `dft_suite.html` / `api.js` | Overlay checkbox + API download helper |
| `tests/test_band_overlay.py` | Schema + dual-solve flag behavior |
| `tests/test_scene_export.py` | Export tuple / script markers |
| `tests/test_wannier_upload.py` | Extend or leave; overlay covered in new test |

---

### Task 1: Align [hkl] camera

**Files:**
- Modify: `tensorspec/web/static/js/viewers/viewer_3d.js` (after `lookAlong`)
- Modify: `tensorspec/web/templates/suites/crystal_suite.html` (Align button ~L152)
- Modify: `tensorspec/web/static/js/crystal_suite.js` (dom + listener)
- Test: manual / reuse `tests/test_miller_plane.py` (already covers normal math)

**Interfaces:**
- Consumes: `millerNormal(cell, h, k, l)` exported from `viewer_3d.js`
- Produces: `CrystalViewer.lookAlongMiller(h: number, k: number, l: number): void`

- [ ] **Step 1: Add `lookAlongMiller` to viewer**

In `viewer_3d.js`, after `lookAlong`:

```javascript
lookAlongMiller(h, k, l) {
  if (!this.geometry) return;
  const n = millerNormal(this.geometry.cell, h, k, l);
  if (!n || (n[0] === 0 && n[1] === 0 && n[2] === 0)) return;
  const dir = new THREE.Vector3(...n).normalize();
  const dist = this.camera.position.length() || 30;
  this.camera.position.copy(dir.multiplyScalar(dist));
  this.controls.target.set(0, 0, 0);
  this.controls.update();
}
```

Note: `millerNormal` already throws/returns null for (0,0,0) in JS — guard both ways. Match existing `millerNormal` signature in this file (it returns a plain array).

- [ ] **Step 2: Enable Align button in HTML**

Replace disabled Align button with:

```html
<button type="button" class="btn" id="cr-align">Align</button>
```

Update hint line that says Align is “not yet” (partial — full hint update in Task 5).

- [ ] **Step 3: Wire `crystal_suite.js`**

Add `align: el("cr-align")` to `dom`. Listener:

```javascript
dom.align?.addEventListener("click", () => {
  const h = Number(dom.cutH?.value) || 0;
  const k = Number(dom.cutK?.value) || 0;
  const l = Number(dom.cutL?.value) || 0;
  if (h === 0 && k === 0 && l === 0) {
    setStatus("Align needs non-zero [h k l].", true);
    return;
  }
  ensureViewer().lookAlongMiller(h, k, l);
  setStatus(`Aligned to [${h} ${k} ${l}]`);
});
```

- [ ] **Step 4: Smoke (browser or static check)**

Confirm `cr-align` id present and `lookAlongMiller` exists:

```bash
rg -n "lookAlongMiller|cr-align" tensorspec/web/static/js/viewers/viewer_3d.js tensorspec/web/static/js/crystal_suite.js tensorspec/web/templates/suites/crystal_suite.html
```

Expected: hits in all three files.

- [ ] **Step 5: Commit**

```bash
git add tensorspec/web/static/js/viewers/viewer_3d.js \
  tensorspec/web/static/js/crystal_suite.js \
  tensorspec/web/templates/suites/crystal_suite.html
git commit -m "$(cat <<'EOF'
feat(crystal): align camera along Miller [hkl]

EOF
)"
```

---

### Task 2: Drop unused `overlay_crystal` from BZ API

**Files:**
- Modify: `tensorspec/web/server/schemas.py` (`BZRequest`)
- Modify: `tensorspec/web/static/js/crystal_suite.js` (`renderBZ` payload)
- Test: `tests/test_bz_overlay_field.py` (new)

**Interfaces:**
- Consumes: existing `renderBZ` client clear-vs-keep logic
- Produces: `BZRequest` without `overlay_crystal`

- [ ] **Step 1: Write failing test**

Create `tests/test_bz_overlay_field.py`:

```python
"""BZRequest no longer carries unused overlay_crystal."""
import unittest

from tensorspec.web.server.schemas import BZRequest


class TestBZRequestNoOverlayField(unittest.TestCase):
    def test_model_has_no_overlay_crystal(self):
        self.assertNotIn("overlay_crystal", BZRequest.model_fields)

    def test_extra_overlay_crystal_ignored_or_rejected(self):
        # Pydantic v2 default: ignore extras unless configured otherwise
        req = BZRequest(scale=1.0, overlay_crystal=True)  # may ignore
        self.assertFalse(hasattr(req, "overlay_crystal") and "overlay_crystal" in req.model_fields)


if __name__ == "__main__":
    unittest.main()
```

Simplify Step 1 test to only `assertNotIn("overlay_crystal", BZRequest.model_fields)` once field removed — write that assertion first so it fails while field still exists.

- [ ] **Step 2: Run test — expect FAIL**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_bz_overlay_field -v
```

Expected: FAIL — `overlay_crystal` still in `model_fields`.

- [ ] **Step 3: Remove field + stop sending**

In `schemas.py` `BZRequest`, delete `overlay_crystal: bool = True`.

In `crystal_suite.js` `renderBZ`, remove `overlay_crystal: Boolean(dom.bzOverlay.checked)` from the API payload. Keep the client `if (!dom.bzOverlay.checked)` branch unchanged.

- [ ] **Step 4: Run test — expect PASS**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_bz_overlay_field -v
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/web/server/schemas.py \
  tensorspec/web/static/js/crystal_suite.js \
  tests/test_bz_overlay_field.py
git commit -m "$(cat <<'EOF'
refactor(crystal): drop unused BZ overlay_crystal field

Client already owns crystal+BZ overlay; stop lying in the schema.
EOF
)"
```

---

### Task 3: W90 dual-solve + BandPlot overlay

**Files:**
- Modify: `tensorspec/web/server/schemas.py` (`BandRequest`, `BandResult`)
- Modify: `tensorspec/web/server/routers/dft.py` (`compute_bands`)
- Modify: `tensorspec/web/static/js/viewers/band_plot.js`
- Modify: `tensorspec/web/static/js/dft_suite.js`, `dft_suite.html`
- Test: `tests/test_band_overlay.py`

**Interfaces:**
- Consumes: `_wannier_hr_path`, `band_service.calculate_bands(..., w90_filepath=)`
- Produces:
  - `BandRequest.overlay_wannier: bool = False`
  - `BandResult.overlay_bands: list[list[float]] | None = None`
  - Calculate payload includes `overlay_wannier`

- [ ] **Step 1: Write failing schema test**

Create `tests/test_band_overlay.py`:

```python
"""Band overlay_wannier / overlay_bands schema contract."""
import unittest

from tensorspec.web.server.schemas import BandRequest, BandResult


class TestBandOverlaySchema(unittest.TestCase):
    def test_overlay_wannier_default_false(self):
        self.assertFalse(BandRequest().overlay_wannier)

    def test_overlay_bands_optional(self):
        # Minimal BandResult-like construction may need many fields —
        # assert field exists on model:
        self.assertIn("overlay_bands", BandResult.model_fields)
        self.assertIn("overlay_wannier", BandRequest.model_fields)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run — expect FAIL**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_band_overlay -v
```

- [ ] **Step 3: Schema fields**

In `schemas.py`:

```python
# BandRequest
overlay_wannier: bool = False

# BandResult
overlay_bands: list[list[float]] | None = None
```

- [ ] **Step 4: Dual-solve in `compute_bands`**

After primary `w90_filepath` resolution and **before** the budget check, adjust budget:

```python
needs_overlay = bool(request.overlay_wannier)
if needs_overlay:
    hr_overlay = _wannier_hr_path(session, name)
    if hr_overlay is None:
        raise HTTPException(
            status_code=422,
            detail="No uploaded wannier90_hr.dat for this crystal. Load one first.",
        )
solve_factor = 2 if (needs_overlay and not request.use_wannier) else 1
if estimated_k * orbitals ** 3 * solve_factor > DIAGONALISATION_BUDGET:
    raise HTTPException(...)  # same message style
```

After primary `result = band_service.calculate_bands(...)`:

```python
overlay_bands = None
if needs_overlay:
    if request.use_wannier:
        # Primary already W90 — reuse eigenvalues (no second diagonalisation)
        overlay_evals = eigenvalues  # set after primary arrays built
    else:
        overlay_result = band_service.calculate_bands(
            engine,
            # same kwargs as primary but:
            w90_filepath=str(hr_overlay),
            # keep path/hoppings/soc/tb_mode identical
            ...
        )
        overlay_evals = np.asarray(overlay_result["eigenvalues"])
    # Build overlay_bands list-of-lists like primary bands
```

Build primary `BandResult` as today; set `overlay_bands=[[float(v) for v in overlay_evals[:, b]] for b in range(overlay_evals.shape[1])]` when overlay active.

Order carefully: compute `eigenvalues` from primary first; if `use_wannier and needs_overlay`, set `overlay_bands` from the same primary `eigenvalues` without a second call.

- [ ] **Step 5: BandPlot dashed red**

In `band_plot.js` `draw()`, after primary band strokes (and before axes), if `this.result.overlay_bands`:

```javascript
const overlay = this.result.overlay_bands;
if (Array.isArray(overlay) && overlay.length) {
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 1.2;
    overlay.forEach((band) => {
        ctx.beginPath();
        for (let i = 0; i < band.length; i++) {
            const x = px(k_dist[i]);
            const y = py(band[i]);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
    });
    ctx.restore();
}
```

Use primary `k_dist` (same path).

- [ ] **Step 6: UI wire**

- `dft_suite.html`: enable `#tb-w90-overlay` only via JS (keep disabled in HTML initially); add hint under W90 block.
- `dft_suite.js`: `w90Overlay: el("tb-w90-overlay")`; on successful upload enable overlay checkbox like Use-W90; `readParameters` add `overlay_wannier: Boolean(dom.w90Overlay?.checked)`.

- [ ] **Step 7: Run tests**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_band_overlay tests.test_wannier_upload -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tensorspec/web/server/schemas.py \
  tensorspec/web/server/routers/dft.py \
  tensorspec/web/static/js/viewers/band_plot.js \
  tensorspec/web/static/js/dft_suite.js \
  tensorspec/web/templates/suites/dft_suite.html \
  tests/test_band_overlay.py
git commit -m "$(cat <<'EOF'
feat(dft): dual-solve W90 band overlay

Optional overlay_wannier returns dashed-red companion bands
on the same k-path without inventing new physics.
EOF
)"
```

---

### Task 4: Scene export API + checkboxes + 3ds/Blender

**Files:**
- Modify: `tensorspec/web/server/schemas.py` — add `SceneExportRequest`
- Modify: `tensorspec/web/server/routers/crystal.py` — helpers + route
- Modify: `tensorspec/web/templates/suites/crystal_suite.html`
- Modify: `tensorspec/web/static/js/crystal_suite.js`, `api.js`
- Test: `tests/test_scene_export.py`
- Reuse: `tensorspec/core/io/exporters.py` `SceneExporter`

**Interfaces:**
- Consumes: `_geometry_from_structure`, existing BZ builder pieces, `SceneExporter.export_3dsmax` / `export_blender`
- Produces:
  - `POST /api/crystal/{name}/export/{fmt}` → file download
  - `SceneExportRequest` with geometry knobs + `include_atoms|cell|bz`

- [ ] **Step 1: Write failing tests for tuple builder**

Create `tests/test_scene_export.py`:

```python
"""Crystal scene export → SceneExporter script markers."""
import tempfile
import unittest
from pathlib import Path

from pymatgen.core import Lattice, Structure

from tensorspec.core.io.exporters import SceneExporter
from tensorspec.web.server.routers import crystal as crystal_router
from tensorspec.web.server.schemas import SceneExportRequest


class TestSceneExportTuples(unittest.TestCase):
    def setUp(self):
        self.structure = Structure(Lattice.cubic(4.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        self.geo = crystal_router._geometry_from_structure("si", self.structure, show_bonds=True)

    def test_request_requires_at_least_one_include(self):
        with self.assertRaises(Exception):
            SceneExportRequest(include_atoms=False, include_cell=False, include_bz=False)

    def test_atoms_tuples_nonempty(self):
        atoms, bonds, lattice, bz = crystal_router._scene_export_parts(
            self.geo, include_atoms=True, include_cell=False, include_bz=False, bz_geometry=None
        )
        self.assertGreater(len(atoms), 0)
        self.assertEqual(lattice, [])
        self.assertIsNone(bz)

    def test_blender_script_contains_atom(self):
        atoms, bonds, lattice, bz = crystal_router._scene_export_parts(
            self.geo, include_atoms=True, include_cell=True, include_bz=False, bz_geometry=None
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.py"
            SceneExporter.export_blender(str(path), atoms, bonds, lattice, bz)
            text = path.read_text()
            self.assertIn("Atom", text)
            self.assertIn("TensorSpec_Crystal", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run — expect FAIL** (`SceneExportRequest` / `_scene_export_parts` missing)

```bash
./TensorSpec_env/bin/python -m unittest tests.test_scene_export -v
```

- [ ] **Step 3: Schema**

```python
class SceneExportRequest(BaseModel):
    nx: int = Field(default=1, ge=1, le=20)
    ny: int = Field(default=1, ge=1, le=20)
    nz: int = Field(default=1, ge=1, le=20)
    basis: Literal["conventional", "primitive"] = "conventional"
    bond_threshold: float = Field(default=1.15, ge=0.5, le=3.0)
    show_bonds: bool = True
    include_atoms: bool = True
    include_cell: bool = True
    include_bz: bool = False
    # BZ options when include_bz
    bz_scale: float = Field(default=1.0, ge=0.1, le=10)
    bz_style: Literal["solid", "skeleton", "both"] = "solid"
    bz_h: int = Field(default=0, ge=-10, le=10)
    bz_k: int = Field(default=0, ge=-10, le=10)
    bz_l: int = Field(default=1, ge=-10, le=10)

    @model_validator(mode="after")
    def _one_include(self):
        if not (self.include_atoms or self.include_cell or self.include_bz):
            raise ValueError("Select at least one of Atoms/Bonds, Unit Cell, or Brillouin Zone.")
        return self
```

(Use the project’s existing Pydantic v2 `model_validator` import pattern from `schemas.py`.)

- [ ] **Step 4: Helpers + route in `crystal.py`**

Default CPK hex for unknown elements: `"#808080"`. Atom tuple: `(x, y, z, radius, "#rrggbb")`. Bond: endpoints from atom positions, radius `0.1`, color `"#888888"`. Lattice: 12 edges of cell parallelepiped from `geo.cell` origin at `geo.center`-relative or absolute cart — **match viewer**: positions in `geo.atoms` are absolute cart; lattice edges from origin `O` along a,b,c and translations (same as typical cell wireframe). Prefer building lattice edges as the 12 segments of the parallelepiped starting at `center - (a+b+c)/2` **or** from `0` using matrix rows — inspect `viewer_3d.js` cell drawing and mirror that coordinate frame so Blender matches the viewer.

`_scene_export_parts(geo, *, include_atoms, include_cell, include_bz, bz_geometry) -> tuple[list, list, list, dict|None]`

BZ solid: if `bz_geometry` has `hull_points` + `simplices`, map to `{verts: [...], faces: [...]}`.

Route:

```python
@router.post("/{name}/export/{fmt}")
def export_scene(name: str, fmt: Literal["3dsmax", "blender"], request: SceneExportRequest, session: Session = Depends(current_session)):
    structure = _require_structure(session, name)
    # apply basis + supercell same as get_geometry
    geo = _geometry_from_structure(...)
    bz_geo = None
    if request.include_bz:
        bz_geo = ...  # call existing BZ construction with BZRequest-like fields
    atoms, bonds, lattice, bz = _scene_export_parts(...)
    # write temp file via SceneExporter, return FileResponse / Response
```

Filename: `{name}_scene.ms` for 3dsmax, `{name}_scene.py` for blender.

- [ ] **Step 5: HTML + JS + api.js**

HTML checkboxes/buttons:

```html
<label class="check"><input type="checkbox" id="cr-exp-atoms" checked> Atoms/Bonds</label>
<label class="check"><input type="checkbox" id="cr-exp-cell" checked> Unit Cell</label>
<label class="check"><input type="checkbox" id="cr-exp-bz"> Brillouin Zone</label>
...
<button type="button" class="btn" id="cr-export-3ds">Export 3ds Max</button>
<button type="button" class="btn" id="cr-export-blender">Export Blender</button>
```

`api.js`:

```javascript
crystalExportScene: async (name, fmt, payload) => {
  const response = await fetch(`/api/crystal/${encodeURIComponent(name)}/export/${fmt}`, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) { /* parse detail like other helpers */ }
  return response.blob();
},
```

`crystal_suite.js`: build payload from nx/ny/nz/thresh/basis + includes + bz_* from Tab 4 fields if present (`dom.bzScale` etc.); trigger download via object URL.

- [ ] **Step 6: Run tests**

```bash
./TensorSpec_env/bin/python -m unittest tests.test_scene_export -v
```

- [ ] **Step 7: Commit**

```bash
git add tensorspec/web/server/schemas.py \
  tensorspec/web/server/routers/crystal.py \
  tensorspec/web/static/js/api.js \
  tensorspec/web/static/js/crystal_suite.js \
  tensorspec/web/templates/suites/crystal_suite.html \
  tests/test_scene_export.py
git commit -m "$(cat <<'EOF'
feat(crystal): export 3ds Max / Blender scene scripts

Checkboxes filter atoms, cell edges, and optional BZ mesh.
EOF
)"
```

---

### Task 5: Render Structure + hi-res PNG

**Files:**
- Modify: `tensorspec/web/static/js/viewers/viewer_3d.js` (constructor + `capturePNG`)
- Modify: `tensorspec/web/templates/suites/crystal_suite.html`
- Modify: `tensorspec/web/static/js/crystal_suite.js`
- Test: static id/method presence via `rg`

**Interfaces:**
- Consumes: `refreshGeometry({ frame: true })`
- Produces: `CrystalViewer.capturePNG(scale?: number): string` (data URL)

- [ ] **Step 1: Enable preserveDrawingBuffer + capturePNG**

Constructor:

```javascript
this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
```

Method:

```javascript
capturePNG(scale = 2) {
  const canvas = this.renderer.domElement;
  const w = canvas.width;
  const h = canvas.height;
  // Simple path: force a render then toDataURL at current buffer;
  // for scale>1, temporarily resize renderer, render, toDataURL, restore.
  const prevPR = this.renderer.getPixelRatio();
  this.renderer.setPixelRatio(scale);
  this._resize();
  this.renderer.render(this.scene, this.camera);
  const url = this.renderer.domElement.toDataURL("image/png");
  this.renderer.setPixelRatio(prevPR);
  this._resize();
  this.renderer.render(this.scene, this.camera);
  return url;
}
```

- [ ] **Step 2: HTML ids**

```html
<button type="button" class="btn btn--primary" id="cr-render">🎨 Render Structure</button>
<button type="button" class="btn btn--block" id="cr-hires">🖼 Save High-Res Image</button>
```

(Use existing HTML entities from the template if present — keep emoji entities consistent with file.)

Update hint to remove Align/3ds/Blender/hi-res “not yet”.

- [ ] **Step 3: Wire JS**

```javascript
dom.crRender?.addEventListener("click", () => refreshGeometry({ frame: true }));
dom.crHires?.addEventListener("click", () => {
  if (!activeCrystal) { setStatus("Load a crystal first.", true); return; }
  const url = ensureViewer().capturePNG(2);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${activeCrystal || "structure"}.png`;
  a.click();
  setStatus("Saved high-res PNG");
});
```

- [ ] **Step 4: Verify hooks**

```bash
rg -n "capturePNG|cr-render|cr-hires|preserveDrawingBuffer" tensorspec/web/static/js/viewers/viewer_3d.js tensorspec/web/static/js/crystal_suite.js tensorspec/web/templates/suites/crystal_suite.html
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/web/static/js/viewers/viewer_3d.js \
  tensorspec/web/static/js/crystal_suite.js \
  tensorspec/web/templates/suites/crystal_suite.html
git commit -m "$(cat <<'EOF'
feat(crystal): wire Render + high-res PNG capture

EOF
)"
```

---

### Task 6: Full test suite + push + Einstein

**Files:** none new (verification + deploy)

- [ ] **Step 1: Run Track A tests**

```bash
./TensorSpec_env/bin/python -m unittest \
  tests.test_miller_plane \
  tests.test_bz_overlay_field \
  tests.test_band_overlay \
  tests.test_wannier_upload \
  tests.test_scene_export \
  tests.test_crystal_geometry \
  -v
```

Expected: all PASS.

- [ ] **Step 2: Push**

```bash
git push -u origin HEAD
```

- [ ] **Step 3: Einstein pull + restart uvicorn**

```bash
ssh einstein 'cd /home/sandy/TensorSpec && git fetch && git checkout HTML_einstein_app && git pull --ff-only && (pkill -f "uvicorn tensorspec.web.server.app" || true); sleep 1; nohup TensorSpec_env/bin/uvicorn tensorspec.web.server.app:app --host 0.0.0.0 --port 8000 --reload > ~/tensorspec-uvicorn.log 2>&1 & sleep 3; curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/'
```

Expected: HTTP `200`, tip commit includes Track A commits.

- [ ] **Step 4: Smoke checklist (manual)**

1. Crystal → Align [001] moves camera  
2. DFT → upload hr → Overlay → Calculate shows red dashed  
3. Crystal → Export Blender downloads `.py`  
4. Save High-Res Image downloads PNG  
5. BZ with Overlay Crystal still works  

---

## Spec coverage self-review

| Spec item | Task |
|-----------|------|
| Align [hkl] | 1 |
| Drop `overlay_crystal` | 2 |
| W90 dual-solve + plot | 3 |
| Export checkboxes + 3ds/Blender | 4 |
| Render + hi-res PNG | 5 |
| Push + Einstein | 6 |
| Non-goals B–E | excluded |

No TBD placeholders. Names consistent: `overlay_wannier`, `overlay_bands`, `_scene_export_parts`, `lookAlongMiller`, `capturePNG`, `SceneExportRequest`.
