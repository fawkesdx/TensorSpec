# Crystal Cut-Plane Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Crystal Suite Tab 1 translucent Miller cut-plane guide (hkl + depth + color); atoms stay visible; no Align / camera-lock / hard clip.

**Architecture:** Pure Miller/depth math covered by Python unit tests (contract); same formulas in `viewer_3d.js` draw a translucent three.js plane; `crystal_suite.js` binds Tab 1 controls.

**Tech Stack:** three.js `CrystalViewer`, vanilla suite JS, unittest + numpy.

**Spec:** `docs/superpowers/specs/2026-08-08-crystal-cut-plane-guide-design.md`

## Global Constraints

- Branch: `HTML_einstein_app`; local clone → push → Einstein pull (path C).
- Guide plane only — never hide/clip atoms.
- Client-only display; no new geometry API fields.
- Align stays disabled; Lock to Camera disabled/removed; always [hkl].
- Use `TensorSpec_env/bin/python` for tests.

## File map

| File | Responsibility |
|------|----------------|
| `tensorspec/core/miller_plane.py` | Pure normal + depth offset (testable contract) |
| `tests/test_miller_plane.py` | Cubic (001) → +c; (0,0,0) rejected |
| `tensorspec/web/static/js/viewers/viewer_3d.js` | `setCutPlane` / draw translucent mesh |
| `tensorspec/web/templates/suites/crystal_suite.html` | Ids, color values, lock disabled, hint |
| `tensorspec/web/static/js/crystal_suite.js` | Bind controls → viewer |

---

### Task 1: Miller math + failing then passing tests

**Files:**
- Create: `tensorspec/core/miller_plane.py`
- Create: `tests/test_miller_plane.py`

**Interfaces:**
- Produces:
  - `miller_normal(cell_matrix: np.ndarray, hkl: tuple[int,int,int]) -> np.ndarray`
  - `plane_offset(cell_matrix: np.ndarray, normal: np.ndarray, depth_frac: float) -> np.ndarray`
  - `plane_size(cell_matrix: np.ndarray) -> float`
  - Raises `ValueError` on (0,0,0)

- [ ] **Step 1: Write failing tests**

```python
"""Miller cut-plane math contract (mirrors browser guide plane)."""
import unittest
import numpy as np

from tensorspec.core.miller_plane import miller_normal, plane_offset, plane_size


class TestMillerPlane(unittest.TestCase):
    def test_001_cubic_along_c(self):
        cell = np.eye(3) * 4.0  # a,b,c along axes, length 4
        n = miller_normal(cell, (0, 0, 1))
        np.testing.assert_allclose(n, [0, 0, 1], atol=1e-9)

    def test_100_cubic_along_a(self):
        cell = np.eye(3) * 4.0
        n = miller_normal(cell, (1, 0, 0))
        np.testing.assert_allclose(n, [1, 0, 0], atol=1e-9)

    def test_zero_hkl_raises(self):
        cell = np.eye(3)
        with self.assertRaises(ValueError):
            miller_normal(cell, (0, 0, 0))

    def test_depth_offset_scales(self):
        cell = np.eye(3) * 4.0
        n = miller_normal(cell, (0, 0, 1))
        # AABB along z is [0,4]; half extent along n = 2
        off = plane_offset(cell, n, depth_frac=1.0)
        np.testing.assert_allclose(off, [0, 0, 2.0], atol=1e-9)
        off0 = plane_offset(cell, n, depth_frac=0.0)
        np.testing.assert_allclose(off0, [0, 0, 0], atol=1e-9)

    def test_plane_size_positive(self):
        cell = np.eye(3) * 4.0
        self.assertGreater(plane_size(cell), 4.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run — expect FAIL (module missing)**

Run: `TensorSpec_env/bin/python -m unittest tests.test_miller_plane -v`  
Expected: ImportError / FAIL

- [ ] **Step 3: Implement `miller_plane.py`**

```python
"""Miller-plane helpers for the Crystal Suite cut-plane guide.

Mirrors the browser formulas in viewer_3d.js so orientation/depth stay
consistent with crystallography convention (n = h a* + k b* + l c*).
"""
from __future__ import annotations

import numpy as np


def miller_normal(cell_matrix: np.ndarray, hkl: tuple[int, int, int]) -> np.ndarray:
    h, k, l = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller index (0,0,0) is undefined")
    cell = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    a, b, c = cell[0], cell[1], cell[2]
    a_star = np.cross(b, c)
    b_star = np.cross(c, a)
    c_star = np.cross(a, b)
    n = h * a_star + k * b_star + l * c_star
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError("Miller normal has zero length")
    return n / norm


def plane_offset(cell_matrix: np.ndarray, normal: np.ndarray, depth_frac: float) -> np.ndarray:
    """Offset from cell center along normal; depth_frac in [-1, 1].

    Matches the viewer frame where atoms are drawn relative to geometry.center.
    depth_frac=0 → plane through cell center; ±1 → cell AABB faces along n.
    """
    cell = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    a, b, c = cell[0], cell[1], cell[2]
    corners = [
        np.zeros(3),
        a, b, c,
        a + b, a + c, b + c,
        a + b + c,
    ]
    center = sum(corners) / 8.0
    projs = [float(np.dot(p - center, n)) for p in corners]
    half = 0.5 * (max(projs) - min(projs))
    frac = float(np.clip(depth_frac, -1.0, 1.0))
    return n * (frac * half)


def plane_size(cell_matrix: np.ndarray) -> float:
    cell = np.asarray(cell_matrix, dtype=float).reshape(3, 3)
    a, b, c = cell[0], cell[1], cell[2]
    face_diags = [
        np.linalg.norm(a + b),
        np.linalg.norm(a + c),
        np.linalg.norm(b + c),
    ]
    return 1.2 * max(face_diags)
```

- [ ] **Step 4: Run — expect PASS**

Run: `TensorSpec_env/bin/python -m unittest tests.test_miller_plane -v`

- [ ] **Step 5: Commit**

```bash
git add tensorspec/core/miller_plane.py tests/test_miller_plane.py
git commit -m "$(cat <<'EOF'
feat(crystal): add Miller cut-plane math helpers

EOF
)"
```

---

### Task 2: Viewer `setCutPlane`

**Files:**
- Modify: `tensorspec/web/static/js/viewers/viewer_3d.js`

**Interfaces:**
- Consumes: `geometry.cell`, `geometry.center` (already on viewer after render)
- Produces: `setCutPlane({ h, k, l, depthFrac, color, visible })`

- [ ] **Step 1: Add helper functions (top of file or near exports)**

```javascript
function cross(u, v) {
  return [
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0],
  ];
}
function sub(u, v) { return [u[0]-v[0], u[1]-v[1], u[2]-v[2]]; }
function add(u, v) { return [u[0]+v[0], u[1]+v[1], u[2]+v[2]]; }
function scale(u, s) { return [u[0]*s, u[1]*s, u[2]*s]; }
function dot(u, v) { return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]; }
function norm(u) {
  const n = Math.hypot(u[0], u[1], u[2]);
  return n < 1e-12 ? null : scale(u, 1 / n);
}

/** @returns {number[]|null} unit normal or null if invalid */
export function millerNormal(cell, h, k, l) {
  if (h === 0 && k === 0 && l === 0) return null;
  const a = cell[0], b = cell[1], c = cell[2];
  const aS = cross(b, c), bS = cross(c, a), cS = cross(a, b);
  return norm(add(add(scale(aS, h), scale(bS, k)), scale(cS, l)));
}

export function planeOffsetFromCenter(cell, normal, depthFrac) {
  const a = cell[0], b = cell[1], c = cell[2];
  const corners = [
    [0,0,0], a, b, c,
    add(a,b), add(a,c), add(b,c), add(add(a,b),c),
  ];
  const center = scale(corners.reduce((s, p) => add(s, p), [0,0,0]), 1/8);
  const projs = corners.map((p) => dot(sub(p, center), normal));
  const half = 0.5 * (Math.max(...projs) - Math.min(...projs));
  const frac = Math.max(-1, Math.min(1, depthFrac));
  return scale(normal, frac * half);
}

export function planeSize(cell) {
  const a = cell[0], b = cell[1], c = cell[2];
  const diags = [
    Math.hypot(...add(a,b)),
    Math.hypot(...add(a,c)),
    Math.hypot(...add(b,c)),
  ];
  return 1.2 * Math.max(...diags);
}
```

- [ ] **Step 2: Constructor state**

```javascript
    this._cut = { h: 0, k: 0, l: 1, depthFrac: 0, color: "#00ffff", visible: false };
    this._cutMesh = null;
```

- [ ] **Step 3: `_syncCutPlane()` — call at end of `render()`**

```javascript
  _syncCutPlane() {
    if (this._cutMesh) {
      this.content.remove(this._cutMesh);
      this._cutMesh.geometry?.dispose();
      this._cutMesh.material?.dispose();
      this._cutMesh = null;
    }
    if (!this._cut.visible || !this.geometry) return;
    const { h, k, l, depthFrac, color } = this._cut;
    const n = millerNormal(this.geometry.cell, h, k, l);
    if (!n) return;
    // Atoms are drawn at position - center; plane lives in same frame.
    // Cell vectors are absolute; offset from center is planeOffsetFromCenter.
    const offset = planeOffsetFromCenter(this.geometry.cell, n, depthFrac);
    const size = planeSize(this.geometry.cell);
    const geom = new THREE.PlaneGeometry(size, size);
    const mat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.25,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geom, mat);
    // PlaneGeometry faces +Z; orient +Z → n
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(...n));
    mesh.position.set(...offset);
    this._cutMesh = mesh;
    this.content.add(mesh);
  }
```

Call `this._syncCutPlane()` at end of `render()` after atoms/bonds/cell (before frame).

- [ ] **Step 4: Public API**

```javascript
  setCutPlane({ h, k, l, depthFrac, color, visible } = {}) {
    if (h !== undefined) this._cut.h = Number(h) || 0;
    if (k !== undefined) this._cut.k = Number(k) || 0;
    if (l !== undefined) this._cut.l = Number(l) || 0;
    if (depthFrac !== undefined) this._cut.depthFrac = Number(depthFrac) || 0;
    if (color !== undefined) this._cut.color = color;
    if (visible !== undefined) this._cut.visible = Boolean(visible);
    this._syncCutPlane();
  }
```

- [ ] **Step 5: Commit**

```bash
git add tensorspec/web/static/js/viewers/viewer_3d.js
git commit -m "$(cat <<'EOF'
feat(viewer): draw translucent Miller cut-plane guide

EOF
)"
```

---

### Task 3: HTML ids + hint

**Files:**
- Modify: `tensorspec/web/templates/suites/crystal_suite.html`

- [ ] **Step 1: Replace Crystallography Tools block controls**

```html
<div class="form-row">
  <label>View [h k l]:</label>
  <div class="inline">
    <input class="field field--num" id="cr-h" type="number" min="-10" max="10" value="0">
    <input class="field field--num" id="cr-k" type="number" min="-10" max="10" value="0">
    <input class="field field--num" id="cr-l" type="number" min="-10" max="10" value="1">
    <button type="button" class="btn" disabled title="HTML viewer: not yet">Align</button>
  </div>
</div>
<label class="check"><input type="checkbox" id="cr-cut"> Show Cut Plane</label>
<div class="inline" style="margin-bottom:8px">
  <select class="field" id="cr-cut-color">
    <option value="#00ffff">cyan</option>
    <option value="#ff00ff">magenta</option>
    <option value="#ffff00">yellow</option>
    <option value="#ffffff">white</option>
    <option value="#808080">gray</option>
  </select>
  <select class="field" id="cr-cut-lock" disabled title="This pass: lock to [h k l] only">
    <option selected>Lock to [h k l]</option>
    <option disabled>Lock to Camera — not yet</option>
  </select>
</div>
<div class="form-row">
  <label for="cr-depth">Depth:</label>
  <input id="cr-depth" type="range" min="-100" max="100" value="0">
</div>
```

Default **l=1** so (0,0,1) is valid when user only enables the checkbox.

- [ ] **Step 2: Fix hint**

```html
<p class="hint">Align, 3ds/Blender, hi-res render: HTML viewer: not yet.</p>
```

- [ ] **Step 3: Commit**

```bash
git add tensorspec/web/templates/suites/crystal_suite.html
git commit -m "$(cat <<'EOF'
feat(crystal-ui): ids for cut-plane guide controls

EOF
)"
```

---

### Task 4: Wire `crystal_suite.js`

**Files:**
- Modify: `tensorspec/web/static/js/crystal_suite.js`

- [ ] **Step 1: Extend `dom`**

```javascript
    cutH: el("cr-h"),
    cutK: el("cr-k"),
    cutL: el("cr-l"),
    cut: el("cr-cut"),
    cutColor: el("cr-cut-color"),
    depth: el("cr-depth"),
```

- [ ] **Step 2: `applyCutPlane(view)`**

```javascript
function applyCutPlane(view) {
    if (!view) return;
    const h = Number(dom.cutH?.value) || 0;
    const k = Number(dom.cutK?.value) || 0;
    const l = Number(dom.cutL?.value) || 0;
    if (dom.cut?.checked && h === 0 && k === 0 && l === 0) {
        setStatus("Cut plane needs a non-zero [h k l].", true);
        view.setCutPlane({ visible: false });
        return;
    }
    if (!activeCrystal && dom.cut?.checked) {
        setStatus("Load a structure first.", true);
        return;
    }
    view.setCutPlane({
        h, k, l,
        depthFrac: (Number(dom.depth?.value) || 0) / 100,
        color: dom.cutColor?.value || "#00ffff",
        visible: Boolean(dom.cut?.checked),
    });
}
```

Call from `applyViewerChrome` or right after `view.render(...)` in `refreshGeometry` / stack / relax paths.

- [ ] **Step 3: Listeners** (with other Tab 1 binds)

```javascript
const syncCut = () => applyCutPlane(ensureViewer());
[dom.cutH, dom.cutK, dom.cutL, dom.cutColor].forEach((n) => n?.addEventListener("change", syncCut));
dom.cut?.addEventListener("change", syncCut);
dom.depth?.addEventListener("input", syncCut);
```

- [ ] **Step 4: Manual checklist** (note in report if no browser)

1. Load MoS₂ / layered CIF  
2. Set [0 0 1], enable Show Cut Plane → translucent sheet  
3. Slide Depth → plane moves; atoms remain  
4. Change color  
5. Set [0 0 0] + enable → status error, no plane  

- [ ] **Step 5: Commit**

```bash
git add tensorspec/web/static/js/crystal_suite.js
git commit -m "$(cat <<'EOF'
feat(crystal): wire cut-plane guide controls to viewer

EOF
)"
```

---

### Task 5: Verify + push + Einstein

- [ ] **Step 1: Tests**

Run: `TensorSpec_env/bin/python -m unittest tests.test_miller_plane tests.test_crystal_geometry -v`  
Expected: all PASS

- [ ] **Step 2: Push**

```bash
git push -u origin HEAD
```

- [ ] **Step 3: Einstein cmds in report**

```bash
cd ~/TensorSpec
git fetch && git checkout HTML_einstein_app && git pull
uvicorn tensorspec.web.server.app:app --reload --host 0.0.0.0 --port 8000
```

---

## Self-review (plan vs spec)

| Spec item | Task |
|-----------|------|
| Translucent [hkl] plane | 2, 4 |
| Depth along normal | 1, 2, 4 |
| Atoms stay | 2 (no clip) |
| Color | 3, 4 |
| Ids + lock disabled | 3 |
| Hint update | 3 |
| (0,0,0) error | 1, 4 |
| Unit test cubic 001 | 1 |
| Align out of scope | 3 (stays disabled) |
| Push / Einstein | 5 |
