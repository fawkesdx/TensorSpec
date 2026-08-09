# Suite Gaps Track C — Coordination Polyhedra — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Crystal Suite Connections → Polyhedra: server coordination hulls, client translucent draw, no sticks; exports ignore faces.

**Architecture:** `show_polyhedra` on geometry request; `CrystalEngine.compute_coordination_polyhedra` via scipy ConvexHull; viewer `_drawPolyhedra`; suite wires `#cr-conn`.

**Tech Stack:** pymatgen, scipy.spatial.ConvexHull, three.js, unittest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-suite-gaps-track-c-polyhedra-design.md`
- Branch: `HTML_einstein_app`
- ≥4 neighbors for a hull; skip Qhull failures
- Viewer-only (no SceneExporter faces)
- Eraser: skip hull if center or any `vertex_atom_indices` erased
- Tests: `./TensorSpec_env/bin/python -m unittest …`

## File map

| File | Role |
|------|------|
| `tensorspec/core/crystallography.py` | `compute_coordination_polyhedra` |
| `schemas.py` | `Polyhedron`, `GeometryRequest.show_polyhedra`, `CrystalGeometry.polyhedra` |
| `routers/crystal.py` | Fill polyhedra when requested |
| `viewer_3d.js` | `_drawPolyhedra` |
| `crystal_suite.js` / HTML | Enable option; send flags |
| `tests/test_coordination_polyhedra.py` | Unit tests |

---

### Task 1: Compute helper + tests

**Files:**
- Modify: `tensorspec/core/crystallography.py`
- Create: `tests/test_coordination_polyhedra.py`

**Interface:**

```python
@staticmethod
def compute_coordination_polyhedra(
    coords: np.ndarray,  # (N,3)
    bonds_i: np.ndarray,
    bonds_j: np.ndarray,
) -> list[dict]:
    """Return list of {center, vertices, simplices, vertex_atom_indices}. Skip <4 neighbors / QhullError."""
```

- [ ] **Step 1: Failing test** — build simple structure with known bonds (e.g. FCC-like or complete graph on 5 points); assert at least one polyhedron with simplices; assert atom with 2 neighbors yields none for that center.

- [ ] **Step 2: Implement; PASS; commit**

```bash
git commit -m "feat(crystal): compute coordination polyhedra via ConvexHull"
```

---

### Task 2: Schema + geometry endpoint

**Files:** `schemas.py`, `crystal.py` router (`_geometry_from_structure` / geometry handler)

- Add `Polyhedron` model; `CrystalGeometry.polyhedra: list[Polyhedron] = []`
- `GeometryRequest.show_polyhedra: bool = False`
- When `show_polyhedra`, after bonds computed, call helper; attach to geometry. When not, `polyhedra=[]`. Bonds list empty if `show_bonds` false (existing).

- [ ] Test schema defaults + geometry includes polyhedra when flag true (TestClient or unit on `_geometry_from_structure` if accessible).

- [ ] Commit: `feat(crystal): geometry API returns coordination polyhedra`

---

### Task 3: Viewer draw

**Files:** `viewer_3d.js`

- `render` option `showPolyhedra`
- `_drawPolyhedra`: skip if erased; MeshStandardMaterial opacity 0.3, doubleSide, depthWrite false; color `elementColor` of center atom
- Commit: `feat(crystal): draw coordination polyhedra in viewer`

---

### Task 4: Suite UI

**Files:** HTML + `crystal_suite.js`

- Enable polyhedra option + short hint
- Geometry payload: `show_bonds: conn==='bonds'`, `show_polyhedra: conn==='polyhedra'`
- `viewer.render({ showBonds, showPolyhedra: conn==='polyhedra', ...})`
- Commit: `feat(crystal-ui): enable polyhedra connection mode`

---

### Task 5: Push + Einstein pull

```bash
./TensorSpec_env/bin/python -m unittest tests.test_coordination_polyhedra -v
git push -u origin HEAD
# einstein pull HTML_einstein_app
```

---

## Spec coverage

| Spec | Task |
|------|------|
| ConvexHull helper ≥4 neighbors | 1 |
| API schema + geometry | 2 |
| Viewer draw + eraser skip | 3 |
| UI wire | 4 |
| Deploy | 5 |
