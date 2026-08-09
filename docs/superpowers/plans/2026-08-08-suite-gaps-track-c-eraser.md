# Suite Gaps Track C — Interactive Eraser — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Camera-locked continuous atom eraser with session omit set feeding 3ds/Blender/CIF/push.

**Architecture:** Client `Set` of geometry atom indices; viewer skips erased atoms/bonds; APIs accept `omit_atom_indices` and filter supercell geometry/structure before export or store.

**Tech Stack:** three.js CrystalViewer, FastAPI crystal router, pymatgen Structure, unittest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-suite-gaps-track-c-eraser-design.md`
- Branch: `HTML_einstein_app`
- Mode: 1+export; interaction A (camera lock + brush)
- Indices = current drawn geometry atom indices; OOR omit ignored
- Clear erase on geometry refresh
- Never merge to `main`
- Tests: `./TensorSpec_env/bin/python -m unittest …`

## File map

| File | Role |
|------|------|
| `tensorspec/web/server/geometry_filter.py` (or under `routers`/core) | Filter atoms/bonds + Structure by omit indices |
| `tensorspec/web/server/schemas.py` | `omit_atom_indices` on export/CIF/push models |
| `tensorspec/web/server/routers/crystal.py` | Apply filter on export/CIF/push |
| `tensorspec/web/static/js/viewers/viewer_3d.js` | Eraser state + pointer + draw skip |
| `tensorspec/web/static/js/crystal_suite.js` / HTML | Wire checkbox, payload, clear on refresh |
| `tensorspec/web/static/js/api.js` | CIF POST / push body fields |
| `tests/test_geometry_filter.py` | Filter unit tests |

---

### Task 1: Geometry filter helper + tests

**Files:**
- Create: `tensorspec/web/server/geometry_filter.py`
- Create: `tests/test_geometry_filter.py`

**Interfaces:**

```python
def normalize_omit_indices(omit: list[int] | None, n_atoms: int) -> set[int]:
    """Non-negative ints in [0, n_atoms); drop OOR/dupes."""

def filter_geometry_atoms_bonds(
    atoms: list,  # objects with no required fields beyond sequence
    bonds: list,  # objects with .i and .j
    omit: set[int],
) -> tuple[list, list]:
    """Keep atoms not in omit; bonds whose both ends survive; remap bond i,j to new compact indices."""

def filter_structure_by_omit(structure, omit: set[int]):
    """Return new Structure without sites at omit indices (OOR ignored)."""
```

- [ ] **Step 1: Failing tests**

```python
import unittest
from pymatgen.core import Lattice, Structure

from tensorspec.web.server.geometry_filter import (
    filter_geometry_atoms_bonds,
    filter_structure_by_omit,
    normalize_omit_indices,
)


class _Atom:
    def __init__(self, i):
        self.i = i


class _Bond:
    def __init__(self, i, j):
        self.i = i
        self.j = j


class TestGeometryFilter(unittest.TestCase):
    def test_normalize_drops_oor(self):
        self.assertEqual(normalize_omit_indices([-1, 0, 2, 2, 99], 3), {0, 2})

    def test_filter_bonds_remap(self):
        atoms = [_Atom(0), _Atom(1), _Atom(2)]
        bonds = [_Bond(0, 1), _Bond(1, 2)]
        a2, b2 = filter_geometry_atoms_bonds(atoms, bonds, {1})
        self.assertEqual(len(a2), 2)
        self.assertEqual([(b.i, b.j) for b in b2], [])  # both bonds touched 1

    def test_filter_structure(self):
        s = Structure(Lattice.cubic(4), ["Si", "Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]])
        out = filter_structure_by_omit(s, {1})
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
```

Adjust bond expectation if keeping 0–2 bond after removing middle: `_Bond(0,2)` should become `_Bond(0,1)` after remap — add that case:

```python
    def test_filter_keeps_remote_bond(self):
        atoms = [_Atom(0), _Atom(1), _Atom(2)]
        bonds = [_Bond(0, 2)]
        a2, b2 = filter_geometry_atoms_bonds(atoms, bonds, {1})
        self.assertEqual(len(a2), 2)
        self.assertEqual([(b.i, b.j) for b in b2], [(0, 1)])
```

- [ ] **Step 2: Implement helper; tests PASS; commit**

```bash
git commit -m "feat(crystal): filter geometry/structure by omit atom indices"
```

---

### Task 2: Schema + export / CIF POST / push

**Files:**
- Modify: `schemas.py` — `SceneExportRequest.omit_atom_indices: list[int] = []`
- Add: `CrystalCifRequest` (or reuse geometry knobs + omit) and extend `PushCrystalRequest` with omit + nx/ny/nz/basis matching Draw
- Modify: `crystal.py` export_scene; add POST cif; extend push
- Modify: `api.js` as needed
- Test: extend `tests/test_geometry_filter.py` or `tests/test_crystal_export_omit.py`

**PushCrystalRequest v1 fields:**

```python
store_as: str
omit_atom_indices: list[int] = []
nx: int = 1
ny: int = 1
nz: int = 1
basis: Literal["conventional", "primitive"] = "conventional"
```

When omit empty and 1×1×1 conventional: behavior = today’s copy.  
When omit or supercell: build same supercell as geometry endpoint, `filter_structure_by_omit`, store result.

**CIF POST** body same knobs + omit; return CIF bytes of filtered structure. Keep GET unfiltered.

**export_scene:** after geo build, convert atoms/bonds through `filter_geometry_atoms_bonds` using omit set (operate on geo model copies), then `_scene_export_parts`.

- [ ] **Step 1: Tests** for schema default + filter applied in a thin unit calling filter on a fake CrystalGeometry if easy; else schema + helper only and router smoke via TestClient if project has it.

- [ ] **Step 2: Implement; commit**

```bash
git commit -m "feat(crystal): omit_atom_indices on export CIF and push"
```

---

### Task 3: Viewer eraser

**Files:**
- Modify: `viewer_3d.js`

- [ ] **Step 1:** Add state + methods per spec §1.
- [ ] **Step 2:** In `_drawAtoms` / `_drawBonds`, skip erased indices (bonds if either end erased). Keep `_atomIndexByMesh` mapping to **original** indices for survivors.
- [ ] **Step 3:** Pointer handlers for eraser brush; `setEraserEnabled` toggles `controls.enabled`.
- [ ] **Step 4: Commit**

```bash
git commit -m "feat(crystal): interactive eraser brush with camera lock"
```

---

### Task 4: Suite UI + payloads

**Files:**
- Modify: `crystal_suite.html`, `crystal_suite.js`, `api.js`

- [ ] Enable `#cr-eraser` + hint + optional Reset.
- [ ] Wire enable/reset; clear erase in `refreshGeometry` success path (and stack render).
- [ ] `sceneExportPayload` includes `omit_atom_indices`.
- [ ] CIF: if omit non-empty (or always), POST filtered CIF with Draw knobs; else GET OK.
- [ ] Push: send omit + nx/ny/nz/basis from Draw controls.

- [ ] **Commit**

```bash
git commit -m "feat(crystal-ui): wire eraser to export CIF and push"
```

---

### Task 5: Push branch + Einstein pull

```bash
./TensorSpec_env/bin/python -m unittest tests.test_geometry_filter -v
git push -u origin HEAD
# ssh einstein: fetch/checkout HTML_einstein_app/pull; uvicorn reload if needed
```

Manual smoke: Draw → eraser on → brush atoms → export/CIF; re-Draw restores.

---

## Spec coverage

| Spec | Task |
|------|------|
| Filter helper | 1 |
| API omit | 2 |
| Viewer brush + lock | 3 |
| UI + payloads | 4 |
| Deploy | 5 |
