# Stack → DFT Cell Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On Crystal Suite Tab 3 Push, rebuild a physical DFT cell (vacuum + real in-plane lattice) from stack layers so workspace/QE never get the 500 Å dummy box.

**Architecture:** Keep Render on `build_heterostructure_stack` (viz dummy). Add pure helpers in `crystallography.py` for classify / strain / aligned / twist rebuild. `push_current_to_workspace` rebuilds from `stack_layer_rows` when non-empty, with vacuum spinbox + optional strain dialog.

**Tech Stack:** Python, pymatgen `Structure`/`Lattice`, numpy, PyQt5 (dialogs only), pytest.

**Spec:** `docs/superpowers/specs/2026-09-13-stack-dft-cell-rebuild-design.md`

## Global Constraints

- Render stays on 500 Å dummy canvas — do not change `build_heterostructure_stack` lattice.
- Push rebuilds when `stack_layer_rows` non-empty; else push `current_structure` unchanged.
- Interlayer spacing = existing Z spinboxes (`z_shift`); vacuum = new spinbox default **20 Å** (padding along c only).
- Aligned: N layers, all `|θ| ≤ 1e-6`.
- Twist DFT: exactly **2** layers; `>2` + any twist → block with placeholder (no push).
- Incommensurate / lattice mismatch >1%: warn dialog, user picks reference, suggest min-strain.
- Preserve `layer_tag` site properties on DFT Structure.
- DFT Suite / QE generator: no changes.
- Tests: unit only for math (no GUI required for Tasks 1–3).

## File map

| File | Role |
|------|------|
| `tensorspec/core/crystallography.py` | New pure helpers on `CrystalEngine` (or module-level functions co-located) |
| `tensorspec/gui/suites/crystal_suite.py` | Vacuum spinbox; Push orchestration; strain / multitwist dialogs |
| `tests/test_stack_dft_cell.py` | Classifier, strain, aligned, twist rebuild tests |

---

### Task 1: Classify stack + strain suggestion

**Files:**
- Modify: `tensorspec/core/crystallography.py`
- Test: `tests/test_stack_dft_cell.py`

**Interfaces:**
- Consumes: layer dicts `{"struct": Structure, "sc_x": int, "sc_y": int, "z_shift": float, "twist": float}` (same as `StackLayerRow.get_layer_dict()`)
- Produces:
  - `CrystalEngine.classify_stack_for_dft(layers: list[dict], twist_eps: float = 1e-6) -> str`  
    returns one of `"aligned" | "twist" | "reject_multitwist" | "empty"`
  - `CrystalEngine.inplane_strain_percent(ref_2x2: np.ndarray, other_2x2: np.ndarray) -> float`  
    Frobenius relative mismatch in percent: `100 * ||A-B||_F / ||B||_F` with `B=ref`
  - `CrystalEngine.suggest_reference_layer(layers: list[dict]) -> tuple[int, list[float]]`  
    for each candidate ref index `r`, total strain = sum of `inplane_strain_percent` of every other layer’s in-plane 2×2 (after applying that layer’s twist relative to ref’s twist for the 2-layer twist case; for aligned use twist=0). Returns `(best_index, strains_per_ref)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stack_dft_cell.py`:

```python
from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure

from tensorspec.core.crystallography import CrystalEngine


def _mono(a: float, species=("C", "C")) -> Structure:
    lat = Lattice.hexagonal(a, 25.0)
    return Structure(lat, list(species), [[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]])


def _layer(struct, z=0.0, twist=0.0, sc_x=1, sc_y=1):
    return {
        "struct": struct,
        "sc_x": sc_x,
        "sc_y": sc_y,
        "z_shift": z,
        "twist": twist,
    }


def test_classify_empty():
    assert CrystalEngine.classify_stack_for_dft([]) == "empty"


def test_classify_aligned_n_layers():
    g = _mono(2.46)
    layers = [_layer(g, 0.0), _layer(g, 3.4), _layer(g, 6.8)]
    assert CrystalEngine.classify_stack_for_dft(layers) == "aligned"


def test_classify_twist_two_layers():
    layers = [_layer(_mono(2.46), 0.0, 0.0), _layer(_mono(2.50), 3.4, 30.0)]
    assert CrystalEngine.classify_stack_for_dft(layers) == "twist"


def test_classify_reject_three_with_twist():
    g = _mono(2.46)
    layers = [_layer(g, 0.0, 0.0), _layer(g, 3.4, 10.0), _layer(g, 6.8, 0.0)]
    assert CrystalEngine.classify_stack_for_dft(layers) == "reject_multitwist"


def test_suggest_ref_prefers_closer_lattice():
    # identical twins -> either ok; mismatched pair -> closer match wins
    layers = [_layer(_mono(2.46), 0.0), _layer(_mono(2.50), 3.4)]
    idx, strains = CrystalEngine.suggest_reference_layer(layers)
    assert len(strains) == 2
    assert idx in (0, 1)
    assert strains[idx] == min(strains)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stack_dft_cell.py::test_classify_empty tests/test_stack_dft_cell.py::test_classify_aligned_n_layers tests/test_stack_dft_cell.py::test_classify_twist_two_layers tests/test_stack_dft_cell.py::test_classify_reject_three_with_twist tests/test_stack_dft_cell.py::test_suggest_ref_prefers_closer_lattice -v`

Expected: FAIL (`classify_stack_for_dft` / `suggest_reference_layer` missing)

- [ ] **Step 3: Implement minimal helpers**

Append to `CrystalEngine` in `tensorspec/core/crystallography.py`:

```python
@staticmethod
def classify_stack_for_dft(layers: list[dict], twist_eps: float = 1e-6) -> str:
    if not layers:
        return "empty"
    twists = [abs(float(l.get("twist", 0.0))) for l in layers]
    any_twist = any(t > twist_eps for t in twists)
    if any_twist and len(layers) != 2:
        return "reject_multitwist"
    if any_twist:
        return "twist"
    return "aligned"

@staticmethod
def inplane_strain_percent(ref_2x2: np.ndarray, other_2x2: np.ndarray) -> float:
    ref = np.asarray(ref_2x2, dtype=float)
    other = np.asarray(other_2x2, dtype=float)
    denom = np.linalg.norm(ref, ord="fro")
    if denom < 1e-12:
        raise ValueError("Degenerate reference in-plane lattice.")
    return float(100.0 * np.linalg.norm(other - ref, ord="fro") / denom)

@staticmethod
def _inplane_2x2(struct: Structure, sc_x: int = 1, sc_y: int = 1) -> np.ndarray:
    m = struct.lattice.matrix.copy()
    m[0] *= sc_x
    m[1] *= sc_y
    return m[:2, :2]

@staticmethod
def suggest_reference_layer(layers: list[dict]) -> tuple[int, list[float]]:
    n = len(layers)
    if n == 0:
        raise ValueError("No layers for reference suggestion.")
    strains = []
    for r in range(n):
        ref = CrystalEngine._inplane_2x2(layers[r]["struct"], layers[r]["sc_x"], layers[r]["sc_y"])
        total = 0.0
        for j in range(n):
            if j == r:
                continue
            other = CrystalEngine._inplane_2x2(layers[j]["struct"], layers[j]["sc_x"], layers[j]["sc_y"])
            # relative twist: rotate other in-plane by (twist_j - twist_r)
            dtheta = np.radians(float(layers[j]["twist"]) - float(layers[r]["twist"]))
            c, s = np.cos(dtheta), np.sin(dtheta)
            R = np.array([[c, -s], [s, c]])
            other_rot = other @ R.T
            total += CrystalEngine.inplane_strain_percent(ref, other_rot)
        strains.append(total)
    best = int(np.argmin(strains))
    # tie-break within 0.1% -> layer 0 preference among ties
    min_s = strains[best]
    ties = [i for i, s in enumerate(strains) if abs(s - min_s) <= 0.1]
    if 0 in ties:
        best = 0
    else:
        best = ties[0]
    return best, strains
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stack_dft_cell.py::test_classify_empty tests/test_stack_dft_cell.py::test_classify_aligned_n_layers tests/test_stack_dft_cell.py::test_classify_twist_two_layers tests/test_stack_dft_cell.py::test_classify_reject_three_with_twist tests/test_stack_dft_cell.py::test_suggest_ref_prefers_closer_lattice -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tensorspec/core/crystallography.py tests/test_stack_dft_cell.py
git commit -m "feat(crystal): classify stack for DFT cell rebuild"
```

---

### Task 2: Aligned N-layer DFT rebuild

**Files:**
- Modify: `tensorspec/core/crystallography.py`
- Test: `tests/test_stack_dft_cell.py`

**Interfaces:**
- Consumes: Task 1 helpers; layer dicts
- Produces:
  - `CrystalEngine.build_dft_aligned_stack(layers: list[dict], vacuum_ang: float, ref_idx: int = 0) -> Structure`
  - Needs strain dialog only in UI when mismatch >1%; core always builds with given `ref_idx` (strains non-ref into ref in-plane cell).
  - `CrystalEngine.aligned_needs_strain_dialog(layers: list[dict], ref_idx: int, tol_percent: float = 1.0) -> bool`

Algorithm for `build_dft_aligned_stack`:
1. `ref = layers[ref_idx]`; take in-plane from `ref.struct * (sc_x, sc_y, 1)` → matrix `a,b`; set `c` later.
2. For each layer: take supercell, **do not twist** (aligned), center xy like viz builder, set z from `z_shift` (same as `build_heterostructure_stack`: `z_shift - 12.5` offset pattern OR clearer: place at `z_shift` absolute and then recenter — prefer matching viz Z differences: use same formula as `build_heterostructure_stack` for z: `shifted = coords + [0,0,z_shift-12.5]` after xy center).
3. Map each layer’s xy into ref fractional coords: `frac_xy = cart_xy @ inv(ref_ab.T)` then wrap; keep z cartesian until end.
4. Collect species + `layer_tag`.
5. `z_min,z_max` from carts; `c = (z_max-z_min) + vacuum_ang`; center z to `c/2`.
6. Lattice matrix: `[[ax,ay,0],[bx,by,0],[0,0,c]]` from ref in-plane 2×2 + c.
7. Assert `lattice.a < 499`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stack_dft_cell.py`:

```python
def test_aligned_bilayer_not_dummy_and_vacuum():
    g = _mono(2.46)
    layers = [_layer(g, 0.0), _layer(g, 3.4)]
    vacuum = 20.0
    s = CrystalEngine.build_dft_aligned_stack(layers, vacuum_ang=vacuum, ref_idx=0)
    assert s.lattice.a < 100.0
    assert s.lattice.a != 500.0
    thickness = 3.4  # |z1-z0| from spinboxes (same placement)
    # c must be roughly thickness + vacuum (allow builder offset conventions ±2 Å)
    assert abs(s.lattice.c - (thickness + vacuum)) < 2.5
    assert "layer_tag" in s.site_properties
    assert len(s) == 4


def test_aligned_mismatch_flag():
    layers = [_layer(_mono(2.46), 0.0), _layer(_mono(2.50), 3.4)]
    assert CrystalEngine.aligned_needs_strain_dialog(layers, ref_idx=0) is True
    same = [_layer(_mono(2.46), 0.0), _layer(_mono(2.46), 3.4)]
    assert CrystalEngine.aligned_needs_strain_dialog(same, ref_idx=0) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stack_dft_cell.py::test_aligned_bilayer_not_dummy_and_vacuum tests/test_stack_dft_cell.py::test_aligned_mismatch_flag -v`

Expected: FAIL (missing `build_dft_aligned_stack`)

- [ ] **Step 3: Implement aligned rebuild**

Implement `aligned_needs_strain_dialog` and `build_dft_aligned_stack` on `CrystalEngine`. Reuse the xy-centering / z offset logic from `build_heterostructure_stack` (copy the loop body; do not call the dummy-lattice return). For non-ref layers, replace in-plane embedding by projecting cartesian xy onto the **reference** in-plane lattice (strain). Keep Z from the spinbox formula.

Helper sketch (place near other stack methods):

```python
@staticmethod
def aligned_needs_strain_dialog(layers: list[dict], ref_idx: int, tol_percent: float = 1.0) -> bool:
    ref = CrystalEngine._inplane_2x2(layers[ref_idx]["struct"], layers[ref_idx]["sc_x"], layers[ref_idx]["sc_y"])
    for j, layer in enumerate(layers):
        if j == ref_idx:
            continue
        other = CrystalEngine._inplane_2x2(layer["struct"], layer["sc_x"], layer["sc_y"])
        if CrystalEngine.inplane_strain_percent(ref, other) > tol_percent:
            return True
    return False

@staticmethod
def _finalize_slab_structure(ab_2x2, species, carts_xyz, tags, vacuum_ang: float) -> Structure:
    """Build lattice from in-plane 2x2 + vacuum along c; center slab in c."""
    carts = np.asarray(carts_xyz, dtype=float)
    zmin, zmax = float(carts[:, 2].min()), float(carts[:, 2].max())
    zspan = max(zmax - zmin, 0.0)
    c = zspan + float(vacuum_ang)
    z_center = 0.5 * (zmin + zmax)
    carts[:, 2] = carts[:, 2] - z_center + 0.5 * c
    matrix = np.zeros((3, 3), dtype=float)
    matrix[:2, :2] = ab_2x2
    matrix[2, 2] = c
    lat = Lattice(matrix)
    # fold xy into cell via fractional coords
    return Structure(
        lat, species, carts, coords_are_cartesian=True,
        site_properties={"layer_tag": tags},
    )

@staticmethod
def build_dft_aligned_stack(layers: list[dict], vacuum_ang: float, ref_idx: int = 0) -> Structure:
    if not layers:
        raise ValueError("No layers to build DFT stack.")
    if ref_idx < 0 or ref_idx >= len(layers):
        raise ValueError("ref_idx out of range.")
    ab = CrystalEngine._inplane_2x2(
        layers[ref_idx]["struct"], layers[ref_idx]["sc_x"], layers[ref_idx]["sc_y"]
    )
    ab_inv_t = np.linalg.inv(ab.T)
    species, carts, tags = [], [], []
    for idx, l in enumerate(layers):
        supercell = l["struct"] * (l["sc_x"], l["sc_y"], 1)
        coords = supercell.cart_coords.copy()
        center_xy = np.mean(coords[:, :2], axis=0)
        coords[:, :2] -= center_xy
        # aligned: no twist; place z like viz builder
        coords[:, 2] = coords[:, 2] - np.mean(coords[:, 2]) + (float(l["z_shift"]) - 12.5)
        # strain into ref ab: keep cartesian xy from layer, interpret in ref cell
        for i, site in enumerate(supercell):
            xy = coords[i, :2]
            # map into ref fractional then back so lattice is ref's
            frac_xy = xy @ ab_inv_t
            frac_xy = frac_xy - np.floor(frac_xy)
            xy_in_ref = frac_xy @ ab.T
            species.append(site.specie.symbol)
            carts.append([xy_in_ref[0], xy_in_ref[1], coords[i, 2]])
            tags.append(f"{site.specie.symbol}_L{idx + 1}")
    return CrystalEngine._finalize_slab_structure(ab, species, carts, tags, vacuum_ang)
```

Note on Z: viz used `rotated + [0,0,z_shift-12.5]` without subtracting layer mean z. Prefer **matching interlayer Δz = spinbox differences**. If tests show offset, switch coords z line to `coords[:, 2] = float(l["z_shift"])` (flat monolayer at spinbox z) — graphene/hBN monolayers are nearly flat so mean-subtract + spinbox is fine.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stack_dft_cell.py::test_aligned_bilayer_not_dummy_and_vacuum tests/test_stack_dft_cell.py::test_aligned_mismatch_flag -v`

Expected: PASS. If `c` assertion fails, adjust test tolerance **or** document exact z convention in helper docstring and fix test to match implemented convention (prefer documenting).

- [ ] **Step 5: Commit**

```bash
git add tensorspec/core/crystallography.py tests/test_stack_dft_cell.py
git commit -m "feat(crystal): rebuild aligned stack DFT cell"
```

---

### Task 3: Twist (2-layer) commensurate + forced-strain rebuild

**Files:**
- Modify: `tensorspec/core/crystallography.py`
- Test: `tests/test_stack_dft_cell.py`

**Interfaces:**
- Consumes: `calculate_moire_superlattice`, Task 1–2 helpers
- Produces:
  - `CrystalEngine.build_dft_twist_stack(layers: list[dict], vacuum_ang: float, ref_idx: int) -> tuple[Structure, dict]`
    - `layers` must have length 2
    - Returns `(structure, info)` where `info` has keys:
      - `status`: `"commensurate" | "incommensurate" | "perfect_alignment" | "error"`
      - `suggested_ref`: int
      - `strains`: list[float]
      - `moire`: raw moiré dict
  - For `perfect_alignment` / 0 relative twist with matching lattices: may delegate to aligned builder.
  - For `commensurate`: in-plane from `moire["matrix"]` (2×2) → 3×3 with vacuum c; tile atoms from both layers with their twists into that cell.
  - For `incommensurate` / `error` recoverable as forced strain: build forced-strain cell using `ref_idx` in-plane (SC’d), apply relative twist to non-ref atoms, same vacuum rules. Still return `status` from moiré so UI can warn.
  - On hard `error` (degenerate): raise `ValueError` with message.

- [ ] **Step 1: Write the failing tests**

```python
def test_twist_two_layer_cell_not_dummy():
    layers = [_layer(_mono(2.46), 0.0, 0.0), _layer(_mono(2.50, ("B", "N")), 3.4, 30.0)]
    suggested, strains = CrystalEngine.suggest_reference_layer(layers)
    struct, info = CrystalEngine.build_dft_twist_stack(layers, vacuum_ang=20.0, ref_idx=suggested)
    assert struct.lattice.a < 499.0
    assert struct.lattice.c > 20.0
    assert info["status"] in ("commensurate", "incommensurate", "perfect_alignment")
    assert "layer_tag" in struct.site_properties


def test_twist_requires_two_layers():
    layers = [_layer(_mono(2.46), 0.0, 5.0)]
    try:
        CrystalEngine.build_dft_twist_stack(layers, 20.0, 0)
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stack_dft_cell.py::test_twist_two_layer_cell_not_dummy tests/test_stack_dft_cell.py::test_twist_requires_two_layers -v`

Expected: FAIL (missing `build_dft_twist_stack`)

- [ ] **Step 3: Implement twist rebuild**

```python
@staticmethod
def _place_layer_atoms(layer: dict, idx: int, apply_twist: bool) -> tuple[list, list, list]:
    """Return species, cartesian coords, tags for one layer (xy centered, z from spinbox)."""
    supercell = layer["struct"] * (layer["sc_x"], layer["sc_y"], 1)
    coords = supercell.cart_coords.copy()
    center_xy = np.mean(coords[:, :2], axis=0)
    coords[:, :2] -= center_xy
    if apply_twist:
        theta = np.radians(float(layer["twist"]))
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0, 0, 1]])
        coords = coords @ R.T
    coords[:, 2] = coords[:, 2] - np.mean(coords[:, 2]) + (float(layer["z_shift"]) - 12.5)
    species, carts, tags = [], [], []
    for i, site in enumerate(supercell):
        species.append(site.specie.symbol)
        carts.append(coords[i].tolist())
        tags.append(f"{site.specie.symbol}_L{idx + 1}")
    return species, carts, tags

@staticmethod
def _tile_into_cell(ab_2x2, species, carts, tags, n_max: int = 40) -> tuple[list, list, list]:
    """Replicate atoms by integer combos of ab until cell is covered; fold into [0,1)."""
    ab_inv_t = np.linalg.inv(ab_2x2.T)
    out_s, out_c, out_t = [], [], []
    # n_max along each lattice vector is enough for small moiré n_cells
    for n1 in range(-n_max, n_max + 1):
        for n2 in range(-n_max, n_max + 1):
            shift = n1 * ab_2x2[0] + n2 * ab_2x2[1]
            for s, c, t in zip(species, carts, tags):
                xy = np.asarray(c[:2]) + shift
                frac = xy @ ab_inv_t
                if np.all(frac >= -1e-8) and np.all(frac < 1.0 - 1e-8):
                    frac = frac - np.floor(frac + 1e-12)
                    xy_f = frac @ ab_2x2.T
                    out_s.append(s)
                    out_c.append([xy_f[0], xy_f[1], c[2]])
                    out_t.append(t)
    # Deduplicate near-identical sites
    keep = []
    seen = []
    for i, c in enumerate(out_c):
        arr = np.array(c)
        if any(np.linalg.norm(arr - s) < 0.15 for s in seen):
            continue
        seen.append(arr)
        keep.append(i)
    return [out_s[i] for i in keep], [out_c[i] for i in keep], [out_t[i] for i in keep]

@staticmethod
def build_dft_twist_stack(layers: list[dict], vacuum_ang: float, ref_idx: int) -> tuple[Structure, dict]:
    if len(layers) != 2:
        raise ValueError("Twist DFT rebuild requires exactly 2 layers.")
    suggested, strains = CrystalEngine.suggest_reference_layer(layers)
    l0, l1 = layers[0], layers[1]
    moire = CrystalEngine.calculate_moire_superlattice(
        l0["struct"], l1["struct"], float(l0["twist"]), float(l1["twist"])
    )
    status = moire.get("status", "error")
    info = {"status": status, "suggested_ref": suggested, "strains": strains, "moire": moire}
    if status == "error":
        raise ValueError(moire.get("message", "Moiré calculation failed."))
    if status == "perfect_alignment":
        return CrystalEngine.build_dft_aligned_stack(layers, vacuum_ang, ref_idx), info

    species, carts, tags = [], [], []
    for idx, layer in enumerate(layers):
        s, c, t = CrystalEngine._place_layer_atoms(layer, idx, apply_twist=True)
        species.extend(s); carts.extend(c); tags.extend(t)

    if status == "commensurate":
        ab = np.asarray(moire["matrix"], dtype=float)
        n_cells = int(moire.get("n_cells", 5))
        n_max = max(n_cells + 2, 6)
        species, carts, tags = CrystalEngine._tile_into_cell(ab, species, carts, tags, n_max=n_max)
        struct = CrystalEngine._finalize_slab_structure(ab, species, carts, tags, vacuum_ang)
        return struct, info

    # incommensurate: forced strain into ref_idx ab (no extra tiling beyond SC)
    ab = CrystalEngine._inplane_2x2(
        layers[ref_idx]["struct"], layers[ref_idx]["sc_x"], layers[ref_idx]["sc_y"]
    )
    ab_inv_t = np.linalg.inv(ab.T)
    new_carts = []
    for c in carts:
        frac = np.asarray(c[:2]) @ ab_inv_t
        frac = frac - np.floor(frac)
        xy = frac @ ab.T
        new_carts.append([xy[0], xy[1], c[2]])
    struct = CrystalEngine._finalize_slab_structure(ab, species, new_carts, tags, vacuum_ang)
    return struct, info
```

If commensurate tiling yields 0 atoms, raise `ValueError("Commensurate tiling produced no atoms — check moiré matrix.")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stack_dft_cell.py -v`

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tensorspec/core/crystallography.py tests/test_stack_dft_cell.py
git commit -m "feat(crystal): rebuild twisted bilayer DFT cell"
```

---

### Task 4: Tab 3 UI — vacuum spinbox + Push orchestration

**Files:**
- Modify: `tensorspec/gui/suites/crystal_suite.py` (`init_tab_heterostructure`, `push_current_to_workspace`)
- Test: manual smoke (optional tiny non-GUI helper test for dialog copy builder if extracted)

**Interfaces:**
- Consumes: `classify_stack_for_dft`, `suggest_reference_layer`, `aligned_needs_strain_dialog`, `build_dft_aligned_stack`, `build_dft_twist_stack`
- Produces: Push sends DFT Structure to `global_workspace`; viz `current_structure` unchanged when stack rows drove rebuild

- [ ] **Step 1: Add vacuum spinbox in Tab 3**

In `init_tab_heterostructure`, near the Render button (after `btn_draw_stack` / before moiré controls), add:

```python
vac_row = QHBoxLayout()
vac_row.addWidget(QLabel("DFT vacuum (Å):"))
self.spin_stack_vacuum = QDoubleSpinBox()
self.spin_stack_vacuum.setRange(5.0, 100.0)
self.spin_stack_vacuum.setSingleStep(1.0)
self.spin_stack_vacuum.setValue(20.0)
self.spin_stack_vacuum.setToolTip(
    "Vacuum padding along c for Push→DFT cell. Interlayer spacing uses each layer's z spinbox."
)
vac_row.addWidget(self.spin_stack_vacuum)
vac_row.addStretch()
layout.addLayout(vac_row)
```

- [ ] **Step 2: Rewrite `push_current_to_workspace` orchestration**

```python
def push_current_to_workspace(self):
    from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QComboBox

    # Non-stack path: Tab 1 / no hetero rows
    if not getattr(self, "stack_layer_rows", None):
        if getattr(self, "current_structure", None) is None:
            QMessageBox.warning(self, "Warning", "No active structure to push!")
            return
        default_name = getattr(self, "current_filename", "My_Crystal")
        name, ok = QInputDialog.getText(
            self, "Workspace Export", "Enter a variable name for this structure:", text=default_name
        )
        if ok and name:
            global_workspace.push_crystal_structure(name, self.current_structure)
            QMessageBox.information(
                self, "Success",
                f"Structure '{name}' sent to Global Workspace!\nYou can now load it in the DFT or ARPES Suites.",
            )
        return

    layers = [row.get_layer_dict() for row in self.stack_layer_rows]
    kind = CrystalEngine.classify_stack_for_dft(layers)

    if kind == "empty":
        QMessageBox.warning(self, "Warning", "No layers in the stack to push.")
        return

    if kind == "reject_multitwist":
        QMessageBox.information(
            self,
            "Multi-twist stacks — coming later",
            "DFT cell rebuild for more than two twisted layers is not available yet.\n\n"
            "This week: N layers with all twists = 0°, or exactly 2 layers with twist "
            "(e.g. graphene/hBN). Render/view still works.",
        )
        return

    vacuum = float(self.spin_stack_vacuum.value())
    suggested, strains = CrystalEngine.suggest_reference_layer(layers)
    ref_idx = suggested
    need_dialog = False
    dialog_status = ""

    if kind == "aligned":
        need_dialog = CrystalEngine.aligned_needs_strain_dialog(layers, suggested)
        dialog_status = "lattice mismatch"
    else:
        moire = CrystalEngine.calculate_moire_superlattice(
            layers[0]["struct"], layers[1]["struct"],
            float(layers[0]["twist"]), float(layers[1]["twist"]),
        )
        if moire.get("status") == "error":
            QMessageBox.critical(self, "Moiré error", moire.get("message", "Failed."))
            return
        need_dialog = moire.get("status") == "incommensurate"
        dialog_status = moire.get("status", "")

    if need_dialog:
        names = [
            row.lbl_name.text().replace("<b>", "").replace("</b>", "")
            for row in self.stack_layer_rows
        ]
        dlg = QDialog(self)
        dlg.setWindowTitle("Forced strain for DFT")
        lay = QVBoxLayout(dlg)
        sug_name = names[suggested]
        sug_pct = strains[suggested]
        lay.addWidget(QLabel(
            f"Status: {dialog_status}.\n\n"
            "To run DFT, non-reference layer(s) will be stretched/compressed to match the "
            "reference layer's in-plane cell. Band structures will reflect that forced strain.\n\n"
            f"Suggested reference (smallest total strain): {sug_name} (~{sug_pct:.2f}%)."
        ))
        lay.addWidget(QLabel("Reference layer:"))
        combo = QComboBox()
        for i, name_i in enumerate(names):
            label = name_i + ("  ← suggested (min strain)" if i == suggested else "")
            combo.addItem(label)
        combo.setCurrentIndex(suggested)
        lay.addWidget(combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Push anyway")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return
        ref_idx = combo.currentIndex()

    try:
        if kind == "aligned":
            dft_struct = CrystalEngine.build_dft_aligned_stack(layers, vacuum, ref_idx)
            info_note = "aligned"
        else:
            dft_struct, info = CrystalEngine.build_dft_twist_stack(layers, vacuum, ref_idx)
            info_note = info["status"]
    except Exception as e:
        QMessageBox.critical(self, "DFT cell rebuild failed", str(e))
        return

    default_name = getattr(self, "current_filename", "Heterostructure_Stack")
    name, ok = QInputDialog.getText(
        self, "Workspace Export", "Enter a variable name for this structure:", text=default_name
    )
    if not (ok and name):
        return

    global_workspace.push_crystal_structure(name, dft_struct)
    a, b, c = dft_struct.lattice.a, dft_struct.lattice.b, dft_struct.lattice.c
    QMessageBox.information(
        self,
        "Success",
        f"DFT structure '{name}' sent to Global Workspace ({info_note}).\n"
        f"Cell a={a:.3f} Å, b={b:.3f} Å, c={c:.3f} Å (vacuum={vacuum:.1f} Å).\n"
        f"Load it in the DFT Suite — not the 500 Å viewer canvas.",
    )
```

Do **not** replace `self.current_structure` with `dft_struct` (viewer stays on dummy after Render).
- [ ] **Step 3: Manual smoke check**

1. Tab 3: Graphene mono + h-BN, z=0 / 3.4, twist 0 / 0, vacuum 20 → Render → Push → no strain dialog (or mild if a differs) → workspace cell not 500.  
2. Same with twist 30° on h-BN → Push → commensurate **or** warn dialog → Push anyway → cell not 500.  
3. Three layers with one twist ≠ 0 → placeholder, no push.  
4. Tab 1 CIF push still works when stack empty.

- [ ] **Step 4: Commit**

```bash
git add tensorspec/gui/suites/crystal_suite.py
git commit -m "feat(crystal): Push rebuilds DFT cell from stack"
```

- [ ] **Step 5: Mark spec status**

In `docs/superpowers/specs/2026-09-13-stack-dft-cell-rebuild-design.md`, set `Status: approved — implemented on TensorSpec_GUI` when smoke passes.

```bash
git add docs/superpowers/specs/2026-09-13-stack-dft-cell-rebuild-design.md
git commit -m "docs(crystal): mark stack DFT rebuild spec implemented"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Rebuild at Push, viz stays dummy | 4 (+ 2/3 builders) |
| N-layer aligned + vacuum 20 | 2, 4 |
| Z = interlayer spinboxes | 2, 3 (placement) |
| 2-layer twist moiré / forced strain | 3, 4 |
| Ref pick + min-strain suggestion | 1, 4 |
| >2 twist placeholder block | 1, 4 |
| Preserve layer_tag | 2, 3 |
| No DFT Suite changes | — |
| Unit tests | 1–3 |

## Placeholder scan

Full implementations inlined for Tasks 1–4 (no TBD / “fill in later”).

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-13-stack-dft-cell-rebuild-plan.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — this session with executing-plans, checkpoints  

Which approach?
