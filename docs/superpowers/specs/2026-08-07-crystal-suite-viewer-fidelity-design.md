# Crystal Suite Viewer Fidelity — Design Spec

Date: 2026-08-07  
Status: approved for implementation  
Branch: `HTML_einstein_app` (local clone → push → Einstein pull)  
Related plan: (to be written after this spec is reviewed)

## Problem

Crystal Suite Tab 1 “Dynamic Element Colors” still shows static Ta/Ir/Te swatches. Atom colors come from a tiny hardcoded CPK table (else teal). CIF `_atom_site_label` never reaches the browser. Many Tab 1 controls exist in HTML but are unwired. Opening a non-TaIrTe CIF looks like naming and colors ignore the file.

## Goals

- Element swatches rebuild from loaded `geometry.elements`; live color overrides recolor the three.js scene.
- Full Jmol/CPK palette for common elements; unknown symbols still get a stable fallback.
- Each `Atom` carries CIF/pymatgen `label`; hover shows `label (element)`.
- Practical three.js wiring (scope B): bond thick, axes, conventional cell toggle, Bonds|None connections, perspective/ortho, +a/+b/+c/111, azimuth/elevation, conventional|primitive basis.
- Load accepts `.cif`, `.vasp`, `.poscar` with correct pymatgen `fmt`.
- Disabled stubs (Matplotlib/PyVista, polyhedra, PBR, eraser, cut plane, 3ds/Blender, hi-res) stay disabled with a clear “HTML viewer: not yet” hint.
- Pytest coverage for labels + basis transform; Einstein pull/restart instructions after push.

## Non-goals

- Matplotlib / PyVista backends
- Polyhedra, PBR materials, interactive eraser, cut-plane tools
- Dual conventional + primitive boxes drawn at once
- Export 3ds Max / Blender / high-res PNG
- Persisting color overrides across page reload or sessions
- Full desktop Crystal Suite parity

## Constraints

- Colors remain a **browser display choice** (existing `CrystalGeometry` contract). Server does not ship hex colors.
- Work in local `/Users/sandyai/Documents/GitHub/TensorSpec` on `HTML_einstein_app`; push to origin; user pulls on Einstein (path C).
- Prefer existing patterns: `viewer_3d.js` draws only; geometry from `/api/crystal/{name}/geometry`.

## Design

### Data flow

```
CIF/VASP/POSCAR upload
  → parse (fmt by extension)
  → workspace Structure (pymatgen)
  → POST /geometry {nx, ny, nz, bond_threshold, show_bonds, basis, CDW…}
  → optional conventional|primitive transform
  → Atom{element, label, position, radius}
  → CrystalViewer
       color = override[element] || CPK[element] || fallback
       bondRadius from #cr-bondthick
       axes / cell / bonds toggles
       camera quick-views + az/el
  → swatch DOM from geometry.elements (+ Bonds row)
```

### Server

**`Atom` schema** — add `label: str`. Populate as:

```python
label = getattr(site, "label", None) or site.specie.symbol
```

For disordered sites, use the dominant specie symbol when `site.specie` is unavailable; if a site cannot be represented, return 422 naming the site index.

**`GeometryRequest`** — add `basis: Literal["conventional", "primitive"] = "conventional"`. Apply via pymatgen `SpacegroupAnalyzer` before supercell/CDW. If primitive conversion fails, return 422 with a clear message; UI keeps conventional selected and shows status.

**Load endpoint** — accept `.cif` / `.vasp` / `.poscar` (case-insensitive). Choose `fmt` from extension (`cif`, `poscar` for vasp/poscar). Fix error text so it does not claim “Expected a .cif file” when other formats are allowed.

### Viewer (`viewer_3d.js`)

- Expand `CPK_COLORS` to a full Jmol/CPK table; keep `FALLBACK_COLOR` for unknowns.
- `colorOverrides` map + `setElementColor(symbol, hex)` / `setBondColor(hex)`; exported `elementColor(symbol)` reads override then table.
- Wire existing `bondRadius` to `#cr-bondthick`.
- Optional `THREE.AxesHelper` (or lattice-aligned a/b/c arrows) toggled by “Show Axes”.
- `showCell` from “Show Conv. Box”; when `basis=primitive`, the drawn cell is the primitive cell (no second prim box this pass).
- Camera: Perspective ↔ Orthographic swap; `lookAlongLattice(+a/+b/+c/111)` using current `geometry.cell`; azimuth/elevation set spherical position about controls target.
- Hover: raycast InstancedMesh → floating tooltip `label (element)`.

### Suite UI (`crystal_suite.html` + `crystal_suite.js`)

- Replace static Ta/Ir/Te rows with an empty swatch container (`#cr-swatches`). Rebuild from `geometry.elements` plus a Bonds row after each geometry refresh. `input[type=color]` updates overrides and re-renders without refetch.
- Add missing `id`s on basis radios, Show Axes, Show Conv. Box, connection select, projection select, quick-view buttons.
- `#cr-conn`: **Bonds (Sticks)** and **None** work; **Polyhedra** stays disabled or shows status “HTML viewer: not yet”.
- File input `accept=".cif,.vasp,.poscar,.POSCAR"`.
- Graphics backend select, PBR, eraser, cut plane, 3ds/Blender, hi-res, Render Structure: remain disabled with hint text.

### Tests

- Pytest: geometry atoms include CIF/site labels when present.
- Pytest: `basis=primitive` yields a different cell (or volume) than conventional for a known structure when symmetry allows; failure path returns 422.
- Manual Einstein checklist: load non-TaIrTe CIF → swatches match elements; hover labels; bond thick / axes / +a; basis toggle.

### Deploy (Einstein)

After merge/push of this work on `HTML_einstein_app`:

```bash
cd ~/TensorSpec   # or actual Einstein clone path
git fetch
git checkout HTML_einstein_app
git pull
# restart the uvicorn process used on Einstein, e.g.:
# uvicorn tensorspec.web.server.app:app --reload --host 127.0.0.1 --port 8000
```

Exact restart command confirmed with the user if Einstein uses a service/supervisor rather than a bare uvicorn.

## Success criteria

1. Load MoS₂ (or any non-TaIrTe CIF) → swatches list those elements (not Ta/Ir/Te); distinct CPK colors.
2. Hover shows CIF site labels when present.
3. Bond thick / axes / cell / +a change the scene without reloading the file.
4. Conventional ↔ Primitive updates geometry when symmetry allows; clear error otherwise.
5. Disabled stubs show “HTML viewer: not yet” — no fake interactive behavior.

## Open decisions (resolved)

| Topic | Choice |
|-------|--------|
| Workspace | Local clone + push; Einstein pull (C) |
| Tab 1 depth | Practical three.js wiring (B) |
| CIF labels | On atoms + hover (A) |
| Architecture | Hybrid: server labels/basis; client colors/camera |
