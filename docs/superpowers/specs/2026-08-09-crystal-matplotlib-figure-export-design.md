# Crystal Suite — Matplotlib Figure Export — Design Spec

Date: 2026-08-09  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: Crystal Suite geometry/export (`tensorspec/web/server/routers/crystal.py`), ARPES figure export (`tensorspec/plotting/backends/arpes_figure.py`), stub `#cr-backend`

## Problem

HTML Crystal Suite interactive viewing is three.js. Desktop Qt used Matplotlib/PyVista as alternate graphics backends. Those engines were removed in the HTML migration. The UI still shows a disabled `#cr-backend` with “Matplotlib/PyVista: HTML viewer: not yet.” Users want publication-quality figures without restoring Qt or a full alternate interactive backend.

## Goals

- Export a Crystal structure figure via **headless Matplotlib** (PNG / SVG / PDF).
- Match current Crystal **styles** where feasible: basis, supercell, bonds/none, radius, erase omit, polyhedra best-effort.
- Optional **use current three.js view** (camera position/target/up); otherwise a fixed default view.
- Keep three.js as the only interactive viewer.

## Non-goals

- PyVista / VTK (later).
- Interactive Matplotlib or replacing three.js.
- PBR materials in the figure (flat colors only).
- Using `#cr-backend` as a live backend switch.

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Product | Export / snapshot only |
| Backend v1 | Matplotlib only |
| UX | Dedicated Export figure button (not backend dropdown) |
| Styles | Match Draw/export path; ignore PBR; polyhedra best-effort or skip if hard |
| Camera | Default isometric-ish + optional “use current view” |
| Architecture | `crystal_figure.py` + `POST …/export/figure` (mirror ARPES) |

## Architecture

```
Crystal Suite UI
  styles + optional camera
        │ POST /api/crystal/{name}/export/figure
        ▼
crystal router → structure path (basis/supercell/omit)
        │
        ▼
plotting/backends/crystal_figure.py  (matplotlib Agg)
        │
        ▼
PNG | SVG | PDF download
```

three.js canvas unchanged.

---

## §1 — UI

- Add **Export figure (Matplotlib)** control group near 3ds/Blender exports:
  - Button `#cr-export-figure`
  - Format select: `png` | `svg` | `pdf` (default `png`)
  - Checkbox `#cr-export-figure-view`: “Use current view”
- On click: POST with Draw-equivalent style params; if checkbox on, include camera from the three.js controls.
- Relabel `#cr-backend` hint: interactive viewer is three.js; publication figures use Export figure. Keep select disabled or single option `three.js` (no Matplotlib option that implies swapping the live canvas).

---

## §2 — API

`POST /api/crystal/{name}/export/figure`

Request fields (align names with existing `GeometryRequest` / scene export where practical):

- Structure transform: `basis`, `nx`, `ny`, `nz` (or `cell_count` pattern already used)
- Style: `show_bonds`, `bond_threshold`, `atom_radius_scale` (or existing radius field name from geometry API), `omit_atom_indices`, `show_polyhedra`
- Export: `fmt: Literal["png","svg","pdf"]`, optional `title`
- Camera: `use_current_view: bool`, optional `camera: { position: [x,y,z], target: [x,y,z], up: [x,y,z] }`

Behavior:

1. Resolve structure from session; apply same basis / supercell / omit path as CIF/3ds export.
2. Cap atom count consistently with geometry render limits (422 if over).
3. Call `export_crystal_figure(...)` → bytes.
4. Return `Response` with correct `media_type` and `Content-Disposition` filename `{name}_figure.{fmt}`.

Errors: missing crystal → 404; too many atoms / bad camera → 422.

---

## §3 — `crystal_figure.py`

- `matplotlib.use("Agg")` only; no Qt / GUI.
- Draw:
  - Atoms as colored spheres/circles (CPK / existing color map)
  - Bonds as line segments when `show_bonds`
  - Unit-cell edges
  - Polyhedra: include if inexpensive (e.g. wireframe faces from server hulls); otherwise omit in v1 and document
- Lighting: simple / flat (no PBR).
- Camera: default look-at cell center from an isometric offset; if `use_current_view` and camera provided, set Matplotlib 3D view from position/target/up (best-effort elevation/azimuth or `view_init` / `set_position` as feasible with mplot3d).
- Return `bytes` for the chosen format (dpi sensible for PNG, e.g. 200–300).

---

## §4 — Tests / ship

### Tests

- Unit: 1–2 atom cell → PNG bytes non-empty; `omit_atom_indices` reduces drawn atoms / changes output.
- Optional: SVG/PDF magic headers.
- No browser CI.

### Success criteria

1. Crystal Suite → Export figure → openable PNG/SVG/PDF reflecting bonds/omit/supercell.  
2. “Use current view” changes viewpoint vs default.  
3. three.js Draw/orbit unchanged.  
4. `#cr-backend` no longer promises unimplemented live Matplotlib/PyVista.

## Out of scope (later)

- PyVista raster / glTF export  
- True interactive alternate backend  
- Hi-res desktop-parity raytrace
