# Suite Gaps Track A — Design Spec

Date: 2026-08-08  
Status: approved for planning  
Branch: `HTML_einstein_app` (local → push → Einstein pull; no merge to `main`)  
Related: batch 1–4 plan `docs/superpowers/plans/2026-08-08-suite-gaps-batch-1-4.md`

## Problem

After batch 1–4, several high-value “wire now” gaps remain: W90 band overlay is stubbed, Align [hkl] is disabled, crystal export checkboxes/3ds/Blender/PNG are dead, and `BZRequest.overlay_crystal` is unused server-side while the client already owns overlay.

Larger tracks (QE XC, ARPES slit physics, 2D isoenergy, polyhedra/eraser, shell suites ML/XAS/PEEM/Transport, ARPES history) are **out of scope** for this spec.

## Goals

- Dual-solve DFT bands: primary (SK or W90) solid + optional W90 overlay dashed red on the same k-path.
- Align camera along Miller [h k l] using existing cut-plane Miller inputs.
- Crystal Export Elements: Atoms/Bonds, Unit Cell, Brillouin Zone checkboxes drive 3ds Max + Blender downloads via `SceneExporter`.
- Client hi-res PNG capture; “Render Structure” frames the three.js scene.
- Remove unused `overlay_crystal` from the BZ API; keep `#bz-overlay` client behavior.

## Non-goals

- QE XC picker, ARPES `slit_size` / deflector physics, DFT 2D k-grid / isoenergy.
- Polyhedra, eraser, PBR, Matplotlib/PyVista backends.
- Shell suites (ML / XAS / PEEM / Transport).
- ARPES sim-history browse UI.
- Fat-band weights on W90 overlay; caching overlay as a separate workspace band node.
- Native Wannier90 executable bands (uploaded `hr.dat` only).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Scope | Track A only; B/C/D/E later |
| Export | Checkboxes + 3ds + Blender + hi-res PNG |
| W90 overlay | Dual solve in one Calculate |
| `overlay_crystal` | Drop from API; client keeps overlay |
| Delivery | Thin vertical slices (Align → schema → W90 → export → PNG → deploy) |

## Architecture

Hybrid: server owns dual band solve and DCC script generation; browser owns Miller camera, BZ overlay toggle, and PNG capture.

```
DFT Calculate ──► POST /bands (overlay_wannier?)
                 ├─ primary solve → BandResult.bands (+ workspace)
                 └─ optional W90 solve → BandResult.overlay_bands
                 └─ BandPlot draws solid + dashed red

Crystal Align ──► viewer.lookAlongMiller(h,k,l)  (client only)

Crystal Export ─► POST /crystal/{name}/export/{3dsmax|blender}
                 └─ SceneExporter script download

Crystal PNG ───► WebGL canvas capture (client only)
BZ cleanup ────► remove BZRequest.overlay_crystal; JS stop sending
```

## §1 — W90 dual-solve + plot

### API

- `BandRequest.overlay_wannier: bool = False`.
- When true: require uploaded `wannier90_hr.dat` for the crystal; run two solves on the same path and TB parameters:
  1. Primary: honor `use_wannier` (SK or W90).
  2. Overlay: always W90 from uploaded hr.
- `BandResult.overlay_bands: list[list[float]] | None` — same `k_dist` / node grid as primary. Primary remains the workspace band node; overlay is response-only this pass.
- Missing hr when overlay requested → HTTP 422.
- Diagonalisation budget must account for both solves (reject with 422 if 2× too heavy).

### UI

- Enable `#tb-w90-overlay` after successful hr upload (same gate as Use-W90).
- Calculate sends `overlay_wannier` from the checkbox.
- Hint text: SK solid + W90 dashed red (dual solve).

### Plot

- `BandPlot`: if `overlay_bands` present, stroke dashed red; no fat/weights on overlay this pass.

## §2 — Align, export, PNG, BZ schema

### Align [hkl]

- Enable `#cr-align`; read `#cr-h`, `#cr-k`, `#cr-l`.
- `CrystalViewer3D.lookAlongMiller(h,k,l)`: direction = `millerNormal(cell,h,k,l)`; same distance/target pattern as `lookAlong`.
- `(0,0,0)` → client status error; no server call.

### Export Elements

- IDs: `#cr-exp-atoms`, `#cr-exp-cell`, `#cr-exp-bz`, `#cr-export-3ds`, `#cr-export-blender`.
- `POST /api/crystal/{name}/export/{fmt}` with `fmt ∈ {3dsmax, blender}`.
- Body includes geometry knobs (nx/ny/nz, bond threshold, basis, connectivity) plus `include_atoms`, `include_cell`, `include_bz`, and BZ params (scale/style/hkl) when BZ included.
- Server builds `SceneExporter` tuples from the same geometry path as the viewer; optional BZ solid from existing BZ builder.
- Response: downloadable script (`.ms` / `.py`).
- No crystal, or all includes false → 422.

### Render / Hi-res PNG

- `#cr-render` → `refreshGeometry({ frame: true })`.
- `#cr-hires` → client canvas capture at ≥2× (preserveDrawingBuffer / offscreen scale) → download `structure.png`.
- Update Tab 1 hint; remove “not yet” for Align / 3ds / Blender / hi-res.

### BZ `overlay_crystal`

- Remove field from `BZRequest`.
- Stop sending from `crystal_suite.js`.
- Keep `#bz-overlay` client clear-vs-keep logic unchanged.

## §3 — Errors, tests, ship order

### Errors

| Case | Behavior |
|------|----------|
| Overlay without hr.dat | 422 |
| Dual-solve over budget | 422 |
| Align (0,0,0) | Client status |
| Export empty / no crystal | 422 |
| Exporter exception | 422 with message |

### Tests

- `BandRequest.overlay_wannier` / `BandResult.overlay_bands` schema.
- Export: tiny structure → 3dsmax/blender text includes expected markers; BZ optional.
- `BZRequest` no longer accepts/requires `overlay_crystal`.
- Miller camera helper unit test if extracted; otherwise covered by viewer math reuse.

### Ship order

1. Align [hkl]
2. Drop `overlay_crystal`
3. W90 dual-solve + plot
4. Scene export API + checkboxes + 3ds/Blender
5. Render + hi-res PNG
6. Push `HTML_einstein_app` + Einstein pull/restart

### Success criteria

1. Overlay on → red dashed W90 on same plot as primary SK (or primary W90).
2. Align → camera faces [hkl].
3. Export downloads scripts that respect Atoms/Cell/BZ checkboxes.
4. Hi-res PNG downloads; Render frames the scene.
5. BZ overlay still works via `#bz-overlay`; API field gone.

## Out of scope (later tracks)

- **B:** QE XC, ARPES slit_size/defl, DFT 2D k-grid/isoenergy  
- **C:** Polyhedra, eraser, PBR, Matplotlib/PyVista  
- **D:** ML / XAS / PEEM / Transport shells  
- **E:** ARPES history browse  
