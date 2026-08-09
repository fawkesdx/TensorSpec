# Suite Gaps Track C — PBR Shiny — Design Spec

Date: 2026-08-08  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: Track A/B suite-gaps specs; Crystal Suite viewer (`viewer_3d.js`)

## Problem

Crystal Suite has a disabled `#cr-pbr` checkbox labeled “PBR Shiny (HTML viewer: not yet)”. The three.js viewer already draws atoms and bonds with `MeshStandardMaterial` at matte-ish metalness/roughness. Users cannot toggle a shiny (3ds Max–style) look from the Styles panel.

## Goals

- Enable `#cr-pbr` and wire it to live shiny vs matte materials on **atoms and bonds only**.
- Checked → shiny defaults; unchecked → current matte defaults.
- No server/API changes.

## Non-goals

- Eraser brush, polyhedra connection mode, Matplotlib/PyVista backends (later Track C slices / deferred).
- Metalness/roughness sliders or per-element PBR overrides.
- Changing cell box, BZ, axes, cut-plane, or arrow materials.
- Texture maps / env maps / IBL.

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Track C order | PBR → Eraser → Polyhedra; Matplotlib/PyVista deferred |
| This spec | PBR only |
| Approach | A — checkbox toggles fixed shiny vs matte params |
| Surfaces | Atoms + bonds only |
| Delivery | Client-only (`viewer_3d.js` + `crystal_suite.js` + HTML enable) |

## Architecture

```
#cr-pbr change ──► crystal_suite.js
                      └─ viewer.setPbrShiny(checked)
                            ├─ store flag
                            ├─ _drawAtoms / _drawBonds use shiny|matte params
                            └─ if geometry already shown: refresh atom/bond materials
                               (patch MeshStandardMaterial or light setGeometry redraw)
```

---

## §1 — Viewer API

### Constants

| Mode | metalness | roughness |
|------|-----------|-----------|
| Matte (default, unchecked) | atoms `0.1`, bonds `0.1` | atoms `0.45`, bonds `0.5` |
| Shiny (checked) | `0.85` | `0.2` |

(Match today’s matte numbers already in `_drawAtoms` / `_drawBonds`.)

### `CrystalViewer3D` (or current viewer class)

- `setPbrShiny(enabled: boolean): void`
  - Sets internal `_pbrShiny` flag.
  - If content is loaded, updates atom InstancedMesh and bond InstancedMesh materials’ `metalness` / `roughness` and marks `needsUpdate`, **or** re-invokes the atom/bond draw path without clearing camera/controls.
- `_drawAtoms` / `_drawBonds` read `_pbrShiny` when creating materials so first Draw also respects the checkbox.

Do not alter BZ / cell / axes materials in this pass.

---

## §2 — UI wire

### HTML (`crystal_suite.html`)

- Enable `#cr-pbr` (remove `disabled`).
- Label: `PBR Shiny` (drop “HTML viewer: not yet”).
- Optional one-line hint under Styles: shiny applies to atoms and bonds.

### JS (`crystal_suite.js`)

- On init / after viewer ready: `dom.pbr` (or `el("cr-pbr")`) enabled; sync `viewer.setPbrShiny(dom.pbr.checked)` (default unchecked = matte).
- `change` listener → `viewer.setPbrShiny(dom.pbr.checked)`.

No geometry POST; no workspace persistence of the flag this pass (session UI only).

---

## §3 — Tests + success

### Tests

- Prefer a small pure helper if extracted, e.g. `pbrMaterialParams(shiny: boolean) -> {metalness, roughness}` for atoms/bonds, unit-tested in Node or Python-free JS if harness exists; otherwise **manual** verify + contract that `_drawAtoms` uses helper.
- No live browser CI required this pass if none exists.

### Success criteria

1. Unchecked: atoms/bonds look like today (matte).
2. Checked: atoms/bonds visibly shinier; cell/BZ unchanged.
3. Toggle without full page reload while a structure is drawn.
4. Checkbox enabled; no “not yet” copy.

## Out of scope (later Track C)

- Interactive eraser brush  
- Polyhedra (planes) connection mode  
- Matplotlib / PyVista graphics backends  
