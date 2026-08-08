# Crystal Suite Cut-Plane Guide — Design Spec

Date: 2026-08-08  
Status: approved for implementation  
Branch: `HTML_einstein_app` (local clone → push → Einstein pull)  
Related: extends Crystal Suite viewer fidelity (Tab 1 Crystallography Tools)

## Problem

Tab 1 “Crystallography Tools” shows **Show Cut Plane**, color/lock selects, and a **Depth** slider, plus View [h k l] / Align. Almost none of this is wired. The hint already says cut plane is “not yet.” Users need a **transparent Miller-plane guide** to visualize a cleaving / termination surface without removing atoms.

## Goals

- Draw a translucent plane oriented to user **[h k l]** in the three.js viewer.
- **Depth** slider moves the plane along the plane normal (perpendicular to the surface).
- Atoms and bonds stay fully drawn — plane is a visual guide only (no hard clip).
- Color select tints the plane.
- Wire [h k l] inputs with ids; disable or remove “Lock to Camera” for this pass (fixed: lock to [hkl]).
- Update the Tab 1 hint so cut plane is no longer listed as “not yet.”

## Non-goals

- **Align** camera to [hkl] (button stays disabled).
- Lock plane to camera (screen-facing).
- Hard-clip / hide atoms on one side of the plane.
- Polyhedra, hi-res export, 3ds/Blender, eraser, Matplotlib/PyVista.
- Server-side plane geometry (client-only display, like colors).

## Constraints

- Work on `HTML_einstein_app` in the local TensorSpec clone; push; Einstein pull (path C).
- Plane math uses `geometry.cell` lattice vectors already in the browser payload.
- Plane must survive or be reapplied after `CrystalViewer.render()` clears content.

## Design

### User model

1. Set **[h k l]** (cleave / termination Miller indices).  
2. Enable **Show Cut Plane** → translucent sheet appears.  
3. Drag **Depth** → sheet slides along the surface normal through the crystal.  
4. Atoms remain; plane shows where a cut would pass.

### Data flow

```
Tab 1: h,k,l + cut checkbox + color + depth
  → crystal_suite.js
  → CrystalViewer.setCutPlane({ h, k, l, depthFrac, color, visible })
  → translucent THREE.Mesh (DoubleSide)
```

### Miller normal (client)

Given cell vectors **a, b, c** from `geometry.cell`:

- **a\*** ∝ **b × c**, **b\*** ∝ **c × a**, **c\*** ∝ **a × b**
- **n** = normalize(h **a\*** + k **b\*** + l **c\***)
- Reject **(0,0,0)** with a status error; hide plane.

### Plane placement

- Viewer atoms are drawn relative to `geometry.center`. Plane uses the same frame.
- `depthFrac` from slider ∈ [−1, 1] (map UI −100…100 → /100).
- `halfExtent` = half the projection of the cell AABB onto **n**.
- Plane origin offset = **n** × (`depthFrac` × `halfExtent`).
- Plane size ≈ 1.2 × longest face diagonal of the cell (covers the cell at any depth).
- Material: transparent, opacity ≈ 0.25, `depthWrite: false`, `DoubleSide`, color from select (cyan/magenta/yellow/white/gray).

### Integration with render

Store cut-plane state on the viewer. After each `render()`, re-add or update the plane mesh so `clear()` does not permanently drop it. Toggling visibility / depth / color / hkl updates without refetching geometry.

### HTML / JS

| Control | Id | Behavior |
|---------|-----|----------|
| h, k, l | `cr-h`, `cr-k`, `cr-l` | Change → update plane orientation |
| Show Cut Plane | `cr-cut` | Toggle visibility |
| Color | `cr-cut-color` | Values = hex or named map |
| Depth | `cr-depth` (exists) | Input → update offset |
| Lock select | disable or remove | This pass always [hkl] |
| Align | stay `disabled` | Out of scope |

Hint text: remove “Cut plane” from the “not yet” list; keep Align / 3ds / Blender / hi-res.

### Errors

- No structure loaded → cut controls no-op; optional status “Load a structure first.”
- (0,0,0) → status error; plane hidden.

### Tests / verification

- Unit or pure-function test: cubic cell, (0,0,1) → normal along +c (within tolerance).
- Manual: load layered structure (e.g. MoS₂), [001], show plane, slide depth through layers; atoms remain.

### Deploy

Same as prior crystal work: push `HTML_einstein_app`, Einstein `git pull` + restart uvicorn.

## Success criteria

1. Show Cut Plane draws a translucent plane oriented to [hkl].  
2. Depth moves the plane along the normal; atoms stay visible.  
3. Color select updates plane tint.  
4. Changing hkl updates orientation.  
5. Align / camera-lock / atom clip remain out of this pass.

## Open decisions (resolved)

| Topic | Choice |
|-------|--------|
| Product intent | Transparent cleave guide; no atom removal |
| Scope | Wire hkl + show + color + depth; Align later |
| Math | Client Miller normal from `geometry.cell` |
| Lock modes | [hkl] only this pass |
