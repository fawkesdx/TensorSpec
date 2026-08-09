# Suite Gaps Track C — Coordination Polyhedra — Design Spec

Date: 2026-08-08  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: Track C PBR / Eraser specs; `CrystalEngine.compute_bonds`; scipy `ConvexHull`

## Problem

Crystal Suite Connections has a disabled `polyhedra` option. Qt drew coordination polyhedra (planes) instead of sticks. HTML still only supports bonds / none.

## Goals

- Enable `conn=polyhedra`: translucent **coordination hulls** around atoms with ≥3 bonded neighbors; **no sticks**.
- Server builds hulls with existing bond graph + `bond_threshold` and scipy `ConvexHull`.
- Client draws faces when mode is polyhedra; erase skips hulls touching erased atoms.
- Viewer-only for DCC: 3ds/Blender/CIF ignore polyhedra faces (atoms/bonds/cell only).

## Non-goals

- Exporting polyhedra faces to 3ds/Blender.
- Face-level delete (atom eraser covers).
- Client-side ConvexHull.
- Matplotlib/PyVista backends.

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Definition | Coordination hulls of bonded neighbors |
| Compute | Server (scipy), Approach A |
| Export | Viewer-only v1 |
| Sticks when polyhedra | Off |
| Degenerate hulls | Skip (catch `QhullError` / &lt;4 coplanar points) |

## Architecture

```
GeometryRequest.connections = polyhedra
  → bonds via compute_bonds
  → for each atom with ≥3 neighbors: ConvexHull(neighbor coords)
  → CrystalGeometry.polyhedra[]

Viewer: showPolyhedra → Mesh per hull (translucent)
Eraser omit → skip hull if center or any vertex index erased
```

---

## §1 — Schema / API

### Request

Prefer additive field (minimal client churn):

```python
# GeometryRequest
show_bonds: bool = True  # kept
show_polyhedra: bool = False  # new
```

Client when `conn=polyhedra`: `show_bonds=False`, `show_polyhedra=True`.  
When `bonds`: `show_bonds=True`, `show_polyhedra=False`.  
When `none`: both false.

### Response

```python
class Polyhedron(BaseModel):
    center: int  # atom index
    vertices: list[list[float]]  # neighbor positions (same frame as Atom.position)
    simplices: list[list[int]]  # triangles into vertices

class CrystalGeometry(...):
    polyhedra: list[Polyhedron] = []
```

### Compute helper

`compute_coordination_polyhedra(structure, bonds_i_j, coords) -> list[Polyhedron]`  
in `crystallography.py` or `geometry_filter`-adjacent module. Cap: skip if `len(structure) > MAX_RENDER_ATOMS` already enforced upstream.

---

## §2 — Viewer

- `render(geometry, { showBonds, showPolyhedra, showCell, frame })`
- `_drawPolyhedra(geometry, center)`: for each polyhedron, if any of `center` or mapped vertex atom indices erased → skip; else BufferGeometry + MeshStandardMaterial (opacity ~0.3, doubleSide, depthWrite false); color from center element via `elementColor`.
- Need vertex→atom index map: store `vertex_atom_indices: list[int]` on Polyhedron (neighbor atom indices aligned with `vertices`).

**Locked:** each Polyhedron includes `vertex_atom_indices: list[int]` same length as `vertices`.

---

## §3 — Suite UI

- Enable polyhedra option; label `Polyhedra (Planes)`.
- Hint: coordination hulls; not exported to 3ds/Blender.
- `refreshGeometry` sends `show_polyhedra` / `show_bonds` from `#cr-conn`.
- Pass `showPolyhedra` into `viewer.render`.

---

## §4 — Tests + success

- Unit: diamond/Si or simple cube — known neighbor counts → ≥1 polyhedron or empty for sparse bonds.
- Degenerate: 3 neighbors → skip (need ≥4 non-coplanar for 3D hull; **if only 3 neighbors, skip** per scipy 3D ConvexHull needing 4 points). Spec: require **≥4** neighbors for a hull.
- Schema default `polyhedra=[]`.

### Success

1. Polyhedra mode → translucent hulls, no sticks.  
2. Bonds/None unchanged.  
3. Eraser hides hulls involving erased atoms.  
4. Export unchanged (no faces).

## Out of scope

- Polyhedra in SceneExporter  
- Soft face delete  
- Matplotlib backends  
