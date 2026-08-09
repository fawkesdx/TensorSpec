# Suite Gaps Track C — Interactive Eraser — Design Spec

Date: 2026-08-08  
Status: approved for planning  
Branch: `HTML_einstein_app`  
Related: Track C PBR spec; Crystal Suite `viewer_3d.js` / export APIs

## Problem

`#cr-eraser` is disabled (“HTML viewer: not yet”). Qt Crystal Suite had continuous eraser brush + camera lock. Users need to remove atoms in the HTML viewer and have **3ds / Blender / CIF / workspace push** see only remaining atoms, without permanently mutating the source workspace crystal until an explicit push of a filtered structure.

## Goals

- Enable interactive eraser: camera lock + drag/click raycast delete of atoms.
- Session erase set (viewer indices); redraw without erased atoms/bonds.
- Pass omit indices into scene export, CIF download, and push so consumers see filtered geometry.
- Clear erase set when drawn geometry is rebuilt (Draw / supercell / basis / CDW / stack refresh).

## Non-goals

- Soft brush radius in Å; undo stack beyond “clear erase / re-Draw”.
- Bond-only erase (bonds drop when an endpoint is erased).
- Polyhedra; Matplotlib/PyVista.
- In-place permanent rewrite of the existing workspace node without push.
- Guaranteed index stability across unrelated geometry rebuilds (erase clears instead).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Persistence | **1+export** — session filter; exports/push use remaining atoms |
| Interaction | **A** — eraser on → OrbitControls off; continuous brush |
| Index space | Atom indices in the **current** `geometry` payload from `/geometry` |
| Clear erase | On geometry refresh; optional Reset control |
| Workspace | Unchanged until Push writes filtered structure |

## Architecture

```
#cr-eraser on ──► controls.enabled=false
pointer drag ──► raycast InstancedMesh atom ──► erased.add(index) ──► render filtered

geometry refresh ──► erased.clear()

export 3ds/blender ──► SceneExportRequest.omit_atom_indices ──► filter geo → SceneExporter
CIF / push        ──► same omit list ──► filter Structure or geo → CIF / workspace
```

---

## §1 — Viewer eraser

### State

- `CrystalViewer.eraserEnabled: boolean` (default false)
- `CrystalViewer._erasedAtomIndices: Set<number>`
- `setEraserEnabled(on: boolean)` — toggles flag; sets `this.controls.enabled = !on`
- `getOmittedAtomIndices(): number[]` — sorted copy of set
- `clearErasedAtoms(): void` — empty set; re-render if geometry present (`frame: false`)
- `omitAtom(index: number): void` — add index; re-render filtered

### Draw path

- Before `_drawAtoms` / `_drawBonds`, build filtered geometry view:
  - atoms = those whose index ∉ erased
  - bonds = those with both `i`,`j` surviving; reindex bond endpoints to new compact atom list **or** keep original indices only in the erase set and skip drawing erased instances while keeping mesh index maps consistent.
- **Preferred implementation:** keep full `geometry` as source of truth; when drawing, skip erased atom instances and bonds touching erased atoms; raycast map still uses original indices. Export uses original indices in `omit_atom_indices`.
- Hover tooltip: skip erased atoms (or do not hit them if not drawn).

### Pointer

- When `eraserEnabled`:
  - `pointerdown` / `pointermove` (while buttons pressed): raycast → `omitAtom(index)`
  - Prevent OrbitControls from capturing (already disabled)
- When off: existing tooltip `pointermove` only.

---

## §2 — Suite UI

### HTML

- Enable `#cr-eraser`; label `Enable Interactive Eraser Brush` (no “not yet”).
- Hint: camera locks while eraser on; erase applies to 3ds/Blender/CIF/push; re-Draw clears erase.
- Optional button `#cr-eraser-reset` “Reset erase”.

### JS

- `dom.eraser` (+ reset); change → `viewer.setEraserEnabled(checked)`
- On every successful `refreshGeometry` / stack render / etc. that replaces geometry: `viewer.clearErasedAtoms()` (or clear before `render`)
- `sceneExportPayload()` adds `omit_atom_indices: viewer.getOmittedAtomIndices()`
- CIF / push: send the same list (see §3)

---

## §3 — API

### Schema

```python
# SceneExportRequest
omit_atom_indices: list[int] = []
```

Validate: all ≥ 0; duplicates ignored; out-of-range indices ignored or 422 — prefer **ignore OOR** with log-friendly YAGNI (ignore).

### Scene export

After `_geometry_from_structure` (and before `_scene_export_parts`):

1. Filter `geo.atoms` / `geo.bonds` by omitting indices.
2. Remap bond `i`,`j` to new compact indices for SceneExporter.
3. Proceed as today.

### CIF

- Add `POST /api/crystal/{name}/cif` with body `{ omit_atom_indices, basis?, nx/ny/nz? }` matching export knobs needed for consistency **or** minimal: omit indices against **conventional unit-cell structure without supercell** only if CIF is always unit-cell today.
- **Today:** CIF is GET of workspace structure (no supercell). For v1 eraser CIF:  
  - If erase indices refer to **drawn supercell geometry**, CIF must either (a) refuse when supercell≠1×1×1 and erase non-empty, or (b) build the same supercell as the viewer then remove atoms then write CIF.
- **Locked for v1:** CIF/push filtered export uses the **same nx,ny,nz,basis,bond knobs as the viewer Draw** (mirror `sceneExportPayload` / geometry request), build supercell, omit indices, then CIF or push that Structure.

### Push

- Extend push request with `omit_atom_indices` + cell/basis knobs **or** new endpoint `POST …/push_filtered`. Prefer extending existing push body if a Pydantic model already exists; else add fields with defaults `[]`.
- Result: new/updated workspace structure = filtered supercell (document in hint).

### GET CIF

- Keep unfiltered GET for “full crystal file” if still linked; suite “Export CIF” switches to POST filtered when erase non-empty, else GET OK.

---

## §4 — Tests + success

### Tests

- Filter helper: given atoms/bonds + omit set → remaining atoms/bonds with remapped indices.
- Schema accepts `omit_atom_indices`.
- Export path unit test: omit one atom → SceneExporter inputs lack it (mock writer or inspect filtered geo).
- No live browser CI required.

### Success criteria

1. Eraser on: orbit locked; brush removes atoms; bonds drop with endpoints.
2. 3ds/Blender/CIF/push omit erased atoms when indices sent.
3. Re-Draw clears erase; full structure returns.
4. Eraser off: orbit works; tooltip unchanged.

## Out of scope (later)

- Polyhedra connection mode  
- Undo stack / soft radius  
- Matplotlib backends  
