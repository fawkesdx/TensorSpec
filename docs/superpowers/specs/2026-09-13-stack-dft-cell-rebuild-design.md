# Stack → DFT cell rebuild (Tab 3 Push)

**Date:** 2026-09-13  
**Status:** draft — awaiting user review  
**Branch:** `TensorSpec_GUI`

## Problem

Crystal Suite Tab 3 (`3. Stack & Twist`) builds heterostructures for visualization on a **500×500×500 Å dummy cubic lattice** (`CrystalEngine.build_heterostructure_stack`). Pushing that Structure to the workspace and loading it in the DFT Suite writes the dummy box verbatim into QE `CELL_PARAMETERS`. Plane-wave DFT then fails or hangs (FFT/MPI `nnr`, absurd cutoffs).

Exfoliation already builds real vacuum slabs and avoids pushing the dummy box. Stack Push does not.

## Goal

On **Push to Workspace**, materialize a **DFT-ready** pymatgen `Structure` with a physical in-plane cell + vacuum along **c**, so DFT/QE/bandstructure consumers get a usable cell. Keep Tab 3 **Render** on the dummy canvas for viewer performance.

Primary users this week: groups stacking 2D devices (aligned multilayers) and graphene/hBN ~30° twist (2 layers).

## Non-goals

- Changing Render to use the real DFT lattice (viewer stays on 500 Å canvas)
- Multi-layer **twisting** DFT cells (more than 2 layers with any θ ≠ 0) — placeholder only
- DFT Suite / QE generator changes (consumers load Structure as today)
- MPI launcher / `nnr` / Einstein–Mac QE parallel fixes
- Auto-relax / interface reconstruction / DFT strain energy

## Decisions (from brainstorm)

| Topic | Choice |
|-------|--------|
| When to rebuild | **At Push** (Approach 1) |
| Aligned stacks | **N layers**, all \|θ\| ≈ 0 |
| Twisted stacks | **Exactly 2 layers**; commensurate moiré preferred |
| Incommensurate | Allow with **warn popup** + user picks reference layer |
| Ref layer | User chooses; UI **suggests min-strain** option |
| Interlayer spacing | Existing per-layer **Z spinboxes** (`z_shift`) |
| Vacuum | New **Vacuum (Å)** spinbox, default **20** (padding along c, not interlayer) |
| >2 layers + twist | Block push + “coming later” placeholder |

## Architecture

Two Structure roles:

1. **Viz stack** — unchanged: Render → `build_heterostructure_stack` → dummy 500 Å  
2. **DFT stack** — new: built only on Push from layer-row dicts + vacuum

```
Tab 3 Render  →  viz Structure (dummy)  →  viewer only
Tab 3 Push    →  classify layers
              →  rebuild DFT Structure
              →  (optional strain dialog)
              →  global_workspace.push_crystal_structure(name, dft_struct)
DFT Suite     →  pull / load as today (no special case)
```

After Push, Crystal Suite may keep `current_structure` as the viz dummy so zoom/render stay stable. Workspace holds only the DFT Structure.

## Classification (Push)

Given stack layer rows:

1. No rows / nothing to push → existing warning  
2. Any \|θ\| > ε (ε ≈ 1e-6°) **and** layer count ≠ 2 → **block** with placeholder message (multi-twist later)  
3. All \|θ\| ≤ ε → **aligned N-layer** rebuild  
4. Exactly 2 layers with any twist → **twist** rebuild (moiré / forced-strain)

Push trigger rule (explicit):

- If `stack_layer_rows` is **non-empty** → always rebuild DFT Structure from those rows (ignore dummy `current_structure` lattice).
- Else → push `current_structure` unchanged (Tab 1 CIF / exfoliated mono / etc.).

## Cell construction

### Shared

- Place atoms using existing stack geometry rules (SC, twist about z, Z from spinboxes)  
- Preserve `layer_tag` site properties  
- `c = (z_max − z_min) + vacuum_Å`; center the slab in the vacuum along c  
- Reject / error on degenerate in-plane lattices  

### Aligned (N layers, all twists ~0)

- In-plane lattice from **reference** layer’s `a,b` (with that layer’s SC)  
- Other layers: if relative in-plane lattice mismatch vs ref exceeds **1%** (Frobenius on 2×2), open the same strain dialog as twist; if within 1%, map without dialog  
- Suggested reference: min total strain metric; if tied within 0.1%, default to layer 1  

### Twist (exactly 2 layers)

Reuse / extend `CrystalEngine.calculate_moire_superlattice`:

- **Commensurate:** in-plane cell = moiré matrix; tile both layers into that cell; apply relative twist; Z + vacuum as shared  
- **Incommensurate / forced strain:**  
  - Dialog before push  
  - User picks reference layer  
  - UI marks **suggested** choice = lowest strain  
  - Copy must explain: non-ref layer(s) will be stretched/compressed to the reference cell so DFT can run; bands/geometry are approximate  
  - Cancel → no push; confirm → build forced-strain Structure  

### Strain metric (suggestion)

For each candidate reference `r`, estimate isotropic (or Frobenius) in-plane mismatch of the other layer’s lattice (after relative twist for the 2-layer case) vs `r`. Suggest `argmin`. Show approximate % strain in the dialog text.

## UI (Tab 3)

- **Vacuum (Å)** `QDoubleSpinBox`, default 20, sensible range (e.g. 5–100), near Render/Push  
- Push button behavior: rebuild path when stack rows present  
- Strain / mismatch dialog: ref dropdown, suggested label, explanatory text, Cancel / Push anyway  
- Multi-twist block: short `QMessageBox` — not supported this week  
- Optional success note: pushed cell lengths `a,b,c` and vacuum used  

## Code touchpoints

- `tensorspec/core/crystallography.py` — pure rebuild helpers (aligned, twist commensurate, forced-strain, classify, strain suggestion); keep `build_heterostructure_stack` for viz  
- `tensorspec/gui/suites/crystal_suite.py` — vacuum spinbox; Push orchestration; dialogs  
- `tests/` — unit tests for rebuild math (no GUI)

## Error handling

| Case | Behavior |
|------|----------|
| Empty stack / no structure | Existing warn |
| >2 layers with twist | Placeholder; no push |
| Moiré / lattice math fail | Critical; no push |
| User cancels strain dialog | No push |
| Degenerate cell | Critical; no push |

## Testing

- Aligned bilayer: lattice not ~500 Å; `c ≈ thickness + vacuum`; atoms inside cell  
- Dummy detect / push path uses rebuild when stack rows exist  
- Commensurate twist fixture: finite physical cell  
- Incommensurate: strain % + suggested ref index  
- Classifier rejects >2 twisted layers  

## Success criteria

1. Aligned multilayer stack → Push → DFT load → QE `CELL_PARAMETERS` is physical (not 500³)  
2. Graphene/hBN ~30° (2 layers) → Push works (commensurate cell or forced-strain after warn)  
3. Render viewer still uses dummy canvas  
4. >2 twisted layers cannot silently push a dummy cell  

## Future (placeholder only)

- N-layer twist / multi-moiré DFT cells  
- Optional: store both viz + DFT under workspace metadata  
- Optional: show DFT cell outline in viewer without replacing dummy Structure  
