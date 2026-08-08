# QE Slab Mode — Design Spec

Date: 2026-08-07  
Status: approved

## Problem

Roadmap: *Slab creation for surface slab calculations (drawn from the crystal suite).*  
Today QE Generate always writes bulk-style inputs (`kz` from UI, no `assume_isolated`). Tab 3 stacks already have vacuum; Tab 1 / bulk CIF need a cut + vacuum before DFT.

## Two paths

1. **Tab 3 stack / heterostructure** — already multilayer + vacuum. DFT Suite selects it and enables **Slab QE** (no re-cut).
2. **Tab 1 / bulk CIF** — DFT Suite **Prepare slab**: presets (thin/medium/thick × common faces) **or** custom hkl / N layers / vacuum Å → store `{name}_slab` → Generate with Slab QE.

## Decisions

- UI lives in **DFT Suite** (Prepare slab + Slab QE checkbox).
- Geometry reuses `CrystalEngine.extract_monolayer_miller`.
- Slab QE: force `kz = 1`, inject `assume_isolated = '2D'` in scf/nscf.
- Auto-suggest Slab QE when `lattice.c > 12` Å (stacks and prepared slabs).

## Presets

| Preset | hkl | layers | vacuum (Å) |
|--------|-----|--------|------------|
| thin_001 | 0 0 1 | 1 | 15 |
| medium_001 | 0 0 1 | 3 | 15 |
| thick_001 | 0 0 1 | 5 | 20 |
| thin_111 | 1 1 1 | 1 | 15 |
| medium_111 | 1 1 1 | 3 | 15 |
| thick_111 | 1 1 1 | 5 | 20 |
| thin_110 | 1 1 0 | 1 | 15 |
| medium_110 | 1 1 0 | 3 | 15 |
| custom | user hkl | user | user |

## Not in this release (planned later)

These are **not available yet**. They stay on the roadmap for a **future iteration** of TensorSpec — use this list as a reminder of what the slab/QE path still lacks:

- **Dipole corrections / asymmetric polar slabs** — not in Generate today; later for polar surfaces.
- **Termination labeling for surface bands** — cannot mark which face/termination owns a band yet; roadmap item still open.
- **Semi-infinite Green’s function** setups — not implemented; separate future DFT surface tool.
- **SCF-only Generate** (skip Wannier) — Generate still always writes the full SCF→NSCF→Wannier set; optional slim pipeline later.

See also `roadmap.md` under DFT → Full DFT Capability for the same open checkboxes.
