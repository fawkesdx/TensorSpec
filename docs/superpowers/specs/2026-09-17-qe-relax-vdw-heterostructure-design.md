# QE ionic relaxation, vdW, and heterostructure geometry follow-through

**Date:** 2026-09-17  
**Status:** approved — ready for implementation  

**Branch:** `TensorSpec_GUI`

## Problem

TensorSpec’s QE path today is **SCF → NSCF → Wannier** on the structure as pushed. For layered heterostructures (e.g. graphene/hBN):

1. Crystal Tab 3 Push may **force-strain** one layer onto a reference cell (no ionic relax).  
2. Plain PBE without **vdW** gives wrong interlayer distances.  
3. Band / isoenergy results look unphysical (gapped/shifted Dirac, warped FS) partly because geometry was never relaxed.  
4. Related gaps users hit in the same workflow: no **selective dynamics**, no **write-back** of relaxed CIF to Crystal/workspace, and Push **strain / moiré** behavior still too blunt for twisted stacks.

## Goal

Add a first-class **geometry relaxation** option to the DFT Suite QE generator, with **2D-safe defaults**, optional **DFT-D3**, and a pipeline that always runs electronic steps on the **relaxed** structure. In the same project, deliver the follow-through items that make heterostructure DFT usable end-to-end: selective dynamics, relaxed-structure write-back, and improved Push strain / moiré handling.

Primary users: 2D stack → Push → Einstein/local QE → bands / ARPES (graphene/hBN and similar).

## Decisions (from brainstorm)

| Topic | Choice |
|-------|--------|
| Geometry modes | **`SCF only`** / **`Relax ions (fixed cell)`** (default) / **`vc-relax` (advanced)** |
| vdW | Optional checkbox; **DFT-D3** when on; warn if relax on + vdW off |
| Pipeline order | **relax → (update geometry) → scf → nscf → wannier** |
| Cell during ionic relax | **Fixed** `CELL_PARAMETERS` (keeps Push strain + vacuum) |
| vc-relax | Available but labeled advanced; document vacuum risk; constraints can come in P1.5/P2 |
| Selective dynamics | In scope — e.g. fix ref/bottom layer `if_pos` |
| Write-back | In scope — relaxed CIF (+ optional workspace update) |
| Auto moiré / Push strain | In scope — prefer commensurate moiré; clearer strain dialog |
| Build priority | **P1** relax+vdW+pipeline+CIF write-back → **P2** selective dynamics → **P3** Push/moiré |

## Non-goals (still out)

- Full interface reconstruction / MLFF / AIMD  
- Automatic experimental lattice matching beyond existing ref suggestion  
- Changing Tab 3 **Render** (viz stays on 500 Å dummy)  
- Multi-twist DFT (>2 layers with θ ≠ 0) — still blocked as today  
- Guaranteeing every QE build has D3 (detect + clear error/fallback message)  
- Replacing Wannier projection quality work (π-only windows, denser k) — separate from geometry  

## Architecture

```
QE Generator UI
  ├─ Geometry mode: scf_only | relax_ions | vc_relax
  ├─ vdW: off | dft-d3
  └─ Selective dynamics (P2): none | fix_ref_layer | fix_bottom_layer | custom later

Generate inputs
  ├─ relax.in     (if relax_ions or vc_relax)
  ├─ scf.in       (placeholder geometry; overwritten after relax in pipeline OR generated post-relax)
  ├─ nscf.in / wannier90.win / pw2wan.in
  └─ run_pipeline.sh

run_pipeline.sh
  1. pw.x < relax.in          # if enabled
  2. extract final positions (/ cell) → rewrite scf.in (+ nscf cell/atoms)
  3. pw.x < scf.in
  4. pw.x < nscf.in
  5. wannier90.x -pp ; pw2wannier90 ; wannier90.x

Post-run / Fetch
  └─ write relaxed_structure.cif ; optional push to workspace / Crystal
```

Crystal Tab 3 Push (P3) remains the source of the **initial** DFT cell; relax improves ions (and optionally cell) before bands.

## P1 — Relax + vdW + pipeline + CIF write-back

### UI (`qe_generator_panel.py`)

- Combo **Geometry**: `SCF only` | `Relax ions (fixed cell)` | `vc-relax (advanced)`  
  Default: **Relax ions (fixed cell)**.  
- Checkbox **vdW (DFT-D3)** default off; if Geometry ≠ SCF only and vdW unchecked → non-blocking warning on Generate.  
- Existing ecut / k-mesh / SOC / MPI unchanged; relax uses same ecut and a (possibly coarser) k-mesh spinbox or same mesh (prefer **same mesh** for P1 simplicity).

### Inputs (`qe_generator.py`)

**`relax.in`** (ionic):

- `calculation = 'relax'`  
- Same `&SYSTEM` as SCF (ecut, occupations, optional SOC, optional `vdw_corr = 'dft-d3'`)  
- **No** cell degrees of freedom (do not set `cell_dofree` / do not use vc-relax)  
- `&IONS`: BFGS defaults; `forc_conv_thr` e.g. `1.0d-3` (document; optional UI later)  
- `wf_collect` as needed for continuity; prefer restart-friendly `outdir`/`prefix` shared with SCF  

**`relax.in`** (vc-relax advanced):

- `calculation = 'vc-relax'`  
- Document risk for 2D vacuum; P1 may ship without `cell_dofree` restrictions; P2 can add `cell_dofree = '2Dxy'` or similar if QE version supports it  

**SCF/NSCF/Wannier:**

- After relax, atomic positions (and cell if vc-relax) must match the relaxed geometry.  
- Implementation options (pick one in plan; prefer A):  
  - **A.** Pipeline script parses `relax.out` / XML and regenerates `scf.in`/`nscf.in` atom blocks before SCF.  
  - **B.** Single `pw.x` chain with `calculation='scf'` restart from relax charge + manual position sync.  
- Wannier `num_wann` / projections unchanged in P1.

### Pipeline script

- If geometry = SCF only: current behavior.  
- Else: run relax first; on failure abort (do not SCF on unrelaxed junk without log).  
- Echo clear stage banners: `=== RELAX ===`, `=== SCF ===`, …

### Write-back

- Always write `relaxed_structure.cif` (and optionally `.vasp`) next to QE outputs when relax ran.  
- Fetch Remote / local finish: offer or auto **Push to workspace** as `*_relaxed` (name includes source).  
- Crystal Suite: loadable via existing **Load CIF** (no mandatory live sync in P1); optional “Apply relaxed CIF to workspace” button acceptable.

### vdW availability

- On Generate or first remote run: if D3 requested and QE rejects `vdw_corr`, surface a clear log/UI message (PBE-only fallback only if user confirms — default = fail the relax step with explanation).

## P2 — Selective dynamics

### UI

- Combo **Fix atoms during relax**: `None` | `Fix reference layer` | `Fix bottom layer (min z)`  
- Requires `layer_tag` site property from stack rebuild when available; if missing tags, disable ref-layer option and show tip.

### QE

- Emit `ATOMIC_POSITIONS {crystal}` with `if_pos` 0/1 flags (QE relax).  
- Fixed layer: all `if_pos = 0 0 0`; free layer: `1 1 1`.  
- Document: fixing ref layer ≈ “hold substrate, relax adsorbate.”

## P3 — Push strain / auto moiré

Extends `2026-09-13-stack-dft-cell-rebuild-design.md` (do not break aligned N-layer path).

### Behavior

1. **Commensurate twist (2 layers):** prefer **moiré supercell tiling** (already partially implemented); do **not** silently fall back to 1×1 forced strain when `status == commensurate`.  
2. **Incommensurate:** keep warn dialog; show **engineering % strain on non-ref layer** (lattice constant), not only the Frobenius+twist metric that can read ~50% at 30°.  
3. Dialog copy: state clearly that bands are approximate until relax; point to QE Relax option.  
4. Optional: “Prefer larger commensurate search” knobs stay out unless already present — no new heavy search in P3 beyond fixing messaging + ensuring commensurate path is used when status says so.

### Strain metric clarity

- UI shows both:  
  - **Isotropic match strain** ≈ `(a_ref − a_native) / a_native` on the strained layer.  
  - **Internal score** used for ref suggestion (may include twist) — labeled as “suggestion score,” not “% stretch.”

## File touch list (expected)

| Area | Files |
|------|--------|
| QE core | `tensorspec/core/dft/qe_generator.py` |
| QE UI + pipeline | `tensorspec/gui/components/qe_generator_panel.py` |
| Fetch / write-back | QE panel and/or small helper under `tensorspec/core/dft/` |
| Stack Push | `tensorspec/gui/suites/crystal_suite.py`, `tensorspec/core/crystallography.py` |
| Tests | `tests/test_qe_relax_*.py`, extend `tests/test_stack_dft_cell.py` |

## Success criteria

1. Generate with default **Relax ions** produces `relax.in` + pipeline that relaxes then SCF→Wannier on Einstein/local.  
2. With DFT-D3 on, `relax.in` contains `vdw_corr` (or explicit unsupported error).  
3. After success, `relaxed_structure.cif` exists and loads in Crystal Suite.  
4. P2: fixing bottom/ref layer yields `if_pos` in `relax.in` and those atoms do not move beyond threshold in a tiny fixture test.  
5. P3: commensurate graphene-like twist uses moiré cell; incommensurate dialog shows ~1.6%-style isotropic strain for Gr/hBN, not ~50% as “stretch.”  
6. Graphene/hBN 0° dry-run path can be re-run with relax+D3 for comparison (manual validation, not CI).

## Risks

| Risk | Mitigation |
|------|------------|
| Relax walltime ≫ SCF | Document; allow SCF-only; keep ranks UI |
| D3 missing in QE build | Detect; clear error |
| vc-relax eats vacuum | Advanced label; prefer ionic default |
| Parsing relax.out brittle | Prefer data-file XML / `ATOMIC_POSITIONS` final block; test fixtures |
| Moiré cells huge | Existing SC≠1×1 reject for twist; commensurate n_cells warning if huge |

## Implementation phases

| Phase | Deliverable |
|-------|-------------|
| **P1** | Geometry combo, vdW checkbox, `relax.in`, pipeline order, geometry sync, `relaxed_structure.cif` |
| **P2** | Selective dynamics UI + `if_pos` |
| **P3** | Push dialog strain clarity + commensurate path enforcement / messaging |

## Open points (resolve in plan if needed)

1. ~~Exact QE version string / D3 keyword on Einstein `qe` conda env~~ — **Resolved 2026-09-18 (Task 10 smoke):** PWSCF **v7.5** at `/home/sandy/miniconda3/envs/qe/bin/pw.x` (conda env `qe`; not on default SSH PATH — `conda activate qe` or full path). **DFT-D3: yes.** Binary embeds `dft-d3` / `DFT-D3` (Grimme); `vdw_corr = 'dft-d3'` in `&SYSTEM` is accepted (same spelling as TensorSpec generator). One-atom C relax smoke with `/home/sandy/TensorSpec/Pseudo/C.us.pbe.z_4.uspp.gbrv.v1.2.upf` printed `DFT-D3 Dispersion Correction (3-body terms):` and C6 table — no unsupported-keyword error.  
2. Whether post-relax SCF reuses relax charge density (`startingwfc`/`startingpot`) — optimization, not required for correctness.  
3. Auto-push relaxed structure to workspace vs file-only + button — default **file + optional button** to avoid surprising Crystal state.
