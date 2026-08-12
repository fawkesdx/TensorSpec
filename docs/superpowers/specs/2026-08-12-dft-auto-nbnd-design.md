# DFT Suite — Auto-suggest nbnd — Design Spec

Date: 2026-08-12  
Status: approved for planning (user: yes; SOC = 2×)  
Branch: `HTML_einstein_app`  
Related: old Qt `tensorspec/gui/suites/dft_suite.py` `load_workspace_structure` (removed Phase 5)

## Problem

Old desktop DFT suite, on loading a workspace crystal, computed a Wannier/QE band count from atomic sites and wrote it into **Number of Bands (nbnd)**. HTML DFT Suite left `#qe-nbnd` at the static default **12**, so large cells (e.g. VTe2 CDW) look wrong until the user edits by hand. Slab QE auto-hint was ported; nbnd was not.

## Goals

- Restore per-site orbital count from the old GUI.
- Apply SOC doubling when **Inject SOC** is checked (HTML enhancement vs old GUI).
- Update the nbnd field when the selected structure changes and when SOC is toggled.
- Keep the value editable afterward; next structure/SOC change overwrites again (same spirit as old auto-set).

## Non-goals

- Changing TB orbital basis / projection dropdown population beyond what HTML already does.
- Auto-editing Wannier mode or `num_wann` separately from `nbnd` (QE path already uses `nbnd` for `num_wann`).
- Blocking modal dialogs (no Qt `QMessageBox`).

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Formula | Per site: transition metal **or** Z>30 → +9 (s+p+d); else +4 (s+p). Sum = `base`. |
| SOC | UI shows `min(500, 2 * base)` when SOC checked; else `base`. |
| Architecture | Core helper + `suggest_nbnd` on structure list API; UI applies ×1/×2. |
| Cap | Clamp to schema max 500. |
| UX | Fill `#qe-nbnd` + short status text (no popup). |

## §1 — Core

Add e.g. `tensorspec/core/dft/nbnd_suggest.py` (or small function in an existing dft helper module):

```text
suggest_nbnd_base(structure) -> int
```

- Iterate `structure` sites; use pymatgen `specie` (`is_transition_metal`, `number` / Z).
- Return sum; never less than 1.

Unit tests: graphene (2×C → 8); one transition-metal site → 9; Z>30 non-TM if needed.

## §2 — API

Extend structure listing used by DFT Suite (e.g. `StructureOption` / `/api/dft/structures`):

- Add `suggest_nbnd: int` (= base, **without** SOC factor).

SOC factor stays in the UI so toggling the checkbox does not require a refetch.

## §3 — UI

In `dft_suite.js` (mirror `syncSlabSuggestion`):

- On structure select / refresh: set `#qe-nbnd` from `suggest_nbnd` × (2 if SOC else 1), clamp 1…500.
- On SOC checkbox change: re-apply from current structure’s `suggest_nbnd`.
- Status line: e.g. `Suggested nbnd=72 (36×2 SOC)` or `Suggested nbnd=36`.

## §4 — Success criteria

- Selecting VTe2 CDW (or any multi-site crystal) updates nbnd away from 12 when base ≠ 12.
- Toggling SOC doubles / halves the suggestion (within cap).
- Tests cover base formula; manual nbnd still accepted on Generate/Queue.
- Ship on `HTML_einstein_app` only; push + Einstein; never merge to main.

## Spec self-review

- Locked decisions have no TBD.
- Scope = suggest + wire UI/API; no Wannier/TB redesign.
- Matches recovered Qt formula + approved SOC×2.
