# QE Fermi UI (separate from On-site E)

**Date:** 2026-09-05  
**Status:** approved — implemented on `TensorSpec_GUI`  
**Branch:** `TensorSpec_GUI`

## Problem

Loading a third-party Wannier `hr.dat` often leaves Fermi at 0 in the GUI because detection only looks for `scf.out` / `nscf.out`. Collaborator packs may ship only a note file (`FERMI_ENERGY.txt`). Users then try to stuff QE EF into **On-site E**, which is the wrong physical knob (and is capped at ±10 eV).

## Goal

Expose an explicit, editable **QE Fermi (eV)** control, auto-filled when possible, never confused with On-site E. After EF is folded into H at Prepare/parse, ARPES energy shift stays 0; the UI still shows the QE EF so the user is not blind.

## Non-goals

- Changing QE pipeline output format on Perlmutter
- Treating `FERMI_ENERGY.txt` as the only or primary standard for TensorSpec-generated runs
- Using On-site E as a Fermi substitute

## UI

In the TB / Wannier panel (`dft_panels.py`), next to On-site E:

| Control | Role |
|---|---|
| **QE Fermi (eV)** | Absolute energy zero of Wannier H (from QE). Range ~±100 eV. Editable always. |
| **On-site E (eV)** | Extra rigid shift after EF fold. Unchanged. |
| Status label | e.g. `QE EF: 12.4202 eV (from FERMI_ENERGY.txt)` or `(manual)` / `(none — enter if known)` |

On **Load wannier90_hr.dat**, auto-fill QE Fermi from the folder beside `hr.dat` (see detection). User may override. Changing the spinbox marks source as manual until the next load (or an explicit “re-detect” is unnecessary for v1).

## Detection order (auto-fill)

Shared helper (core, not GUI-only), e.g. `detect_qe_fermi_eV(work_dir) -> (value, source)`:

1. `nscf.out` — line containing `the Fermi energy is`
2. `scf.out` — same
3. `FERMI_ENERGY.txt` — same QE-style line, or a bare float (collaborator convenience)
4. else `(0.0, "none")`

Standard TensorSpec QE runs keep using outs. Friend packs work via (3) or manual entry.

## Semantics (critical)

When preparing / parsing Wannier H:

- Fold **QE Fermi** into onsite hoppings: `H_ii -= E_F` (plus On-site E as today).
- Band plot and ARPES use **EF-relative** eigenvalues; do **not** subtract QE Fermi again after fold.
- Workspace / ARPES push metadata:
  - `fermi_energy` / `qe_fermi_eV` = the QE value (for display)
  - `h_includes_qe_fermi_shift` = True after successful W90 prepare with that fold
  - `arpes_e_fermi_shift` = **0.0** when H already includes the fold

Fix any path that currently double-subtracts (parse fold **and** `eigenvalues -= fermi_energy` from the same QE value).

## Code touchpoints

- `tensorspec/core/dft/` — shared `detect_qe_fermi_eV`; `chinook_tb` parse uses explicit EF arg or detected value (prefer UI-passed value when provided)
- `tensorspec/gui/components/dft_panels.py` — spinbox + status label; autofill on load
- `tensorspec/gui/suites/dft_suite.py` — replace `_fermi_energy_from_w90`-only usage with panel value; fix double-shift
- `tb_remote_client` / `tb_remote_runner` — job carries `qe_fermi_eV`; remote parse uses it; result eigenvalues already EF-relative (no second subtract)
- Tests for detect order + no double-shift

## Acceptance

1. Load `qe_workspace/wannier90_hr.dat` → QE Fermi spinbox shows **12.4202**, label cites `FERMI_ENERGY.txt`.
2. On-site E remains 0 by default; changing it does not change the QE Fermi field.
3. User can clear/override QE Fermi manually; Prepare uses the spinbox value.
4. After Prepare/bands, ARPES push reports QE EF for info and `arpes_e_fermi_shift == 0`.
5. Folder with only `scf.out` / `nscf.out` still autofills (no regression).

## Out of scope for follow-ups

- Syncing Einstein after merge (deploy note only)
- Perlmutter changes (not required for this path)
