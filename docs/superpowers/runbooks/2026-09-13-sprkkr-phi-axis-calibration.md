# NOT DONE: SPR-KKR PHI axis calibration (phase 1a baseline)

## Problem

SPR-KKR's `SPEC_EL PHI` is measured from an in-plane reference axis **x** that `ase2sprkkr` picks when it slices the bulk potential by `MILLER_HKL`. The Oct 2023 SPR-KKR manual provides no formula — it only states "`x`" without pinning what it is.

Phase 1a (pointwise-deflector mode) **assumes**: at sample azimuth 0, the lab analyzer slit direction == SPR-KKR's `x` axis. This assumption is encoded as `phi_offset_deg = 0.0` (the "PHI zero offset (UNCALIBRATED)" spinbox in the GUI).

This runbook, when executed, will calibrate that offset for the first material and surface. Once calibrated, set `phi_offset_deg` to the result, and azimuth becomes free.

See design doc §7: `phi_e's reference axis (x) is still not pinned down`.

## Procedure

Two cheap reference runs on VTe2, deflector off (0 deg), wide in energy but coarse in angle to keep cost down. Both are **non-pointwise** (no `--pointwise` flag) because this step measures the raw, fixed-PHI axis itself.

### Run A: PHI = 0

```bash
python3 scripts/sprkkr/sprkkr_e2e.py \
  --pot scratch/sprkkr_gui_run/scf_20260909_205106/scf.pot_new \
  --hkl 0 1 -1 --iq-surf 2 \
  --hv 84 --pol P --theta-ph 45 --ework 4.5 \
  --theta -15 15 21 --erange -1.0 0.1 8 \
  --phi 0 0 1 \
  --deflector 0.0 --slit 0.0 --manip-theta 0.0 --tilt 0.0 --azimuth 0.0 \
  --out /tmp/sprkkr_phi_calib_run_a \
  --dry-run
```

When the binary is ready, remove `--dry-run` to execute. Output: `.spc` file in the run directory.

### Run B: PHI = 90

```bash
python3 scripts/sprkkr/sprkkr_e2e.py \
  --pot scratch/sprkkr_gui_run/scf_20260909_205106/scf.pot_new \
  --hkl 0 1 -1 --iq-surf 2 \
  --hv 84 --pol P --theta-ph 45 --ework 4.5 \
  --theta -15 15 21 --erange -1.0 0.1 8 \
  --phi 90 90 1 \
  --deflector 0.0 --slit 0.0 --manip-theta 0.0 --tilt 0.0 --azimuth 0.0 \
  --out /tmp/sprkkr_phi_calib_run_b \
  --dry-run
```

When the binary is ready, remove `--dry-run` to execute. Output: `.spc` file in the run directory.

## Comparison

Overlay both runs' intensity `I_tot(E, k_par)` in the same plot. Compare against:

1. **Chinook baseline** (B1/GrizzlyME data on the same VTe2 surface, same geometry): look at a cut along the Gamma-K and Gamma-M directions. Record the expected band positions (E, k_par) at a few scattered points as ground truth.

2. **Known fact** (SPRKKR_GUI_THREAD.md): `PHI=0 -> Gamma-K` (CIF b axis), `PHI=90 -> Gamma-M`.

Whichever run (A or B) reproduces the Gamma-K pattern from Chinook is the correct zero. If run A matches K, then `phi_offset_deg = 0.0` is right. If run B matches K, then `phi_offset_deg = -90.0` (or equivalently 270.0).

## Result

Once both runs finish and are compared:

1. Determine which `PHI` value matches Gamma-K. Note that value as `CORRECT_PHI_ZERO`.
2. Compute offset: `phi_offset_deg = CORRECT_PHI_ZERO`.
3. **Set in GUI**: open ARPES panel, B3 group, spinbox "PHI zero offset (UNCALIBRATED)" -> enter the value.
4. **Set in CLI default**: in `scripts/sprkkr/sprkkr_e2e.py`, line 157, change `default=0.0` to `default=<CORRECT_PHI_ZERO>`.
5. **Record** in a commit message or project note:
   - Material: VTe2
   - Surface: `hkl_abas = (0, 1, -1)`, `iq_surf = 2`
   - Pot: `scratch/sprkkr_gui_run/scf_20260909_205106/scf.pot_new`
   - `phi_offset_deg = <value>`
   - Run dates and `.spc` file names for archive.

After this calibration, **azimuth is free** — the calibration was done at azimuth 0, so any other azimuth value rotates from that known reference.

## Status

**This runbook is NOT EXECUTED YET.** Sandy will decide when to run it. This is the recipe, not the result.
