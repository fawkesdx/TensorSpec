# Bare ARPES: no Fermi–Dirac cutoff

**Date:** 2026-09-06  
**Status:** implemented  
**Branch:** `TensorSpec_GUI`

## Behaviour

- **Bare Spectral Function (ME Off):** Lorentzian spectral weight only — no Fermi–Dirac occupation factor.
- **Full / dipole ME (Chinook or Grizzly `spectral()`):** unchanged; Fermi–Dirac remains inside ME spectral assembly.

## Change

`tensorspec/core/arpes/one_step/chinook_arpes_kmesh.py` — `_bare_intensity` drops `fermi_dirac` multiply.

## Test

`tests/test_bare_arpes_no_fermi_cutoff.py` — unoccupied band at +0.5 eV retains intensity.
